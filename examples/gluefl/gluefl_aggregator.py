# -*- coding: utf-8 -*-
import sys

from examples.gluefl.gluefl_client_manager import GlueflClientManager
import fedscale.core.channels.job_api_pb2 as job_api_pb2
from fedscale.core import commons
from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core.logger.aggragation import *
from fedscale.utils.compressor.topk import TopKCompressor
from fedscale.utils.eval.round_evaluator import RoundEvaluator
from fedscale.utils.eval.sparsification import Sparsification

class GlueFL_Aggregator(Aggregator):
    """Feed aggregator using tensorflow models"""
    def __init__(self, args):
        super().__init__(args)
        self.dataset_total_worker = args.dataset_total_worker # to distinguish between self.args.total_worker which is the total worker in a round
        # TODO make this be the len of client_profiles (or derived from it)
        self.num_participants = args.num_participants

        self.total_mask_ratio = args.total_mask_ratio  # = shared_mask + local_mask
        self.shared_mask_ratio = args.shared_mask_ratio
        self.regenerate_epoch = args.regenerate_epoch
        self.max_prefetch_round = args.max_prefetch_round
        
        self.sampling_strategy = args.sampling_strategy
        self.sticky_group_size = args.sticky_group_size
        self.sticky_group_change_num = args.sticky_group_change_num
        self.sampled_sticky_client_set = []
        self.sampled_sticky_clients = []
        self.sampled_changed_clients = []

        self.fl_method = args.fl_method

        self.compressed_gradient = None
        self.mask_record_list = []
        self.shared_mask = []
        self.apf_ratio = 1.0

        self.last_update_index = []
        self.round_lost_clients = []
        self.clients_to_run = []
        self.slowest_client_id = -1
        self.round_evaluator = RoundEvaluator()
        # logging.info("Good Init")

    def init_client_manager(self, args):
        """ Initialize Gluefl client sampler
        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
        
        Returns:
            GlueflClientManager: The client manager class
        """

        # sample_mode: random or oort
        client_manager = GlueflClientManager(args.sample_mode, args=args)

        return client_manager

    def init_mask(self):
        self.shared_mask = []
        for idx, param in enumerate(self.model.state_dict().values()):
            self.shared_mask.append(torch.zeros_like(param, dtype=torch.bool).to(dtype=torch.bool))

    def client_register_handler(self, executorId, info):
        """Triggered once after all executors have registered
        
        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info['size']:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (self.num_of_clients+1) % len(
                self.client_profiles) if len(self.client_profiles) > 0 else 1
            systemProfile = self.client_profiles.get(
                mapped_id, {'computation': 1.0, 'communication': 1.0, 'dl_kbps': 1.0, 'ul_kbps': 1.0})
            # logging.info(systemProfile)
            clientId = (
                self.num_of_clients+1) if self.experiment_mode == commons.SIMULATION_MODE else executorId
            self.client_manager.register_client(
                executorId, clientId, size=_size, speed=systemProfile)
            self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                                                 upload_step=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)
            self.num_of_clients += 1

        logging.info("Info of all feasible clients {}".format(
            self.client_manager.getDataInfo()))

    def run(self):
        """Start running the aggregator server by setting up execution 
        and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.init_mask()
        self.save_last_param()
        self.model_update_size = sys.getsizeof(
            pickle.dumps(self.model))/1024.0*8.  # kbits
        self.model_bitmap_size = self.model_update_size / 32
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file)
        self.last_update_index = [0 for _ in range(self.dataset_total_worker * 2)] # FIXME
        self.event_monitor()

    def get_shared_mask(self):
        """Get shared mask that would be used by all FL clients (in default FL)

        Returns:
            List of PyTorch tensor: Based on the executor's machine learning framework, initialize and return the mask for training.

        """
        return [p.to(device='cpu') for p in self.shared_mask]

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.
        
        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            tuple: Return the sampled clients and client execution information in the last round.

        """

        if len(sampled_clients) == 0:
            return [], [], [], {}, 0, []

        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            sampledClientsLost = []
            completionTimes = []
            virtual_client_clock = {}
            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)
                exe_cost = {'computation': 0, 'downstream': 0, 'upstream': 0, 'round_duration': 0}
                if self.fl_method == "FedAvg":
                    exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                                                 batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
                                                                 upload_size=self.model_update_size, download_size=self.model_update_size)
                elif self.fl_method == "STC":
                    l = self.last_update_index[client_to_run]
                    r = self.round - 1
                    downstream_update_ratio = Sparsification.check_model_update_overhead(l, r, self.model, self.mask_record_list, self.device, use_accurate_cache=True)
                    dl_size = min(self.model_update_size * downstream_update_ratio + self.model_bitmap_size, self.model_update_size)
                    ul_size = self.total_mask_ratio * self.model_update_size + self.model_bitmap_size

                    exe_cost = self.client_manager.getCompletionTime(client_to_run, batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                    self.round_evaluator.recordClient(client_to_run, dl_size, ul_size, exe_cost)
                elif self.fl_method == "APF":
                    l = self.last_update_index[client_to_run]
                    r = self.round - 1
                    # logging.info(f"{l} and {r}")

                    downstream_update_ratio = Sparsification.check_model_update_overhead(l, r, self.model, self.mask_record_list, self.device, use_accurate_cache=True)
                    if self.last_update_index[client_to_run] == 0:
                        downstream_update_ratio = 1.

                    dl_size = min(self.model_update_size, self.model_bitmap_size + self.model_update_size * downstream_update_ratio)
                    ul_size = self.apf_ratio * self.model_update_size

                    exe_cost = self.client_manager.getCompletionTime(client_to_run, batch_size=client_cfg.batch_size,
                                                            upload_step=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                    self.round_evaluator.recordClient(client_to_run, dl_size, ul_size, exe_cost)
                elif self.fl_method == "GlueFL":
                    l = self.last_update_index[client_to_run]
                    r = self.round - 1

                    downstream_update_ratio = Sparsification.check_model_update_overhead(l, r, self.model, self.mask_record_list, self.device, use_accurate_cache=True)
                    dl_size = min(self.model_update_size * downstream_update_ratio + self.model_bitmap_size, self.model_update_size)
                    ul_size = self.total_mask_ratio * self.model_update_size + min((self.total_mask_ratio - self.shared_mask_ratio) * self.model_update_size, self.model_bitmap_size)
                    
                    exe_cost = self.client_manager.getCompletionTime(client_to_run, batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                    dl_bw, ul_bw = self.client_manager.getBwInfo(client_to_run)
                    self.round_evaluator.recordClient(client_to_run, dl_size, ul_size, exe_cost, dl_bw, ul_bw)


                roundDuration = exe_cost['computation'] + exe_cost['downstream'] + exe_cost['upstream']
                exe_cost['round'] = roundDuration
                virtual_client_clock[client_to_run] = exe_cost
                self.last_update_index[client_to_run] = self.round - 1

                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                else:
                    sampledClientsLost.append(client_to_run)

            num_clients_to_collect = min(
                num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            sortedWorkersByCompletion = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k])
            top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [sampledClientsReal[k]
                             for k in sortedWorkersByCompletion[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            # return TictakResponse(clients_to_run, dummy_clients, sampledClientsLost, virtual_client_clock, round_duration, completionTimes)
            slowest_client_id = sampledClientsReal[top_k_index[-1]]
            return (clients_to_run, dummy_clients, sampledClientsLost,
                    virtual_client_clock, round_duration,
                    completionTimes[:num_clients_to_collect])
        else:
            virtual_client_clock = {
                client: {'computation': 1, 'communication': 1} for client in sampled_clients}
            completionTimes = [1 for c in sampled_clients]
            # return TictakResponse(sampled_clients, sampled_clients, [], virtual_client_clock, 1, compl)
            return (sampled_clients, sampled_clients, [], virtual_client_clock,
                    1, completionTimes, -1)

    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, 
        if so, we need to append results to the cache
        
        Args:
            results (dictionary): client's training result
        
        """
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'update_gradient': gradient_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.register_feedback(results['clientId'], results['utility'],
                                          auxi=math.sqrt(
                                              results['moving_loss']),
                                          time_stamp=self.round,
                                          duration=self.virtual_client_clock[results['clientId']]['computation'] +
                                          self.virtual_client_clock[results['clientId']
                                                                    ]['communication']
                                          )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()
        # download_ratio = check_model_update_overhead(0, self.round - 1, self.model, self.mask_record_list)
        # logging.info(f"Download sparsification: {results['clientId']} {download_ratio}")

        self.model_in_update += 1
        self.aggregate_client_weights(results)

        # logging.info("Good 1")
        # All clients are done
        if self.model_in_update == self.tasks_round:
            if self.fl_method == "APF":
                logging.info("APF server")
                if self.round <= 1:
                    # initialize APF
                    self.ema = []
                    self.ema_abs = []
                    self.l_frez = []
                    self.i_frez = []
                    
                    for idx, param in enumerate(self.model.state_dict().values()):
                        self.ema.append(torch.zeros_like(param).to(device=self.device))
                        self.ema_abs.append(torch.zeros_like(param).to(device=self.device))
                        self.l_frez.append(torch.zeros_like(param).to(torch.float32).to(device=self.device))
                        self.i_frez.append(torch.zeros_like(param).to(torch.float32).to(device=self.device))
                        self.shared_mask[idx] |= True

                F_c = 5
                free_thd = 0.3
                alpha = 0.99

                # update mask 
                if self.round % F_c == 0:
                    # update ema
                    tot_free = []
                    for idx, param in enumerate(self.model.state_dict().values()):
                        
                        # unfreeze 
                        self.ema[idx] = alpha * self.ema[idx] + (1 - alpha) * self.compressed_gradient[idx]
                        self.ema_abs[idx] = alpha * self.ema_abs[idx] + (1 - alpha) * self.compressed_gradient[idx].abs()
                        self.ep = (self.ema[idx].abs() / self.ema_abs[idx]).to(device=self.device)
                        
                        self.free_mask = (self.ep < free_thd).to(torch.bool).to(device=self.device)
                        tot_free.append(self.free_mask)
                        self.shared_mask[idx] = self.shared_mask[idx].to(device=self.device)
                        self.false_froze = (self.free_mask & (self.shared_mask[idx])).to(device=self.device) # mask_model ^ True - free is false
                        self.false_nofro = (self.false_froze ^ (self.shared_mask[idx])).to(device=self.device)
                        self.l_frez[idx] += (self.false_froze * F_c).to(torch.float32)
                        self.l_frez[idx][self.false_nofro] /= 2.0
                        self.i_frez[idx][self.shared_mask[idx]] = self.round + self.l_frez[idx][self.shared_mask[idx]]
                        
                        # self.new_froze = ((self.round + F_c) <= self.i_frez).to(torch.bool) & (self.mask_model ^ True)
                        self.shared_mask[idx] = (self.i_frez[idx] < (self.round + F_c)).to(torch.bool)
            else:
                self.apply_and_update_mask()

            spar_ratio = Sparsification.check_sparsification_ratio([self.compressed_gradient])
            mask_ratio = Sparsification.check_sparsification_ratio([self.shared_mask])
            self.apf_ratio = mask_ratio
            # ovlp_ratio = check_sparsification_ratio([self.overlap_gradient])
            logging.info(f"Gradients sparsification: {spar_ratio}")
            logging.info(f"Mask sparsification: {mask_ratio}")

            # ==== update global model =====
            model_state_dict = self.model.state_dict()
            for idx, param in enumerate(model_state_dict.values()):
                # logging.info(f"weight: {idx} {self.compressed_gradient[idx]}")
                param.data = param.data.to(device=self.device).to(dtype=torch.float32) - self.compressed_gradient[idx]
            self.model.load_state_dict(model_state_dict)
            
            # ===== update mask list =====
            mask_list = []
            for p_idx, key in enumerate(self.model.state_dict().keys()):
                mask = (self.compressed_gradient[p_idx] != 0).to(device=torch.device("cpu"))
                mask_list.append(mask)

            self.mask_record_list.append(mask_list)
        self.update_lock.release()

    def check_sparsification_ratio(self, global_g_list):
        worker_number = len(global_g_list)
        spar_ratio = 0.

        total_param = 0
        for g_idx, g_param in enumerate(global_g_list[0]):
            total_param += len(torch.flatten(global_g_list[0][g_idx]))

        for i in range(worker_number):
            non_zero_param = 0
            for g_idx, g_param in enumerate(global_g_list[i]):
                mask = g_param != 0.
                # print(mask)
                non_zero_param += float(torch.sum(mask))

            spar_ratio += (non_zero_param / total_param) / worker_number

        return spar_ratio

    def aggregate_client_weights(self, results):
        """May aggregate client updates on the fly
        """

        # ===== initialize compressed gradients =====
        if self.model_in_update == 1:
            self.compressed_gradient = [torch.zeros_like(param.data).to(device=self.device).to(dtype=torch.float32) for param in self.model.state_dict().values()]
        
        prob = self.get_gradient_weight(results['clientId'])
        # prob = (1.0 / self.tasks_round)
        # logging.info(f"weight: {self.get_gradient_weight(results['clientId'])} {results['clientId']}")
        keys = [] 
        for idx, key in enumerate(self.model.state_dict()):
            keys.append(key)
        for idx, param in enumerate(self.model.state_dict().values()):
            if not (('num_batches_tracked' in keys[idx]) or ('running' in keys[idx])):
                self.compressed_gradient[idx] += (torch.from_numpy(results['update_gradient'][idx]).to(device=self.device) * prob)
            else:
                self.compressed_gradient[idx] += (torch.from_numpy(results['update_gradient'][idx]).to(device=self.device) * (1.0/self.tasks_round))
            
    def get_gradient_weight(self, clientId):
        prob = 0
        if self.sampling_strategy == "STICKY":
            if self.round <= 1:
                prob = (1.0 / float(self.tasks_round))
            elif clientId in self.sampled_sticky_client_set:
                prob = (1.0 / float(self.dataset_total_worker)) * (1.0 / ((float(self.tasks_round) - float(self.sticky_group_change_num)) / float(self.sticky_group_size)))
            else:
                prob = (1.0 / float(self.dataset_total_worker)) * (1.0 / (float(self.sticky_group_change_num) / (float(self.dataset_total_worker) - float(self.sticky_group_size))))
        else:
            """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
            """
            # Importance of each update is 1/#_of_participants
            # importance = 1./self.tasks_round
            prob = 1.0 / self.tasks_round
        return prob

    def apply_and_update_mask(self):
        compressor_tot = TopKCompressor(compress_ratio=self.total_mask_ratio)
        compressor_shr = TopKCompressor(compress_ratio=self.shared_mask_ratio)

        keys = [] 
        for idx, key in enumerate(self.model.state_dict()):
            keys.append(key)
        
        for idx, param in enumerate(self.model.state_dict().values()):
            if (('num_batches_tracked' in keys[idx]) or ('running' in keys[idx])):
                continue
            
            # --- STC ---
            if self.fl_method == "STC" or self.round % self.regenerate_epoch == 1:
                # local mask
                self.compressed_gradient[idx], ctx_tmp = compressor_tot.compress(
                    self.compressed_gradient[idx])
                
                self.compressed_gradient[idx] = compressor_tot.decompress(self.compressed_gradient[idx], ctx_tmp)
            else:
                # shared + local mask
                max_value = float(self.compressed_gradient[idx].abs().max())
                update_mask = self.compressed_gradient[idx].clone().detach()
                update_mask[self.shared_mask[idx] == True] = max_value
                update_mask, ctx_tmp = compressor_tot.compress(update_mask)
                update_mask = compressor_tot.decompress(update_mask, ctx_tmp)
                update_mask = update_mask.to(torch.bool)
                self.compressed_gradient[idx][update_mask != True] = 0.0

        # --- update shared mask ---
        for idx, param in enumerate(self.model.state_dict().values()):
            if (('num_batches_tracked' in keys[idx]) or ('running' in keys[idx])):
                continue

            determined_mask = self.compressed_gradient[idx].clone().detach()
            determined_mask, ctx_tmp = compressor_shr.compress(determined_mask)
            determined_mask = compressor_shr.decompress(determined_mask, ctx_tmp)
            self.shared_mask[idx] = determined_mask.to(torch.bool)

    def create_client_task(self, executorId):
        """Issue a new client training task to specific executor
        
        Args:
            executorId (int): Executor Id.
        
        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_clientId = self.resource_manager.get_next_task(executorId)
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        model = None
        if next_clientId != None:
            config = self.get_client_conf(next_clientId)
            train_config = {'client_id': next_clientId, 
                            'task_config': config, 
                            'agg_weight': (self.get_gradient_weight(next_clientId) * float(self.dataset_total_worker)),
                            'sticky_client_flag': (next_clientId in self.sampled_sticky_client_set)}
        return train_config, model
    
    def CLIENT_PING(self, request, context):
        """Handle client ping requests
        
        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            current_event = self.individual_client_events[executor_id].popleft(
            )
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    client_id)
                
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                                commons.CLIENT_TRAIN)
            elif current_event == commons.MODEL_TEST:
                response_msg = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                response_data = self.get_global_model()
            elif current_event == commons.UPDATE_MASK:
                response_data = self.get_shared_mask()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(event=current_event,
                                          meta=response_msg, data=response_data)
        if current_event != commons.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")
        
        return response

    def select_participants(self, select_num_participants, overcommitment=1.0):
        return sorted(self.client_manager.select_participants(
            round(select_num_participants*overcommitment),
            cur_time=self.global_virtual_clock),
        )


    # def select_participants_sticky(self, select_num_participants, overcommitment=1.0):
    #     self.sampled_sticky_clients, self.sampled_changed_clients = self.client_manager.select_participants_sticky(
    #             round(select_num_participants*overcommitment),
    #             cur_time=self.global_virtual_clock,
    #             K=self.sticky_group_size,
    #             change_num=round(self.sticky_group_change_num*overcommitment)
    #         )
    #     self.sampled_sticky_client_set = set(self.sampled_sticky_clients)
    #     return self.sampled_sticky_clients, self.sampled_changed_clients


    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        self.global_virtual_clock += self.round_duration
        self.round += 1

        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_gradient_weights)

        avgUtilLastround = sum(self.stats_util_accumulator) / \
            max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.register_feedback(clientId, avgUtilLastround,
                                              time_stamp=self.round,
                                              duration=self.virtual_client_clock[clientId]['computation'] +
                                              self.virtual_client_clock[clientId]['communication'],
                                              success=False)

        avg_loss = sum(self.loss_accumulator) / \
            max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: " +
                     f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        if self.round > 1:
            self.round_evaluator.recordRoundCompletion(self.clients_to_run, self.round_stragglers + self.round_lost_clients, self.slowest_client_id)
            logging.info(f"Cumulative bandwidth usage:\n \
            total (excluding overcommit): {self.round_evaluator.total_bandwidth:.2f} kbit\n \
            total (including overcommit): {self.round_evaluator.total_bandwidth + self.round_evaluator.total_overcommit_bandwidth:.2f} kbit\n \
            downstream: {self.round_evaluator.total_bandwidth_dl:.2f} kbit\tupstream: {self.round_evaluator.total_bandwidth_ul:.2f} kbit\tprefetch: {self.round_evaluator.total_bandwidth_schedule:.2f} kbit\tovercommit: {self.round_evaluator.total_overcommit_bandwidth:.2f} kbit")
            logging.info(f"Cumulative round durations:\n \
            (wall clock time) total:\t{self.round_evaluator.total_duration:.2f} s\n \
            total_dl:\t{self.round_evaluator.total_duration_dl:.2f} s\t \
            total_ul:\t{self.round_evaluator.total_duration_ul:.2f} s\t \
            total_compute:\t{self.round_evaluator.total_duration_compute:.2f} s\n \
            avg_dl:\t{self.round_evaluator.avg_duration_dl:.2f} s\t \
            avg_ul:\t{self.round_evaluator.avg_duration_ul:.2f} s\t \
            avg_compute:\t{self.round_evaluator.avg_duration_compute:.2f} s\n \
            client_avg_dl:\t{self.round_evaluator.client_avg_duration_dl:.2f} s\t \
            client_avg_ul:\t{self.round_evaluator.client_avg_duration_ul:.2f} s\t \
            client_avg_compute:\t{self.round_evaluator.client_avg_duration_compute:.2f} s \
            ")
            self.round_evaluator.startNewRound()

        # update select participants
        if self.sampling_strategy == "STICKY":
            numOfClients = round(self.args.num_participants*self.args.overcommitment)
            numOfClientsOvercommit = numOfClients - self.args.num_participants
            numOfChangedClients = round(self.args.overcommit_weight * numOfClientsOvercommit) + self.sticky_group_change_num if self.args.overcommit_weight>= 0 else round(self.sticky_group_change_num*self.args.overcommitment)
            self.sampled_sticky_clients, self.sampled_changed_clients = self.client_manager.select_participants_sticky(
                numOfClients,
                cur_time=self.global_virtual_clock,
                K=self.sticky_group_size,
                change_num=numOfChangedClients
            )
            self.sampled_sticky_client_set = set(self.sampled_sticky_clients)

            # Choose fastest online changed clients
            (change_to_run, change_stragglers, change_lost, change_virtual_client_clock, 
            change_round_duration, change_flatten_client_duration) = self.tictak_client_tasks(
                self.sampled_changed_clients, 
                self.args.sticky_group_change_num if self.round > 1 else self.args.num_participants)
            logging.info(f"Selected change participants to run: {sorted(change_to_run)}\nchange stragglers: {sorted(change_stragglers)}\nchange lost: {sorted(change_lost)}")

            # Randomly choose from online sticky clients
            sticky_to_run_count = self.args.num_participants - self.args.sticky_group_change_num
            (sticky_fast, sticky_slow, sticky_lost, sticky_virtual_client_clock, 
            _, sticky_flatten_client_duration) = self.tictak_client_tasks(
                self.sampled_sticky_clients, sticky_to_run_count)
            all_sticky = sticky_fast + sticky_slow
            all_sticky.sort(key=lambda c: sticky_virtual_client_clock[c]['round'])
            all_sticky_faster_than_change = [c for c in all_sticky if sticky_virtual_client_clock[c]['round'] <= change_round_duration]

            if len(all_sticky_faster_than_change) >= sticky_to_run_count:
                sticky_to_run = random.sample(all_sticky_faster_than_change, sticky_to_run_count)
            else:
                extra_sticky_clients = sticky_to_run_count - len(all_sticky_faster_than_change)
                logging.info(f"Sticky needs {extra_sticky_clients} additional clients")
                sticky_to_run = all_sticky_faster_than_change + all_sticky[len(all_sticky_faster_than_change):len(all_sticky_faster_than_change) + extra_sticky_clients]

            if (len(self.sampled_sticky_clients) > 0): # There are no sticky clients in round 1
                slowest_sticky_client_id = max(sticky_to_run, key=lambda k: sticky_virtual_client_clock[k]['round'])
                sticky_round_duration = sticky_virtual_client_clock[slowest_sticky_client_id]['round']
            else:
                sticky_round_duration = 0

            sticky_ignored = [c for c in all_sticky if c not in sticky_to_run]

            logging.info(f"Selected sticky participants to run: {sorted(sticky_to_run)}\nunselected sticky participants: {sorted(sticky_ignored)}\nsticky lost: {sorted(sticky_lost)}")

            # Combine sticky and changed clients together
            self.clients_to_run = sticky_to_run + change_to_run
            self.round_stragglers = sticky_ignored + change_stragglers
            self.round_lost_clients = sticky_lost + change_lost
            self.virtual_client_clock = {**sticky_virtual_client_clock, **change_virtual_client_clock}
            self.round_duration = max(sticky_round_duration, change_round_duration)
            self.flatten_client_duration = numpy.array(sticky_flatten_client_duration + change_flatten_client_duration)
            self.clients_to_run.sort(key=lambda k: self.virtual_client_clock[k]['round']);
            self.slowest_client_id = self.clients_to_run[-1]

            # Make sure that there are change_num number of new clients added each epoch 
            # logging.info(f"Old group {self.client_manager.cur_group}")
            # logging.info(f"change to run {change_to_run}")
            if self.round > 1:
                self.client_manager.cur_group = self.client_manager.cur_group[:-len(change_to_run)] + change_to_run
            # logging.info(f"New group {self.client_manager.cur_group}")
        else:
            self.sampled_participants = self.select_participants(
                select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
            logging.info(f"Sampled clients: {sorted(self.sampled_participants)}")
            (self.clients_to_run, self.round_stragglers, self.round_lost_clients, self.virtual_client_clock, self.round_duration, self.flatten_client_duration) = self.tictak_client_tasks(self.sampled_participants, self.args.num_participants)
            self.slowest_client_id = self.clients_to_run[-1]
            self.flatten_client_duration = numpy.array(self.flatten_client_duration)
        
        logging.info(f"Selected participants to run: {sorted(self.clients_to_run)}\nstragglers: {sorted(self.round_stragglers)}\nlost: {sorted(self.round_lost_clients)}")

        # logging.info("Client to run")
        # for c in self.clients_to_run:
        #     logging.info(f"Client id {c} IsSticky: {c in self.sampled_sticky_client_set}\n\t{self.virtual_client_clock[c]}\n\t{self.round_evaluator.cur_used_bandwidths[c]}")
        # logging.info("Stragglers")
        # for c in self.round_stragglers:
        #     logging.info(f"Client id {c} IsSticky: {c in self.sampled_sticky_client_set}\n\t{self.virtual_client_clock[c]}\n\t{self.round_evaluator.cur_used_bandwidths[c]}")
        # logging.info(f"Slowest client {self.slowest_client_id} {self.virtual_client_clock[self.slowest_client_id]}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(self.clients_to_run)
        self.tasks_round = len(self.clients_to_run)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(
                self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants]

        self.save_last_param()
        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()

                if current_event in (commons.UPDATE_MODEL, commons.MODEL_TEST, commons.UPDATE_MASK):
                    self.dispatch_client_events(current_event)

                elif current_event == commons.START_ROUND:
                    self.dispatch_client_events(commons.CLIENT_TRAIN)

                elif current_event == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.sever_events_queue) > 0:
                client_id, current_event, meta, data = self.sever_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(
                        self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data))

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)

if __name__ == "__main__":
    aggregator = GlueFL_Aggregator(args)
    aggregator.run()