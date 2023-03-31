import logging
from typing import Dict
from examples.gluefl.gluefl_client_metadata import GlueflClientMetadata
from fedscale.core.client_manager import clientManager


class GlueflClientManager(clientManager):
    def __init__(self, mode, args, sample_seed=233):
        super().__init__(mode, args, sample_seed)
        self.Clients: Dict[str, GlueflClientMetadata]= {}
        self.sticky_group = []

    def register_client(self, hostId: int, clientId: int, size: int, speed: Dict[str, float], duration: float=1) -> None:
        """Register client information to the client manager.
        Args: 
            hostId (int): executor Id.
            clientId (int): client Id.
            size (int): number of samples on this client.
            speed (Dict[str, float]): device speed (e.g., compuutation and communication).
            duration (float): execution latency.
        """
        uniqueId = self.getUniqueId(hostId, clientId)
        user_trace = None if self.user_trace is None else self.user_trace[self.user_trace_keys[int(
            clientId) % len(self.user_trace)]]

        self.Clients[uniqueId] = GlueflClientMetadata(hostId, clientId, speed, augmentation_factor=self.args.augmentation_factor, upload_factor=self.args.upload_factor, download_factor=self.args.download_factor, traces=user_trace)

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(clientId)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward': min(size, self.args.local_steps*self.args.batch_size),
                             'duration': duration,
                             }
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)
        else:
            del self.Clients[uniqueId]

    def update_sticky_group(self, new_clients):
        self.rng.shuffle(self.sticky_group)
        self.sticky_group = self.sticky_group[:-len(new_clients)] + new_clients

    # Sticky sampling
    def select_participants_sticky(self, numOfClients, cur_time = 0, K = 0, change_num = 0, overcommit=1.3):
        self.count += 1
    
        logging.info(f"Sticky sampling num {numOfClients} K {K} Change {change_num}")
        clients_online = self.getFeasibleClients(cur_time)
        clients_online_set = set(clients_online)
        # logging.info(f"clients online: {clients_online}")
        if len(clients_online) <= numOfClients:
            return clients_online

        selected_sticky_clients, selected_new_clients = [], []
        if len(self.sticky_group) == 0:
            # initalize the sticky group
            self.rng.shuffle(clients_online)
            client_len = min(K, len(clients_online) -1)
            temp_group = clients_online[:round(client_len * overcommit)]
            temp_group.sort(key=lambda c: min(self.getBwInfo(c)))
            self.sticky_group = temp_group[-client_len:]
            self.rng.shuffle(self.sticky_group)
            # We treat the clients sampled from the first round as the initial sticky clients
            selected_new_clients = self.sticky_group[:min(numOfClients, client_len)]
        else:
            # randomly delete some clients
            self.rng.shuffle(self.sticky_group)
            # find the clients that are available in the sticky group
            online_sticky_group = [i for i in self.sticky_group if i in clients_online_set]
            selected_sticky_clients = online_sticky_group[:(numOfClients - change_num)]
            # randomly include some clients
            self.rng.shuffle(clients_online)
            client_len = min(change_num, len(clients_online)-1)
            selected_new_clients = []
            for client in clients_online:
                if client in self.sticky_group:
                    continue
                selected_new_clients.append(client)
                if len(selected_new_clients) == client_len:
                    break
            
        logging.info(f"Selected sticky clients ({len(selected_sticky_clients)}): {sorted(selected_sticky_clients)}\nSelected new clients({len(selected_new_clients)}) {sorted(selected_new_clients)}")
        return selected_sticky_clients, selected_new_clients


    def getBwInfo(self, clientId):
        return self.Clients[self.getUniqueId(0, clientId)].dl_bandwidth,  self.Clients[self.getUniqueId(0, clientId)].ul_bandwidth