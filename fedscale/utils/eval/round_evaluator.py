import logging


class RoundEvaluator(object):
    def __init__(self) -> None:
        self.round_durations = [] # Unused
        self.round_durations_aggregated = []
        self.total_duration = 0.
        self.total_duration_dl = 0.
        self.total_duration_ul = 0.
        self.total_duration_compute = 0.
        self.client_avg_duration_dl = 0.
        self.client_avg_duration_ul = 0.
        self.client_avg_duration_compute = 0.
        self.avg_duration_dl = 0.
        self.avg_duration_ul = 0.
        self.avg_duration_compute = 0.
        
        self.bandwidths = []
        self.total_bandwidth = 0.
        self.total_bandwidth_dl = 0.
        self.total_bandwidth_ul = 0.
        self.total_bandwidth_schedule = 0.
        self.total_overcommit_bandwidth = 0. # Bandwidth recording just the extra downstream from overcommitment

        self.epoch = 0

        # Per round
        self.cur_client_count = 0
        self.cur_bandwidth_total = 0.
        self.cur_durations = {} # client_id : { "upstream", "downstream", "computation"}
        self.cur_used_bandwidths = {} # client_id : {"upstream", "downstream", "prefetch"}
        self.cur_clients = [] # Unused all clients in this round

    def startNewRound(self):
        self.cur_client_count = 0
        self.cur_bandwidth_total = 0.
        self.cur_durations = {}
        self.cur_used_bandwidths = {}

    def recordClient(self, client_id, dl_size, ul_size, duration, dl_bw=0, ul_bw=0, prefetch_dl_size = 0):
        self.cur_durations[client_id] = duration
        self.cur_used_bandwidths[client_id] = {"upstream": ul_size, "downstream": dl_size, "prefetch": prefetch_dl_size, "dl_bw": dl_bw, "ul_bw": ul_bw}


    def recordRoundCompletion(self, clients_to_run, dummy_clients, slowest_client_id):
        self.epoch += 1

        # self.round_durations.append(self.cur_durations) # UNUSED

        round_duration = self.cur_durations[slowest_client_id]
        self.total_duration_ul += round_duration["upstream"]
        self.total_duration_dl += round_duration["downstream"]
        self.total_duration_compute += round_duration["computation"]
        self.avg_duration_ul = (self.avg_duration_ul * (self.epoch - 1)) / self.epoch + round_duration["upstream"] / self.epoch
        self.avg_duration_dl = (self.avg_duration_dl * (self.epoch - 1)) / self.epoch + round_duration["downstream"] / self.epoch
        self.avg_duration_compute = (self.avg_duration_compute * (self.epoch - 1)) / self.epoch + round_duration["computation"] / self.epoch

        round_duration_aggregated = round_duration["upstream"] + round_duration["downstream"] + round_duration["computation"]
        self.round_durations_aggregated.append(round_duration_aggregated)
        self.total_duration += round_duration_aggregated
        
        avg_ul_duration, avg_dl_duration, avg_compute_duration = 0, 0, 0
        for id in clients_to_run:
            bw = self.cur_used_bandwidths[id]
            self.total_bandwidth += bw["upstream"] + bw["downstream"] + bw["prefetch"]
            self.total_bandwidth_ul += bw["upstream"]
            self.total_bandwidth_dl += bw["downstream"]
            self.total_bandwidth_schedule += bw["prefetch"]

            avg_ul_duration += self.cur_durations[id]["upstream"]
            avg_dl_duration += self.cur_durations[id]["downstream"]
            avg_compute_duration += self.cur_durations[id]["computation"]
        
        avg_ul_duration /= len(clients_to_run)
        avg_dl_duration /= len(clients_to_run)
        avg_compute_duration /= len(clients_to_run)
        self.client_avg_duration_dl = (self.client_avg_duration_dl * (self.epoch - 1))/ self.epoch + (avg_dl_duration / self.epoch)
        self.client_avg_duration_ul = (self.client_avg_duration_ul * (self.epoch - 1))/ self.epoch + (avg_ul_duration / self.epoch)
        self.client_avg_duration_compute = (self.client_avg_duration_compute * (self.epoch - 1))/ self.epoch + (avg_compute_duration / self.epoch)

        for id in dummy_clients:
            bw = self.cur_used_bandwidths[id]
            self.total_overcommit_bandwidth += bw["downstream"]
        return self.total_bandwidth, self.total_duration