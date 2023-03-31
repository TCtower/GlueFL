import logging
from fedscale.core.client_manager import clientManager


class GlueflClientManager(clientManager):
    def __init__(self, mode, args, sample_seed=233):
        super().__init__(mode, args, sample_seed)
        self.sticky_group = []

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