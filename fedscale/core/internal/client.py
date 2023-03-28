
import logging


class Client(object):

    def __init__(self, hostId, clientId, speed, augmentation_factor=3.0, upload_factor=1.0, download_factor=1.0, traces=None):
        self.hostId = hostId
        self.clientId = clientId
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.score = 0
        self.traces = traces
        self.behavior_index = 0
        self.dl_bandwidth = speed['dl_kbps']
        self.ul_bandwidth = speed['ul_kbps']
        self.augmentation_factor = augmentation_factor
        self.upload_factor = upload_factor
        self.download_factor = download_factor

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    def isActive(self, cur_time):
        if self.traces is None:
            return True

        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
            return True

        return False

    def getCompletionTime(self, batch_size, upload_step, upload_size, download_size):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers, 
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        # return {'computation': augmentation_factor * batch_size * upload_step*float(self.compute_speed)/1000.,
        #         'communication': (upload_size+download_size)/float(self.bandwidth)}

        return {'computation':self.augmentation_factor * batch_size * upload_step*float(self.compute_speed)/1000., \
                'communication': (upload_size+download_size)/float(self.bandwidth), 'downstream': download_size/(self.dl_bandwidth * self.download_factor), \
                'upstream': upload_size/(self.ul_bandwidth * self.upload_factor)}

