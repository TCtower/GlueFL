from fedscale.core.internal.client_metadata import ClientMetadata

class GlueflClientMetadata(ClientMetadata):
    def __init__(self, hostId, clientId, speed, augmentation_factor=3.0, upload_factor=1.0, download_factor=1.0, traces=None) -> None:
        super().__init__(hostId, clientId, speed, traces)
        self.dl_bandwidth = speed['dl_kbps']
        self.ul_bandwidth = speed['ul_kbps']
        self.augmentation_factor = augmentation_factor
        self.upload_factor = upload_factor
        self.download_factor = download_factor


    def getCompletionTime(self, batch_size, upload_step, upload_size, download_size):
        return {'computation':self.augmentation_factor * batch_size * upload_step*float(self.compute_speed)/1000., \
                'communication': (upload_size+download_size)/float(self.bandwidth), 'downstream': download_size/(self.dl_bandwidth * self.download_factor), \
                'upstream': upload_size/(self.ul_bandwidth * self.upload_factor)}