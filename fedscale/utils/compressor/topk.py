import torch

# ===== Top-K implementation =====
# From "GRACE: A Compressed Communication Framework for Distributed Machine Learning0"  

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices


def desparsify(tensors, numel):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(object):

    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor):
        tensors = sparsify(tensor, self.compress_ratio)
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel)
        return tensor_decompressed.view(shape)