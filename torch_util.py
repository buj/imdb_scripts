import torch
import sys


cuda_id = -1

def set_cuda(new_cuda_id):
    """Sets whether we should use cuda or not."""
    global cuda_id
    if cuda_id >= 0:
        torch.cuda.set_device(cuda_id)
    cuda_id = new_cuda_id

def make_cuda(tensor):
    """Turn the tensor into cuda if possible."""
    if cuda_id >= 0:
        return tensor.cuda()
    return tensor
