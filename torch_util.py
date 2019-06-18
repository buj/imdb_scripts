import torch


def make_cuda(tensor):
  """Turn the tensor into cuda if possible."""
  if torch.cuda.current_device() != -1:
    return tensor.cuda()
  return tensor
