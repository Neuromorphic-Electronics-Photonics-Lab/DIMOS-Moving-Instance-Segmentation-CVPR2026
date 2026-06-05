import torch
import torch.distributed as dist


def copy_to_device(inputs, device, non_blocking=True):
    if isinstance(inputs, list):
        inputs = [copy_to_device(item, device, non_blocking) for item in inputs]
    elif isinstance(inputs, dict):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = copy_to_device(v, device, non_blocking)
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError('Unknown type: %s' % str(type(inputs)))
    return inputs


def dist_reduce_sum(value, n_gpus):
    if n_gpus <= 1:
        return value
    tensor = torch.Tensor([value]).cuda()
    dist.all_reduce(tensor)
    return tensor.item()
