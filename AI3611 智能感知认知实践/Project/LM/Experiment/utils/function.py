import torch


def count_parameter(model):
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            count += parameter.numel()
    return count


def adjust_alpha(optimizer, learning_rate):
    for parameter in optimizer.param_groups:
        parameter['lr'] = learning_rate


def split_batch(device, data, batch_size):
    count = data.shape[0] // batch_size
    data = data.narrow(0, 0, count * batch_size)
    data = data.view(batch_size, -1).transpose(0, 1).contiguous()
    return data.to(device)


def detach_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(detach_hidden(h) for h in hidden)
