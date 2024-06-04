import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)


def get_model_grad_vec(model):
    # Return the model grad as a vector
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)


def update_grad(model, grad_vec):
    idx = 0
    for name, param in model.named_parameters():
        arr_shape = param.grad.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
        idx += size


def update_param(model, param_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.data.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.data = param_vec[idx:idx+size].reshape(arr_shape)
        idx += size


def accuracy(output, target, topk=(1,)):
    # Computes the precision@k for the specified values of k
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
