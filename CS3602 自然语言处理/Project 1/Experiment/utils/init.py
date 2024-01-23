import random
import numpy as np
import torch


def set_random_seed(random_seed=999):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)


def set_torch_device(deviceId):
    if deviceId < 0:
        device = torch.device("cpu")
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device(f'cuda:{deviceId}')
        torch.backends.cudnn.enabled = False
    return device
