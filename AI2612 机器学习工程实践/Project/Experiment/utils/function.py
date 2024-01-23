import torch
from torchvision.utils import make_grid


def hinge_loss(input, positive=True):
    if positive:
        return torch.relu(1 - input).mean()
    else:
        return torch.relu(1 + input).mean()


def plot_sample(image_source, image_target, image_result):
    def plot_grid(input):
        return 0.5 * make_grid(input[:8].detach().cpu(), nrow=8) + 0.5
    image_source, image_target, image_result = plot_grid(image_source), plot_grid(image_target), plot_grid(image_result)
    return torch.cat([image_source, image_target, image_result], dim=1).numpy()
