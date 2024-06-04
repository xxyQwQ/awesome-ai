import torch


def convert_batch_to_tensor(batch, device):
    return {key: torch.from_numpy(value).to(device, non_blocking=True) for key, value in batch.items()}


def soft_update_target_network(source_network, target_network, update_rate):
    for source_parameter, target_parameter in zip(source_network.parameters(), target_network.parameters()):
        target_parameter.data = (1 - update_rate) * target_parameter.data + update_rate * source_parameter.data


def extend_and_repeat_tensor(tensor, dim, repeat):
    shape = [1 for _ in range(tensor.ndim + 1)]
    shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(shape)
