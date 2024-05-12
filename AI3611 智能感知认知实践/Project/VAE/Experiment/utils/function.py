import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model.vae import VariationalAutoencoder


def create_dataset(root_path, batch_size, num_workers):
    dataset_train = MNIST(root_path, train=True, transform=transforms.ToTensor(), download=True)
    dataset_test = MNIST(root_path, train=False, transform=transforms.ToTensor(), download=True)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataset_train, dataset_test, loader_train, loader_test


def create_model(device, input_dims, latent_dims, hidden_dims, hidden_layers):
    model = VariationalAutoencoder(input_dims, latent_dims, hidden_dims, hidden_layers).to(device)
    return model


def compute_loss(input, output, mu, logvar):
    input, output = torch.flatten(input, start_dim=1), torch.flatten(output, start_dim=1)
    loss_reconstruction = F.binary_cross_entropy(output, input, reduction='sum')
    loss_kl_divergence = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))
    return loss_reconstruction, loss_kl_divergence


def plot_sample(device, model, dataset, path):
    sample_input = torch.stack([dataset[i][0] for i in range(128)]).to(device)
    with torch.no_grad():
        sample_output, _, _, _ = model(sample_input)
    sample = torch.cat([sample_input, sample_output])
    save_image(sample, path, nrow=16)


def plot_curve(record, path):
    plt.figure(figsize=(8, 6))
    plt.plot(record['epoch'], record['train_loss'], label='Total Loss (Train)')
    plt.plot(record['epoch'], record['train_loss_reconstruction'], label='Reconstruction Loss (Train)')
    plt.plot(record['epoch'], record['train_loss_kl_divergence'], label='KL Divergence Loss (Train)')
    plt.plot(record['epoch'], record['test_loss'], label='Total Loss (Test)')
    plt.plot(record['epoch'], record['test_loss_reconstruction'], label='Reconstruction Loss (Test)')
    plt.plot(record['epoch'], record['test_loss_kl_divergence'], label='KL Divergence Loss (Test)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
