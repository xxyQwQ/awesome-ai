import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.utils import save_image
from utils.function import create_dataset, create_model


# load configuration
checkpoint_path = './checkpoint/visual'
dataset_path = './dataset'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
_, dataset, _, loader = create_dataset(dataset_path, 256, 0)


# batch encoding
def encode(device, model, loader):
    latents, labels = [], []
    for input, label in loader:
        input, label = input.to(device), label.to(device)
        with torch.no_grad():
            latent = model.encode(input)
        latents.append(latent)
        labels.append(label)
    latents = torch.cat(latents).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()
    return latents, labels


# latent_dims: 1
save_path = os.path.join(checkpoint_path, 'latent_1')
os.makedirs(save_path, exist_ok=True)
model_path = './checkpoint/latent_1/epoch_300/model.pth'
model = create_model(device, 784, 1, 256, 2)
model.load_state_dict(torch.load(model_path, map_location=device))

latent = torch.arange(-5, 5, 0.025).view(-1, 1).to(device)
output = model.decode(latent)
save_image(output, os.path.join(save_path, 'sample.png'), nrow=20)

plt.figure(figsize=(6, 6))
latents, labels = encode(device, model, loader)
for label in range(10):
    x = latents[labels == label]
    y = label * np.ones_like(x)
    plt.scatter(x, y, s=10, alpha=1.0, label=str(label))
plt.xlabel('x')
plt.ylabel('label')
plt.title('Latent Space')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'latent.png'))
plt.close()


# latent_dims: 2
save_path = os.path.join(checkpoint_path, 'latent_2')
os.makedirs(save_path, exist_ok=True)
model_path = './checkpoint/latent_2/epoch_300/model.pth'
model = create_model(device, 784, 2, 256, 2)
model.load_state_dict(torch.load(model_path, map_location=device))

latent = torch.stack(torch.meshgrid(torch.arange(-5, 5, 0.5), torch.arange(-5, 5, 0.5), indexing='ij')).view(2, -1).transpose(0, 1).to(device)
output = model.decode(latent)
save_image(output, os.path.join(save_path, 'sample.png'), nrow=20)

plt.figure(figsize=(6, 6))
latents, labels = encode(device, model, loader)
for label in range(10):
    x, y = latents[labels == label].transpose()
    plt.scatter(x, y, s=5, alpha=0.8, label=str(label))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Latent Space')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'latent.png'))
plt.close()


# latent_dims: 64
save_path = os.path.join(checkpoint_path, 'latent_64')
os.makedirs(save_path, exist_ok=True)
model_path = './checkpoint/latent_64/epoch_300/model.pth'
model = create_model(device, 784, 64, 256, 2)
model.load_state_dict(torch.load(model_path, map_location=device))

latent = torch.randn(400, 64).to(device)
output = model.decode(latent)
save_image(output, os.path.join(save_path, 'sample.png'), nrow=20)

plt.figure(figsize=(6, 6))
latents, labels = encode(device, model, loader)
embeddings = TSNE(n_components=2).fit_transform(latents)
for label in range(10):
    x, y = embeddings[labels == label].transpose()
    plt.scatter(x, y, s=5, alpha=0.8, label=str(label))
plt.xlabel('x\'')
plt.ylabel('y\'')
plt.title('Latent Space')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'latent.png'))
plt.close()
