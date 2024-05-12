import torch
from torch import nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 256, hidden_layers: int = 2):
        super(MultilayerPerceptron, self).__init__()
        self.network = [nn.Linear(in_dims, hidden_dims), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers - 1): # hidden layers
            self.network.append(nn.Linear(hidden_dims, hidden_dims))
            self.network.append(nn.ReLU(inplace=True))
        self.network.append(nn.Linear(hidden_dims, out_dims))
        self.network = nn.Sequential(*self.network)

    def forward(self, input):
        output = self.network(input)
        return output


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dims: int, latent_dims: int, hidden_dims: int, hidden_layers: int):
        super(VariationalAutoencoder, self).__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.encoder = MultilayerPerceptron(input_dims, 2 * latent_dims, hidden_dims, hidden_layers) # 2 * latent_dims for mu and logvar
        self.decoder = MultilayerPerceptron(latent_dims, input_dims, hidden_dims, hidden_layers)

    def reparameterize(self, mu, logvar, deterministic=False):
        sigma = torch.exp(0.5 * logvar) # logvar = log(sigma ^ 2)
        epsilon = torch.randn_like(sigma) # sample from N(0, 1)
        latent = mu if deterministic else mu + sigma * epsilon
        return latent

    def forward(self, input, deterministic=False):
        batch_size, channel, height, width = input.shape
        assert channel * height * width == self.input_dims
        input = input.view(batch_size, self.input_dims)
        mu, logvar = self.encoder(input).chunk(2, dim=1) # split output into mu and logvar
        latent = self.reparameterize(mu, logvar, deterministic=deterministic) # reparameterization trick
        output = torch.sigmoid(self.decoder(latent)).view(batch_size, channel, height, width)
        return output, mu, logvar, latent
    
    def encode(self, input, channel=1, height=28, width=28):
        assert input.shape[1:] == (channel, height, width)
        return self.reparameterize(*self.encoder(input.view(-1, self.input_dims)).chunk(2, dim=1), deterministic=True)

    def decode(self, latent, channel=1, height=28, width=28):
        assert latent.shape[1] == self.latent_dims
        return torch.sigmoid(self.decoder(latent)).view(-1, channel, height, width)
