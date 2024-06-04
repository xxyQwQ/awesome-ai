from copy import deepcopy

import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from utils.torch import soft_update_target_network, extend_and_repeat_tensor


class CSACScaler(nn.Module):
    def __init__(self, value):
        super(CSACScaler, self).__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self):
        return self.value


class CSACActor(nn.Module):
    def __init__(self, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super(CSACActor, self).__init__()
        layers = [nn.Linear(state_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        self.latent = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_dims, action_dims)
        self.log_sigma = nn.Linear(hidden_dims, action_dims)
        self.log_sigma_scale = CSACScaler(1.0)
        self.log_sigma_bias = CSACScaler(-1.0)

    def forward(self, state, deterministic=False, repeat=None):
        if repeat is not None:
            state = extend_and_repeat_tensor(state, 1, repeat)
        latent = self.latent(state)
        mu = self.mu(latent)
        log_sigma = self.log_sigma_scale() * self.log_sigma(latent) + self.log_sigma_bias()
        sigma = torch.exp(torch.clamp(log_sigma, -20, 2))
        dist = TransformedDistribution(Normal(mu, sigma), TanhTransform(cache_size=1))
        action = torch.tanh(mu) if deterministic else dist.rsample()
        log_prob = torch.sum(dist.log_prob(action), dim=-1)
        return action, log_prob

    def log_prob(self, state, action, repeat=None):
        if repeat is not None:
            state = extend_and_repeat_tensor(state, 1, repeat)
        latent = self.latent(state)
        mu = self.mu(latent)
        log_sigma = self.log_sigma_scale() * self.log_sigma(latent) + self.log_sigma_bias()
        sigma = torch.exp(torch.clamp(log_sigma, -20, 2))
        dist = TransformedDistribution(Normal(mu, sigma), TanhTransform(cache_size=1))
        log_prob = torch.sum(dist.log_prob(action), dim=-1)
        return log_prob


class CSACCritic(nn.Module):
    def __init__(self, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super(CSACCritic, self).__init__()
        layers = [nn.Linear(state_dims + action_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        self.latent = nn.Sequential(*layers)
        self.value = nn.Linear(hidden_dims, 1)

    def forward(self, state, action, repeat=None):
        if repeat is not None:
            state = extend_and_repeat_tensor(state, 1, repeat)
            state = state.reshape(-1, state.shape[-1])
            action = action.reshape(-1, action.shape[-1])
        latent = self.latent(torch.cat([state, action], dim=-1))
        value = self.value(latent)
        if repeat is not None:
            value = value.reshape(-1, repeat)
        return value


class CSACAgent(object):
    """Conservative Soft Actor-Critic"""
    def __init__(
        self,
        state_dims,
        action_dims,
        num_layers=2,
        hidden_dims=256,
        actor_lr=3e-5,
        critic_lr=3e-4,
        discount_factor=0.99,
        update_rate=5e-3,
        num_samples=10,
        penalty_weight=5.0,
        **kwargs
    ):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.discount_factor = discount_factor
        self.update_rate = update_rate
        self.num_samples = num_samples
        self.penalty_weight = penalty_weight
        self.target_entropy = -self.action_dims

        self.log_alpha = CSACScaler(0.0)
        self.alpha_optimizer = Adam(self.log_alpha.parameters(), lr=actor_lr)

        self.actor = CSACActor(state_dims, action_dims, num_layers, hidden_dims)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = CSACCritic(state_dims, action_dims, num_layers, hidden_dims)
        self.critic_2 = CSACCritic(state_dims, action_dims, num_layers, hidden_dims)
        self.critic_optimizer = Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)

        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

    @property
    def modules(self):
        return {
            'log_alpha': self.log_alpha,
            'actor': self.actor,
            'critic_1': self.critic_1,
            'critic_2': self.critic_2,
            'target_critic_1': self.target_critic_1,
            'target_critic_2': self.target_critic_2
        }

    def to_device(self, device):
        for module in self.modules.values():
            module.to(device)

    def save_model(self, path):
        torch.save({name: module.state_dict() for name, module in self.modules.items()}, path)

    def load_model(self, path):
        weights = torch.load(path)
        for name, module in self.modules.items():
            module.load_state_dict(weights[name])

    def take_action(self, state):
        with torch.no_grad():
            action, _ = self.actor(state.unsqueeze(0), deterministic=True)
        return action.squeeze(0).detach()

    def soft_update(self, update_rate):
        soft_update_target_network(self.critic_1, self.target_critic_1, update_rate)
        soft_update_target_network(self.critic_2, self.target_critic_2, update_rate)

    def train_batch(self, batch, warmup=False):
        state = batch['state']
        action = batch['action']
        reward = batch['reward']
        next_state = batch['next_state']
        done = batch['done']

        pred_action, log_pi = self.actor(state)
        alpha = self.log_alpha().exp()
        alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()

        if warmup:
            log_prob = self.actor.log_prob(state, action)
            actor_loss = (alpha * log_pi - log_prob).mean()
        else:
            pred_value = torch.min(self.critic_1(state, pred_action), self.critic_2(state, pred_action))
            actor_loss = (alpha * log_pi - pred_value).mean()

        pred_value_1 = self.critic_1(state, action)
        pred_value_2 = self.critic_2(state, action)
        pred_next_action, next_log_pi = self.actor(next_state)
        target_value = torch.min(self.target_critic_1(next_state, pred_next_action), self.target_critic_2(next_state, pred_next_action))
        td_target = reward + self.discount_factor * (1 - done) * target_value
        critic_1_loss = F.mse_loss(pred_value_1, td_target.detach())
        critic_2_loss = F.mse_loss(pred_value_2, td_target.detach())

        batch_size = state.shape[0]
        cql_random_action = action.new_empty((batch_size, self.num_samples, self.action_dims), requires_grad=False).uniform_(-1, 1)
        cql_current_action, cql_current_log_pi = self.actor(state, repeat=self.num_samples)
        cql_current_action, cql_current_log_pi = cql_current_action.detach(), cql_current_log_pi.detach()
        cql_next_action, cql_next_log_pi = self.actor(next_state, repeat=self.num_samples)
        cql_next_action, cql_next_log_pi = cql_next_action.detach(), cql_next_log_pi.detach()

        cql_random_value_1 = self.critic_1(state, cql_random_action, repeat=self.num_samples)
        cql_random_value_2 = self.critic_2(state, cql_random_action, repeat=self.num_samples)
        cql_current_value_1 = self.critic_1(state, cql_current_action, repeat=self.num_samples)
        cql_current_value_2 = self.critic_2(state, cql_current_action, repeat=self.num_samples)
        cql_next_value_1 = self.target_critic_1(next_state, cql_next_action, repeat=self.num_samples)
        cql_next_value_2 = self.target_critic_2(next_state, cql_next_action, repeat=self.num_samples)

        random_density = np.log(0.5 ** self.action_dims)
        cql_union_value_1 = torch.cat([cql_random_value_1 - random_density, cql_next_value_1 - cql_next_log_pi.detach(), cql_current_value_1 - cql_current_log_pi], dim=1)
        cql_union_value_2 = torch.cat([cql_random_value_2 - random_density, cql_next_value_2 - cql_next_log_pi.detach(), cql_current_value_2 - cql_current_log_pi], dim=1)
        cql_diff_value_1 = (torch.logsumexp(cql_union_value_1, dim=1) - pred_value_1).mean()
        cql_diff_value_2 = (torch.logsumexp(cql_union_value_2, dim=1) - pred_value_2).mean()

        cql_critic_1_loss = self.penalty_weight * cql_diff_value_1
        cql_critic_2_loss = self.penalty_weight * cql_diff_value_2
        critic_loss = critic_1_loss + critic_2_loss + cql_critic_1_loss + cql_critic_2_loss

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.soft_update(self.update_rate)

        return {
            'Alpha Value': self.log_alpha().exp().item(),
            'Actor Loss': actor_loss.item(),
            'Critic Loss': critic_1_loss.item() + critic_2_loss.item(),
            'CQL Loss': cql_critic_1_loss.item() + cql_critic_2_loss.item()
        }
