import torch
from torch import nn
from torch.optim import Adam


class BCPolicy(nn.Module):
    def __init__(self, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super(BCPolicy, self).__init__()
        layers = [nn.Linear(state_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        layers += [nn.Linear(hidden_dims, action_dims)]
        self.policy = nn.Sequential(*layers)

    def forward(self, state):
        action = self.policy(state)
        return action


class BCAgent(object):
    def __init__(
            self,
            state_dims,
            action_dims,
            num_layers=2,
            hidden_dims=256,
            policy_lr=1e-3,
            **kwargs
        ):
        self.policy = BCPolicy(state_dims, action_dims, num_layers, hidden_dims)
        self.optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.criterion = nn.MSELoss()

    @property
    def modules(self):
        return {
            "policy": self.policy
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
            action = self.policy(state.unsqueeze(0))
        return action.detach()

    def train_batch(self, batch):
        state = batch['state']
        action = batch['action']
        prediction = self.policy(state)
        loss = self.criterion(prediction, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'MSE Loss': loss.item()
        }
