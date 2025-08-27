import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_dim):
        super(ActorCritic, self).__init__()
        # Actor Network
        self.action_space = action_space
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(obs_dim, 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(obs_dim, 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.shape[0]), std=0.01),
        )

        self.actor_logstd = nn.Parameter(
            torch.zeros(1, action_space.shape[0])
        )


    def forward(self):
        raise NotImplementedError

    def act(self, state):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def evaluate(self, state, action):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        value = self.get_value(state)
        return log_prob, entropy, value

    def get_policy(self, state):
        mean = self.actor(state)
        
        return mean, self.actor_logstd

    def get_value(self, state):
        
        value = self.critic(state)
        return value

    def get_log_prob(self, state, action):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return log_prob
