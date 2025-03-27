import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, action_space=None):
        super(ActorCritic, self).__init__()
        # Actor Network
        self.actor_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)

        # Critic Network (Value Network)
        self.critic_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        # Apply tanh without any scalin
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
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_value(self, state):
        x = F.relu(self.critic_fc1(state))
        x = F.relu(self.critic_fc2(x))
        value = self.critic_value(x)
        return value

    def get_log_prob(self, state, action):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        eps = 1e-6
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return log_prob
