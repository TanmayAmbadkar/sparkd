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
        normal = Normal(mean, std)
        
        # Sample from the Normal distribution (unbounded)
        x_t = normal.rsample()  # Use rsample() for reparameterization trick
        
        # Squash the sample to [-1, 1] using tanh
        y_t = torch.tanh(x_t)
        
        # Calculate the log probability, correcting for the tanh transformation
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # Rescale the squashed action to the environment's action space bounds
        action = y_t * torch.tensor(self.action_space.high, dtype=torch.float32)

        return action, log_prob

    def evaluate(self, state, action):
        # Get the policy distribution parameters
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # --- Start of Correction ---
        # The 'action' from the buffer is squashed and rescaled.
        # We need to reverse this process to find the original sample 'x_t'.

        # 1. Un-scale the action from environment bounds back to the [-1, 1] range of tanh
        #    (Assumes symmetric action space where low = -high)
        y_t = action / torch.tensor(self.action_space.high, dtype=torch.float32)
        
        # To avoid instability at the boundaries of tanh, clip y_t
        y_t = torch.clamp(y_t, -0.999, 0.999)

        # 2. Apply the inverse of tanh (atanh) to get the pre-squashed action
        x_t = torch.atanh(y_t)

        # 3. Now, calculate the log-prob using the pre-squashed action 'x_t'
        #    and apply the same correction for the tanh transformation.
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # --- End of Correction ---

        # Entropy is calculated from the base distribution
        entropy = normal.entropy().sum(-1, keepdim=True)
        
        # Value function is calculated as before
        value = self.get_value(state)
        
        return log_prob, entropy, value

    def get_policy(self, state):
        mean = self.actor(state)
        
        return mean, torch.clamp(self.actor_logstd, LOG_STD_MIN, LOG_STD_MAX)

    def get_value(self, state):
        
        value = self.critic(state)
        return value

    def get_log_prob(self, state, action):
        # Get the base Normal distribution as before
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        # --- Start of Correction ---
        # The 'action' from the buffer is squashed and rescaled.
        # We must reverse this process to find the log-probability.

        # 1. Un-scale the action from the environment's bounds back to the [-1, 1] range.
        #    (This assumes a symmetric action space, e.g., [-2, 2])
        action_high = torch.tensor(self.action_space.high, device=action.device, dtype=torch.float32)
        y_t = action / action_high

        # 2. Clamp for numerical stability before applying the inverse of tanh.
        y_t = torch.clamp(y_t, -0.9999, 0.9999)

        # 3. Apply the inverse of tanh (atanh) to get the original, pre-squashed sample.
        x_t = torch.atanh(y_t)

        # 4. Calculate the log-prob of the pre-squashed sample and apply the
        #    correction for the change of variables from the tanh transformation.
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        
        # 5. Sum across the action dimensions.
        log_prob = log_prob.sum(1)
        # --- End of Correction ---
        
        return log_prob
