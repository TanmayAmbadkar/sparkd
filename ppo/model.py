import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear layer with orthogonal initialization for weights
    and a constant for biases.

    Args:
        layer (nn.Linear): The linear layer to initialize.
        std (float): The standard deviation for the orthogonal initialization.
        bias_const (float): The constant value for the bias.

    Returns:
        nn.Linear: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """
    A standard Actor-Critic network that outputs actions for a continuous
    action space using a Gaussian policy. This implementation does not use
    action squashing (e.g., tanh), making it suitable for algorithms like PPO.
    """
    def __init__(self, obs_dim, action_space, hidden_dim=64):
        """
        Initializes the Actor and Critic networks.

        Args:
            obs_dim (int): The dimension of the observation space.
            action_space: The environment's action space (e.g., from Gymnasium).
            hidden_dim (int): The number of neurons in the hidden layers.
        """
        super(ActorCritic, self).__init__()
        self.action_space = action_space
        action_dim = action_space.shape[0]

        # Critic Network: Estimates the value of a state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor Network: Outputs the mean of the action distribution
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        # A learnable parameter for the standard deviation of the action distribution
        self.actor_logstd = nn.Parameter(-torch.ones(1, action_dim))

    def get_value(self, state):
        """
        Gets the value of a state from the critic network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The estimated value of the state.
        """
        return self.critic(state)

    def get_policy(self, state):
        """
        Gets the policy's action distribution for a given state.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.distributions.Normal: The action distribution.
        """
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def act(self, state):
        """
        Samples an action from the policy for a given state.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            tuple: A tuple containing the action, its log probability, and its entropy.
        """
        dist = self.get_policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)

        # Clip the action to the valid range of the environment's action space
        # action = torch.clamp(
        #     action,
        #     torch.tensor(self.action_space.low, dtype=torch.float32, device=action.device),
        #     torch.tensor(self.action_space.high, dtype=torch.float32, device=action.device)
        # )
        return action, log_prob

    def evaluate(self, state, action):
        """
        Evaluates a given state-action pair, returning the action's log probability,
        the distribution's entropy, and the state's value.

        Args:
            state (torch.Tensor): The state to evaluate.
            action (torch.Tensor): The action to evaluate.

        Returns:
            tuple: A tuple containing the log probability of the action, the entropy
                   of the distribution, and the estimated value of the state.
        """
        dist = self.get_policy(state)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        value = self.get_value(state)
        return log_prob, entropy, value

    def get_log_prob(self, state, action):
        """
        Computes the log probability of an action given a state.

        Args:
            state (torch.Tensor): The input state.
            action (torch.Tensor): The action for which to compute the log probability.

        Returns:
            torch.Tensor: The log probability of the action.
        """
        dist = self.get_policy(state)
        return dist.log_prob(action).sum(-1, keepdim=True)