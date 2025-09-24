import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import numpy as np
import lightgbm as lgb

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

class CPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_dim=64):
        super(CPOActorCritic, self).__init__()
        # Actor Network
        self.action_space = action_space
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_space.shape[0]), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space.shape[0]))

        # Reward Critic (V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Cost Critic (V_c(s))
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self):
        raise NotImplementedError

    
    def evaluate(self, state, action):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        value = self.get_value(state)
        cost_value = self.get_cost_value(state)
        return log_prob, entropy, value, cost_value

    def get_policy(self, state):
        mean = self.actor(state)
        log_std = torch.clamp(self.actor_logstd, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_distribution(self, state):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist
    
    def act(self, state):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob


    def get_value(self, state):
        value = self.critic(state)
        return value

    def get_cost_value(self, state):
        cost_value = self.cost_critic(state)
        return cost_value

    def get_log_prob(self, state, action):
        mean, log_std = self.get_policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return log_prob


class EnsembleDynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim, num_ensemble=5, hidden_size=256, learning_rate=1e-3):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Create an ensemble of models
        # Each model predicts the *change* in state (delta_state)
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, obs_dim)
            ) for _ in range(num_ensemble)
        ])
        
        self.optimizers = [Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, states, actions):
        """Returns predictions from all models in the ensemble."""
        inputs = torch.cat([states, actions], dim=-1)
        return torch.stack([model(inputs) for model in self.models], dim=0)

    @torch.no_grad()
    def predict(self, states, actions):
        """
        Predicts the next state using the ensemble.
        For planning, we can average the predictions or use trajectory sampling (TS).
        """
        assert states.dim() == 2 and actions.dim() == 2
        
        # Predict delta_state for all models
        delta_states = self(states, actions) # Shape: [num_ensemble, batch_size, obs_dim]
        
        # Add the predicted change to the original state
        next_states = states.unsqueeze(0) + delta_states
        
        return next_states

    def train_model(self, states, actions, next_states, epochs=70, batch_size=256):
        """Train all models in the ensemble."""
        inputs = torch.cat([states, actions], dim=-1).to(self.device)
        labels = (next_states - states).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        
        for i, model in enumerate(self.models):
            # Use a subset of data for each model to encourage diversity [cite: 90]
            # This is a simple version; bootstrapping is more common.
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            model.train()
            for epoch in range(epochs):
                for batch_inputs, batch_labels in loader:
                    predictions = model(batch_inputs)
                    loss = nn.MSELoss()(predictions, batch_labels)
                    
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    
# --- New Component: CostModel ---

class CostModel:
    def __init__(self):
        # Using parameters similar to the paper's description [cite: 317, 318]
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'n_estimators': 400,
            'max_depth': 8,
            'num_leaves': 12,
            'learning_rate': 0.3,
            'verbose': -1,
        }
        self.model = lgb.LGBMClassifier(**params)

    def train_model(self, safe_states, unsafe_states):
        """Train the classifier on safe and unsafe states."""
        # The paper uses separate buffers to control the ratio [cite: 97]
        # Here we just combine them for simplicity
        if unsafe_states.shape[0] == 0:
            print("Warning: No unsafe states to train cost model.")
            return

        states = np.vstack([safe_states, unsafe_states])
        # Labels: 0 for safe, 1 for unsafe [cite: 66]
        labels = np.concatenate([
            np.zeros(safe_states.shape[0]), 
            np.ones(unsafe_states.shape[0])
        ])
        
        self.model.fit(states, labels)

    def predict(self, states):
        """Predict if states are unsafe (1) or safe (0)."""
        return self.model.predict(states)
    
    
# --- New Component: RCEPlanner ---
import torch
import numpy as np

class RCEPlanner:
    def __init__(self, dynamics_model, cost_model, action_dim, planner_cfgs):
        self.dynamics = dynamics_model
        self.cost_model = cost_model
        self.action_dim = action_dim
        
        self.horizon = planner_cfgs['horizon'] # Planning horizon T
        self.num_samples = planner_cfgs['num_samples'] # N samples
        self.num_elites = planner_cfgs['num_elites'] # k elites
        self.num_iter = planner_cfgs['num_iter']
        self.device = self.dynamics.device

    def plan(self, initial_state):
        """
        Optimize an action sequence using the RCE method.
        This corresponds to Algorithm 1 in the paper.
        """
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        for i in range(self.num_iter):
            # 1. Draw N action sequences from the distribution [cite: 121]
            # Shape: [num_samples, horizon, action_dim]
            actions = torch.normal(mean.repeat(self.num_samples, 1, 1), std.repeat(self.num_samples, 1, 1))
            actions = torch.clamp(actions, -1.0, 1.0)
            
            rewards = torch.zeros(self.num_samples, device=self.device)
            costs = torch.zeros(self.num_samples, device=self.device)
            
            # 2. Evaluate each action sequence
            current_states = initial_state.repeat(self.num_samples, 1)
            
            for t in range(self.horizon):
                action_t = actions[:, t, :]
                
                # Predict next state with the dynamics ensemble
                # Shape: [num_ensemble, num_samples, obs_dim]
                next_states_ensemble = self.dynamics.predict(current_states, action_t)
                
                # Estimate cost using the worst-case prediction across the ensemble [cite: 122, 127]
                # Reshape for cost model: [num_ensemble * num_samples, obs_dim]
                next_states_flat = next_states_ensemble.reshape(-1, next_states_ensemble.shape[-1])
                predicted_costs = self.cost_model.predict(next_states_flat.cpu().numpy())
                predicted_costs = torch.from_numpy(predicted_costs).to(self.device).reshape_as(next_states_ensemble[..., 0])
                
                # Worst-case cost over the ensemble dimension
                max_cost_t, _ = torch.max(predicted_costs, dim=0) # Shape: [num_samples]
                costs += max_cost_t
                
                # Average reward over the ensemble (assuming reward is part of state or known)
                # This part needs a reward function, for simplicity let's assume it's based on state
                # In SafetyGym, reward is for reaching a goal, so we'd need to predict goal proximity
                # Let's use a placeholder reward: negative distance to origin
                rewards_t = -torch.mean(torch.linalg.norm(next_states_ensemble, dim=2), dim=0)
                rewards += rewards_t
                
                # Propagate state for next step (using one model for trajectory consistency)
                current_states = next_states_ensemble[0]

            # 3. Select the feasible set (trajectories with zero cost) [cite: 121]
            feasible_indices = torch.where(costs == 0)[0]

            if len(feasible_indices) > 0:
                # 4a. If feasible set is not empty, select elites based on highest reward [cite: 124]
                top_rewards, top_indices = torch.topk(rewards[feasible_indices], self.num_elites)
                elites = actions[feasible_indices[top_indices]]
            else:
                # 4b. If feasible set is empty, select elites based on lowest cost [cite: 125]
                _, top_indices = torch.topk(costs, self.num_elites, largest=False)
                elites = actions[top_indices]

            # 5. Update the sampling distribution based on the elites [cite: 121]
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0)
            
            # Smoothly update distribution parameters
            mean = 0.7 * mean + 0.3 * new_mean
            std = 0.7 * std + 0.3 * new_std

        # Return the first action of the final mean sequence [cite: 130]
        return mean[0].unsqueeze(0)