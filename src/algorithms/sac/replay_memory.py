import numpy as np
from torch.utils.data import Dataset

class ReplayMemory:
    def __init__(self, capacity, observation_space, action_dim, seed, horizon=1):
        np.random.seed(seed)
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.state_dim = observation_space.shape   # e.g. (8,) or (3, 64, 64)
        self.action_dim = action_dim
        self.horizon = horizon  # multi-step horizon; 1 means one-step
        
        # Check if image based on shape length (3 means C,H,W)
        self.is_image = len(self.state_dim) == 3
        dtype = np.uint8 if self.is_image else np.float32
        
        self.states = np.zeros((capacity, *self.state_dim), dtype=dtype)
        self.next_states = np.zeros((capacity, *self.state_dim), dtype=dtype)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done, cost):
        # If image, expect state to be in range [0, 255] or compatible with uint8 if we forced it.
        # But usually env gives float [0, 255] or [0, 1].
        # We assume standard gym image envs often give [0, 255] uint8.
        # If the input is float 0-1, we should probably scale it? 
        # For simplicity, assuming the env returns correct format for storage or we cast.
        if self.is_image and state.dtype != np.uint8:
             # Assuming input might be float 0-255 or 0-1. 
             # Safety check: if max <= 1.0, scale up? No, let's strictly assume input is standard gym image (uint8).
             # If user passes float, we strictly cast.
             pass

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.costs[self.position] = cost

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, get_cost=False, remove_samples=False, horizon = 1):
        # Helper to process batch
        def process_obs(obs_batch):
            if self.is_image:
                return obs_batch.astype(np.float32) / 255.0
            return obs_batch

        if horizon == 1:
            idx = np.random.choice(self.size, batch_size, replace=False)
            batch_states = process_obs(self.states[idx])
            batch_actions = self.actions[idx]
            batch_rewards = self.rewards[idx]
            batch_next_states = process_obs(self.next_states[idx])
            batch_dones = self.dones[idx]
            if get_cost:
                batch_costs = self.costs[idx]
                return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_costs
            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
        else:
            valid_length = batch_size - horizon
            if valid_length <= 0:
                raise ValueError("Not enough samples in memory to form multi-step sequences.")
            idx = np.random.choice(batch_size, valid_length, replace=False)
            
            batch_states = np.array([self.states[i : i + horizon] for i in idx])
            batch_next_states = np.array([self.next_states[i : i + horizon] for i in idx])
            
            batch_states = process_obs(batch_states)
            batch_next_states = process_obs(batch_next_states)
            
            batch_actions = np.array([self.actions[i : i + horizon] for i in idx])
            batch_rewards = np.array([self.rewards[i : i + horizon] for i in idx])
            batch_dones = np.array([self.dones[i : i + horizon] for i in idx])
            
            if get_cost:
                batch_costs = np.array([self.costs[i : i + horizon] for i in idx])
                return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_costs
            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def _remove_indices(self, indices):
        """
        Remove the transitions at the given indices by swapping them 
        with the last valid transitions in the buffer and decreasing size.
        """
        # Sort indices in descending order to avoid conflicts when swapping
        indices = np.sort(indices)[::-1]

        for i in indices:
            if i < self.size - 1:
                # Replace with the last valid sample
                last_index = self.size - 1
                self.states[i] = self.states[last_index]
                self.actions[i] = self.actions[last_index]
                self.rewards[i] = self.rewards[last_index]
                self.next_states[i] = self.next_states[last_index]
                self.dones[i] = self.dones[last_index]
                self.costs[i] = self.costs[last_index]

            # Decrement the size of the buffer
            self.size -= 1

    def __len__(self):
        return self.size

    def save_buffer(self, env_name, suffix="", save_path=None):
        # Implement saving arrays if needed
        pass

    def load_buffer(self, save_path):
        # Implement loading arrays if needed
        pass

    def clear_memory(self):
        self.position = 0
        self.size = 0
