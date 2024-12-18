import numpy as np

class ReplayMemory:
    def __init__(self, capacity, observation_space, action_dim, seed):
        np.random.seed(seed)
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.state_dim = observation_space.shape
        self.action_dim = action_dim
        
        self.states = np.zeros((capacity, *self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, *self.state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.costs = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done, cost):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.costs[self.position] = cost

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, get_cost=False, remove_samples=False):
        idx = np.random.choice(self.size, batch_size, replace=False)

        batch_states = self.states[idx]
        batch_actions = self.actions[idx]
        batch_rewards = self.rewards[idx]
        batch_next_states = self.next_states[idx]
        batch_dones = self.dones[idx]
        
        if get_cost:
            batch_costs = self.costs[idx]

        if remove_samples:
            self._remove_indices(idx)

        if get_cost:
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
