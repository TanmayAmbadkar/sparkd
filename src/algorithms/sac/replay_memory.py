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

    def sample(self, batch_size, get_cost=False, remove_samples=False, horizon=1, sample_range=None):
        # Helper to process batch
        def process_obs(obs_batch):
            if self.is_image:
                return obs_batch.astype(np.float32) / 255.0
            return obs_batch

        start_idx = 0
        end_idx = self.size
        if sample_range is not None:
            start_idx = max(0, sample_range[0])
            end_idx = min(self.size, sample_range[1])

        if horizon == 1:
            if end_idx <= start_idx:
                raise ValueError("Not enough samples in memory/range for horizon 1.")
            
            # Use randint for speed and replacement
            idx = np.random.randint(start_idx, end_idx, batch_size)
            
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
            # Multi-step
            # Ensure we don't sample past the end (sequence length)
            # Valid start indices are [start_idx, end_idx - horizon]
            # Actually limit is inclusive of start, exclusive of end?
            # if end_idx is 100, horizon 5. max start is 95. (95,96,97,98,99).
            # So valid_end = end_idx - horizon + 1 (for python range/randint exclusive upper)
            valid_end = end_idx - horizon + 1
            if valid_end <= start_idx:
                raise ValueError("Not enough samples in memory/range to form multi-step sequences.")
            
            idx = np.random.randint(start_idx, valid_end, batch_size)
            
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

    def save_buffer(self, save_path):
        print(f"Saving Replay Buffer to {save_path}...")
        np.savez_compressed(
            save_path,
            states=self.states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],
            dones=self.dones[:self.size],
            costs=self.costs[:self.size],
            size=self.size,
            position=self.position
        )
        print("Buffer Saved.")

    def load_buffer(self, load_path):
        print(f"Loading Replay Buffer from {load_path}...")
        data = np.load(load_path)
        
        # Check capacity
        loaded_size = int(data['size'])
        if loaded_size > self.capacity:
            print(f"Warning: Loaded buffer size {loaded_size} exceeds capacity {self.capacity}. Truncating.")
            loaded_size = self.capacity
            
        self.size = loaded_size
        self.position = int(data['position']) % self.capacity # reset position or keep? usually keep if resuming, but offline is static.
        
        self.states[:self.size] = data['states'][:self.size]
        self.actions[:self.size] = data['actions'][:self.size]
        self.rewards[:self.size] = data['rewards'][:self.size]
        self.next_states[:self.size] = data['next_states'][:self.size]
        self.dones[:self.size] = data['dones'][:self.size]
        self.costs[:self.size] = data['costs'][:self.size]
        
        print(f"Buffer Loaded. Size: {self.size}")

    def load_d4rl_dataset(self, d4rl_env):
        dataset = d4rl_env.get_dataset()
        print(f"Loading D4RL Dataset via get_dataset()...")
        
        N = dataset['observations'].shape[0]
        if N > self.capacity:
             print(f"Warning: Dataset size {N} > Capacity {self.capacity}. Truncating.")
             N = self.capacity
             
        self.states[:N] = dataset['observations'][:N]
        self.actions[:N] = dataset['actions'][:N]
        self.rewards[:N] = dataset['rewards'][:N]
        self.next_states[:N] = dataset['next_observations'][:N]
        
        # D4RL has 'terminals' (True=Done) and 'timeouts' (True=Truncated)
        terminals = dataset['terminals'][:N]
        timeouts = dataset['timeouts'][:N]
        
        # We store 'done' as unified terminal signal? Or separate? 
        # ReplayMemory currently has 'dones'. Replay buffer usually stores 'done' (terminal state), NOT timeout.
        # Timeout is usually handled by environment not buffer, or buffer needs to know if it's terminal.
        # Standard: 'dones' stores terminals. Timeouts are usually treated as non-terminal for bootstrapping (bootstrap from V(s')),
        # but terminal for episode reset.
        # ALLC treats 'done' as mask (1-done)*V(s'). So timeout should NOT be done (mask=1).
        # Real terminal should be done (mask=0).
        self.dones[:N] = terminals
        
        # costs? D4RL might not have costs unless it's a safety task.
        if 'cost' in dataset:
             self.costs[:N] = dataset['cost'][:N]
        else:
             self.costs[:N] = 0.0
             
        self.size = N
        self.position = N % self.capacity
    def load_minari_dataset(self, dataset_id):
        try:
            import minari
        except ImportError:
             print("Error: Minari not installed. Run 'pip install minari'.")
             return
             
        print(f"Loading Minari Dataset: {dataset_id}")
        try:
            dataset = minari.load_dataset(dataset_id)
        except (ValueError, FileNotFoundError):
            print(f"Dataset {dataset_id} not found locally. Attempting download...")
            try:
                minari.download_dataset(dataset_id)
                dataset = minari.load_dataset(dataset_id)
            except Exception as e:
                print(f"Failed to download/load dataset: {e}")
                return

        total_transitions = 0
        for episode in dataset.iterate_episodes():
             # obs: (T+1, D), act: (T, D), rew: (T,), term: (T,), trunc: (T,)
             obs = episode.observations
             act = episode.actions
             rew = episode.rewards
             term = episode.terminations
             trunc = episode.truncations
             
             T = len(act)
             start = self.position
             
             # Check capacity
             if self.size + T > self.capacity:
                 available = self.capacity - self.size
                 if available <= 0:
                     print("Buffer Full. Stopping load.")
                     break
                 T = available
                 
             end = start + T
             
             # Copy data
             # Minari observations are (T+1, ...), we take 0..T-1 for state, 1..T for next_state
             self.states[start:end] = obs[:T]
             self.next_states[start:end] = obs[1:T+1]
             self.actions[start:end] = act[:T]
             self.rewards[start:end] = rew[:T]
             self.dones[start:end] = term[:T]
             
             # Cost handling (if available in infos)
             # Minari 0.4.0+ accesses infos via episode.infos (dict of arrays)
             # But standard Minari datasets (like mujoco) might not have 'cost'.
             # We default to 0.
             # If 'cost' is in infos, we'd load it. 
             # SafetyGym Minari datasets?? Not standard yet.
             self.costs[start:end] = 0.0 
             
             self.position = (self.position + T) % self.capacity
             self.size += T
             total_transitions += T
             
        print(f"Loaded {total_transitions} transitions from Minari.")
