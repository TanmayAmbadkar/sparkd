# --- Main Algorithm Class: RCEPETS ---
import torch
import numpy as np
from cpo.model import EnsembleDynamicsModel, CostModel, RCEPlanner
class RCEPETS:
    def __init__(self, obs_dim, action_space, args):
        self.device = torch.device("cuda" if getattr(args, "cuda", True) else "cpu")
        
        # 1. Initialize the models instead of actor-critic
        self.dynamics_model = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_space.shape[0],
            num_ensemble=args.num_ensemble,
            hidden_size=args.hidden_size,
        ).to(self.device)
        
        self.cost_model = CostModel()
        
        # 2. Initialize the planner instead of optimizers
        self.planner = RCEPlanner(
            dynamics_model=self.dynamics_model,
            cost_model=self.cost_model,
            action_dim=action_space.shape[0],
            planner_cfgs=args.planner_cfgs # e.g., {'horizon': 8, 'num_samples': 500, 'num_elites': 12, 'num_iter': 8}
        )

        # 3. Use a large buffer to store all historical data for model training [cite: 130]
        # This replaces the on-policy memory buffer
        self.data_buffer = {'states': [], 'actions': [], 'next_states': []}
        self.unsafe_buffer = {'states': []} # For cost model training

    def select_action(self, state):
        """Select action using the RCE planner."""
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        
        # In RCEPETS, action selection is planning
        action_tensor = self.planner.plan(state_tensor)
        
        return action_tensor.cpu().numpy()[0]

    def _update_models(self):
        """
        This is the main "learning" step in RCEPETS.
        It updates the dynamics and cost models using all collected data.
        This replaces `update_parameters`.
        """
        print("Updating dynamics and cost models...")
        
        # Prepare data for dynamics model
        states = torch.tensor(np.array(self.data_buffer['states']), dtype=torch.float32)
        actions = torch.tensor(np.array(self.data_buffer['actions']), dtype=torch.float32)
        next_states = torch.tensor(np.array(self.data_buffer['next_states']), dtype=torch.float32)
        
        # Train dynamics model
        self.dynamics_model.train_model(states, actions, next_states, epochs=70)
        
        # Prepare data for cost model
        safe_states = np.array(self.data_buffer['states'])
        unsafe_states = np.array(self.unsafe_buffer['states']) if self.unsafe_buffer['states'] else np.empty((0, states.shape[1]))

        # Train cost model
        self.cost_model.train_model(safe_states, unsafe_states)
        print("Model updates complete.")

    def store_transition(self, state, action, next_state, cost):
        """Store a transition in the appropriate buffers."""
        self.data_buffer['states'].append(state)
        self.data_buffer['actions'].append(action)
        self.data_buffer['next_states'].append(next_state)
        
        if cost > 0:
            # The paper assumes a cost of 1 for violation, 0 otherwise [cite: 66]
            self.unsafe_buffer['states'].append(next_state)

    def learn(self, env, total_steps):
        """Main training loop, adapted from your structure."""
        # This corresponds to Algorithm 2 in the paper
        
        # Initial random data collection
        state, _ = env.reset()
        for _ in range(2000): # Collect some initial data
             action = env.action_space.sample()
             next_state, reward, cost, terminated, truncated, _ = env.step(action)
             self.store_transition(state, action, next_state, cost)
             state = next_state if not (terminated or truncated) else env.reset()[0]
        
        # Train models for the first time
        self._update_models()

        # Main loop
        state, _ = env.reset()
        for step in range(total_steps):
            # 1. Observe state and select action via planning [cite: 130]
            action = self.select_action(state)
            
            # 2. Apply action to environment [cite: 130]
            next_state, reward, cost, terminated, truncated, _ = env.step(action)
            
            # 3. Store data in buffer [cite: 130]
            self.store_transition(state, action, next_state, cost)

            state = next_state
            if terminated or truncated:
                state, _ = env.reset()

            # 4. Periodically update the models with all data [cite: 130]
            if (step + 1) % 1000 == 0: # Update models every 1000 steps
                self._update_models()
            
            if (step + 1) % 500 == 0:
                print(f"Step {step+1}/{total_steps}")