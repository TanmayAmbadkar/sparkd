import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from collections import deque
import gymnasium as gym

# You would import your PPOPolicy class here
# from your_file import PPOPolicy 

class SafetyModel:
    """
    A wrapper for the Gaussian Process model that learns the safety cost.
    This class acts as the "uncertainty quantifier" from the MASE paper.
    """
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int = 10000):
        """
        Args:
            state_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            buffer_size (int): The maximum number of recent safety data points to store.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input to the GP is the concatenated state and action
        input_dim = state_dim + action_dim
        
        # A standard RBF kernel for the GP is a good starting point
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*input_dim, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-2, normalize_y=True)
        
        # It's crucial to scale the inputs for a GP
        self.scaler = StandardScaler()
        
        # Buffer to store the most recent safety data (s, a, g)
        self.safety_data_buffer = deque(maxlen=buffer_size)
        self.is_fitted = False

    def predict(self, state: np.ndarray, action: np.ndarray):
        """
        Predicts the mean (mu) and standard deviation (sigma) of the safety cost.
        Returns (0.0, 1.0) if the model has not been trained yet.
        """
        if not self.is_fitted:
            return 0.0, 1.0 # Return high uncertainty if not fitted yet
            
        sa_concat = np.hstack([state, action]).reshape(1, -1)
        sa_scaled = self.scaler.transform(sa_concat)
        mean, std = self.gp.predict(sa_scaled, return_std=True)
        
        # Ensure std is non-negative
        return mean[0], max(std[0], 1e-6)

    def update(self):
        """
        Updates (retrains) the Gaussian Process model with the data currently in the buffer.
        """
        if len(self.safety_data_buffer) < 100: # Don't train with too little data
            return

        # Unpack the data from the buffer
        states, actions, costs = zip(*self.safety_data_buffer)
        X = np.hstack([np.array(states), np.array(actions)])
        y = np.array(costs)
        
        # Fit the scaler and transform the data
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Fit the GP model
        print("Updating SafetyModel Gaussian Process...")
        self.gp.fit(X_scaled, y)
        self.is_fitted = True
        print("SafetyModel GP update complete.")

    def add_data_point(self, state: np.ndarray, action: np.ndarray, cost: float):
        """Adds a new safety data point (s, a, g) to the buffer."""
        self.safety_data_buffer.append((state, action, cost))

class MASE:
    """
    The MASE meta-algorithm for safe exploration.
    This class wraps the underlying RL policy and the safety model to provide
    the proactive shielding logic.
    """
    def __init__(self, state_dim: int, action_space: gym.Space, args: dict):
        """
        Args:
            state_dim (int): Dimension of the observation space.
            action_space (gym.Space): The environment's action space.
            args (dict): A dictionary of hyperparameters for MASE.
        """
        self.safety_model = SafetyModel(state_dim, action_space.shape[0])
        self.action_space = action_space
        
        # MASE hyperparameters from the paper
        self.beta = args.get('beta', 2.5) # Confidence bound parameter (e.g., 2.5 for >99% confidence)
        self.penalty_c = args.get('penalty_c', 100.0) # Penalty multiplier for emergency stop
        self.num_dead_end_checks = args.get('num_dead_end_checks', 30) # Num actions to sample for the lookahead check

    def get_uncertainty_bound(self, state: np.ndarray, action: np.ndarray) -> float:
        """Calculates Gamma(s, a), the uncertainty quantifier from the paper."""
        _, std = self.safety_model.predict(state, action)
        return self.beta * std

    def is_action_safe(self, state: np.ndarray, action: np.ndarray, safety_threshold: float) -> bool:
        """Checks if an action `a` belongs to the safe set A+ for a given state `s`."""
        mean, _ = self.safety_model.predict(state, action)
        gamma = self.get_uncertainty_bound(state, action)
        # The core safety condition from the paper
        return mean + gamma <= safety_threshold

    def check_for_dead_end(self, state: np.ndarray, safety_threshold: float) -> bool:
        """
        Performs the lookahead check to see if a state is a "dead end"
        (i.e., has no viable safe actions).
        """
        if not self.safety_model.is_fitted:
            return False # Assume not a dead end if the safety model isn't ready

        # To check for a dead end, we sample a number of random actions and see if ANY of them are safe.
        for _ in range(self.num_dead_end_checks):
            random_action = self.action_space.sample()
            if self.is_action_safe(state, random_action, safety_threshold):
                return False # Found at least one safe action, so this is NOT a dead end
        
        # If the loop finishes without finding any safe actions, it's a dead end.
        return True

    def get_emergency_penalty(self, state: np.ndarray) -> float:
        """
        Calculates the penalty reward when an emergency stop is triggered.
        The penalty is inversely proportional to the minimum uncertainty at the dead-end state.
        """
        min_uncertainty = np.inf
        for _ in range(self.num_dead_end_checks):
            random_action = self.action_space.sample()
            gamma = self.get_uncertainty_bound(state, random_action)
            if gamma < min_uncertainty:
                min_uncertainty = gamma
        
        return -self.penalty_c / (min_uncertainty + 1e-6)

    def update_safety_model(self):
        """A convenience method to trigger the GP update."""
        self.safety_model.update()

    def add_safety_data(self, state: np.ndarray, action: np.ndarray, cost: float):
        """A convenience method to add data to the safety model."""
        self.safety_model.add_data_point(state, action, cost)
