import numpy as np
from itertools import product

class Box:
    """
    A geometric Box domain implemented using NumPy.
    It represents an N-dimensional axis-aligned bounding box or a batch of them.
    """
    def __init__(self, lower_bounds, upper_bounds):
        """
        Initialize the Box domain with lower and upper bounds.

        Args:
            lower_bounds (np.ndarray): Lower bounds, shape (n,) or (B, n).
            upper_bounds (np.ndarray): Upper bounds, shape (n,) or (B, n).
        """
        self.lower = np.asarray(lower_bounds, dtype=np.float64)
        self.upper = np.asarray(upper_bounds, dtype=np.float64)

        if self.lower.ndim == 1:
            self.lower = np.expand_dims(self.lower, axis=0)
            self.upper = np.expand_dims(self.upper, axis=0)

        if self.lower.shape != self.upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")

    def to_hyperplanes(self, observation_space=None):
        """
        Convert the DeepPoly domain into a set of hyperplane inequalities.
        Each dimension yields two hyperplanes.
        
        If the domain is batched (shape (B, n)), the function returns a list of length B,
        where each entry is a list of 2*n hyperplane inequalities (each a NumPy array).
        """
        lower, upper = self.lower, self.upper
        
        if lower.ndim == 1:
            dims = lower.shape[0]
            inequalities = []
            for i in range(dims):
                A_upper = np.zeros(dims)
                A_upper[i] = 1
                A_lower = np.zeros(dims)
                A_lower[i] = -1
                if observation_space is not None:
                    if not np.allclose(observation_space.high[i], upper[i]):
                        inequalities.append(np.append(A_upper, -upper[i]))
                    if not np.allclose(observation_space.low[i], lower[i]):
                        inequalities.append(np.append(A_lower, lower[i]))
                else:                    
                    inequalities.append(np.append(A_upper, -upper[i]))    
                    inequalities.append(np.append(A_lower, lower[i]))
                
            return inequalities
        else:
            B, dims = lower.shape
            all_inequalities = []
            for b in range(B):
                inequalities = []
                for i in range(dims):
                    A_upper = np.zeros(dims)
                    A_upper[i] = 1
                    A_lower = np.zeros(dims)
                    A_lower[i] = -1
                    if observation_space is not None:
                        if not np.allclose(observation_space.high[i], upper[b, i]):
                            inequalities.append(np.append(A_upper, -upper[b, i]))
                        if not np.allclose(observation_space.low[i], lower[b, i]):
                            inequalities.append(np.append(A_lower, lower[b, i]))
                    else:
                        inequalities.append(np.append(A_upper, -upper[b, i]))
                        inequalities.append(np.append(A_lower, lower[b, i]))
                    
                    # inequalities.append(np.append(A_upper, -upper[b, i].item()))
                    # inequalities.append(np.append(A_lower, lower[b, i].item()))
                all_inequalities.append(np.array(inequalities))
                
            return all_inequalities

    def invert_polytope(self, observation_space=None):
        """
        Computes the hyperplanes for the inverted (unsafe) regions.
        This represents the area outside the box.
        """
        B, dims = self.lower.shape
        all_inverted_polytopes = []

        for b in range(B):
            inverted_polytopes_for_batch = []
            for i in range(dims):
                # Region x_i > upper[b, i] => -x_i + upper[b, i] < 0
                A_upper_inv = np.zeros(dims + 1)
                A_upper_inv[i] = -1
                A_upper_inv[-1] = self.upper[b, i] + 1e-6 # Add epsilon for strict inequality

                # Region x_i < lower[b, i] => x_i - lower[b, i] < 0
                A_lower_inv = np.zeros(dims + 1)
                A_lower_inv[i] = 1
                A_lower_inv[-1] = -self.lower[b, i] + 1e-6 # Add epsilon for strict inequality

                if observation_space is not None:
                    if observation_space.high[i] != self.upper[b, i]:
                        inverted_polytopes_for_batch.append(np.expand_dims(A_upper_inv, axis=0))
                    if observation_space.low[i] != self.lower[b, i]:
                        inverted_polytopes_for_batch.append(np.expand_dims(A_lower_inv, axis=0))
                else:
                    inverted_polytopes_for_batch.append(np.expand_dims(A_upper_inv, axis=0))
                    inverted_polytopes_for_batch.append(np.expand_dims(A_lower_inv, axis=0))
            
            all_inverted_polytopes.append(inverted_polytopes_for_batch)
        
        return all_inverted_polytopes[0] if B == 1 else all_inverted_polytopes

    def intersects(self, other: 'Box') -> np.ndarray:
        """
        Check whether this Box domain (or each element in a batched domain)
        intersects with another.
        """
        other_lower = other.lower
        other_upper = other.upper
        
        if other_lower.ndim == 1:
            other_lower = np.expand_dims(other_lower, axis=0)
            other_upper = np.expand_dims(other_upper, axis=0)
        
        return np.all(self.lower < other_upper, axis=1) & np.all(self.upper > other_lower, axis=1)

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        """
        Return a string representation.
        """
        return (f"Box(batch_shape={self.lower.shape}, "
                f"first_sample_lower={np.round(self.lower[0], 4)}, "
                f"first_sample_upper={np.round(self.upper[0], 4)})")

