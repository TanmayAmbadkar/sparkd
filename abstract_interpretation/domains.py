import numpy as np
import torch
from scipy.optimize import linprog

class Zonotope:
    def __init__(self, center, generators):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.generators = [torch.tensor(g, dtype=torch.float32) for g in generators]
        self.inequalities = self.to_hyperplanes()
    
    def affine_transform(self, W, b):
        W = torch.tensor(W, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        new_center = W @ self.center + b
        new_generators = [W @ g for g in self.generators]
        return Zonotope(new_center, new_generators)
    
    def relu(self):
        new_center = torch.relu(self.center)
        new_generators = []
        for g in self.generators:
            new_g = torch.where(self.center > 0, g, torch.zeros_like(g))
            new_generators.append(new_g)
        return Zonotope(new_center, new_generators)
    
    def sigmoid(self):
        return self._nonlinear_transform(torch.sigmoid)

    def tanh(self):
        return self._nonlinear_transform(torch.tanh)

    def _nonlinear_transform(self, func):
        new_center = func(self.center)

        new_generators = []
        for g in self.generators:
            # Calculate lower and upper bounds of the linear approximation
            lower = self.center - torch.norm(g, p=1)  # Approximate lower bound
            upper = self.center + torch.norm(g, p=1)  # Approximate upper bound
            
            # Apply the nonlinear function to the bounds
            func_lower = func(lower)
            func_upper = func(upper)

            # Compute slopes λ and λ'
            lambda_ = (func_upper - func_lower) / (upper - lower + 1e-9)  # Avoid division by zero
            lambda_prime = torch.minimum(func_lower * (1 - func_lower), func_upper * (1 - func_upper))

            # Define new generators based on the slopes
            new_g = g * torch.where(self.center > 0, lambda_, lambda_prime)
            new_generators.append(new_g)

        return Zonotope(new_center, new_generators)
    
    def to_hyperplanes(self):
        """
        Convert the zonotope to a set of hyperplane inequalities.
        Each generator contributes two hyperplanes.
        """
        c = self.center  # Assuming center is already a NumPy array
        G = np.array(self.generators)  # Convert the list of generators to a NumPy array
        
        inequalities = []
        for g in G:
            # For each generator, create two inequalities representing the positive and negative halfspaces
            norm_positive = np.dot(g, c) + np.linalg.norm(g)  # Positive direction bound
            norm_negative = -np.dot(g, c) + np.linalg.norm(g)  # Negative direction bound
            
            # Append the inequality for the positive direction
            inequalities.append((g, norm_positive))  # Ax <= b for positive direction
            # Append the inequality for the negative direction
            inequalities.append((-g, norm_negative))  # Ax <= b for negative direction
            
        return inequalities

    
   
    
    def in_zonotope(self, y):
        """
        Check whether the numpy array `y` is contained within the zonotope using Linear Programming.
        """
        y = np.array(y, dtype=np.float32)
        G = np.array([g.numpy() for g in self.generators])
        c = self.center.numpy()
        
        # Number of generators
        num_generators = G.shape[0]
        
        # Objective: Minimize the auxiliary variable t
        # The variable vector x will have size (num_generators + 1) where the last element is t
        c_lp = np.zeros(num_generators + 1)
        c_lp[-1] = 1  # We want to minimize the last variable (t)
        
        # Constraints: y = Gx + c, and -t <= x_i <= t
        A_eq = np.hstack([G.T, np.zeros((G.shape[1], 1))])  # G * x = y - c, so A_eq is G and b_eq is y - c
        b_eq = y - c
        
        # Inequality constraints for the t variable (infinity norm)
        A_ub = np.vstack([np.hstack([np.eye(num_generators), -np.ones((num_generators, 1))]),
                          np.hstack([-np.eye(num_generators), -np.ones((num_generators, 1))])])
        b_ub = np.ones(2 * num_generators)
        
        # Bounds: x_i has no explicit bounds; t >= 0
        bounds = [(None, None)] * num_generators + [(0, None)]
        
        # Solve the LP problem
        res = linprog(c_lp, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method='highs')
        
        # Check if the solution is feasible and if t <= 1
        if res.success and res.x[-1] <= 1:
            return True
        else:
            return False
    
    
    
class Box:
    def __init__(self, lower, upper):
        self.lower = torch.tensor(lower, dtype=torch.float32)
        self.upper = torch.tensor(upper, dtype=torch.float32)
    
    def affine_transform(self, W, b):
        W = torch.tensor(W, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        new_lower = W @ self.lower + b
        new_upper = W @ self.upper + b
        return Box(new_lower, new_upper)
    
    def relu(self):
        new_lower = torch.relu(self.lower)
        new_upper = torch.relu(self.upper)
        return Box(new_lower, new_upper)
