import numpy as np
import torch

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
    
    def to_hyperplanes(self):
        """
        Convert the zonotope to a set of hyperplane inequalities.
        Each generator contributes two hyperplanes.
        """
        c = self.center.numpy()
        G = np.array([g.numpy() for g in self.generators])

        num_generators = G.shape[0]
        A = np.vstack([G, -G])
        b = np.ones(2 * num_generators)

        inequalities = []
        for i in range(2 * num_generators):
            inequalities.append((A[i], np.dot(A[i], c) + b[i]))
        return inequalities
    
    def in_zonotope(self, x):
        """
        Check whether the numpy array `x` is contained within the zonotope.
        """
        x = np.array(x)
        
        for A, b in self.inequalities:
            if np.dot(A, x) > b:
                return False
        
        return True
    
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
