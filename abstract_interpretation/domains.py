import numpy as np
import torch
from scipy.optimize import linprog
import torch
import numpy as np

class Zonotope:
    def __init__(self, center, generators):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.generators = [torch.tensor(g, dtype=torch.float32) for g in generators]
        self.inequalities = self.to_hyperplanes()

    def affine_transform(self, W, b):
        """
        Apply an affine transformation W * x + b to the zonotope.
        """
        W = torch.tensor(W, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        new_center = W @ self.center + b
        new_generators = [W @ g for g in self.generators]
        return Zonotope(new_center, new_generators)

    def relu(self):
        """
        Apply the ReLU transformation with optimal approximation.
        This minimizes the area of the parallelogram in the input-output plane.
        """
        new_center = torch.relu(self.center)
        new_generators = []
        
        for g in self.generators:
            lower = self.center - torch.norm(g, p=1)  # Approximate lower bound
            upper = self.center + torch.norm(g, p=1)  # Approximate upper bound

            # Check if ReLU is exact (lx > 0 or ux <= 0)
            if torch.all(lower >= 0):  # Positive region: y = x
                new_generators.append(g)
            elif torch.all(upper <= 0):  # Non-positive region: y = 0
                new_generators.append(torch.zeros_like(g))
            else:
                # Mixed case: lx < 0 < ux
                lambda_opt = upper / (upper - lower + 1e-9)  # Optimal slope (minimizes area)
                new_g = lambda_opt * g  # Modify the generator by optimal slope
                new_generators.append(new_g)
        
        return Zonotope(new_center, new_generators)

    def sigmoid(self):
        """
        Apply the Sigmoid transformation with optimal approximation.
        """
        return self._nonlinear_transform(torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)))

    def tanh(self):
        """
        Apply the Tanh transformation with optimal approximation.
        """
        return self._nonlinear_transform(torch.tanh, lambda x: 1 - torch.tanh(x) ** 2)

    def _nonlinear_transform(self, func, func_prime):
        """
        Generalized nonlinear transformation for Sigmoid and Tanh with optimal approximation.
        """
        new_center = func(self.center)
        new_generators = []

        for g in self.generators:
            lower = self.center - torch.norm(g, p=1)  # Approximate lower bound
            upper = self.center + torch.norm(g, p=1)  # Approximate upper bound

            # Apply the non-linear function to the bounds
            func_lower = func(lower)
            func_upper = func(upper)

            # Compute optimal slope λ
            lambda_opt = (func_upper - func_lower) / (upper - lower + 1e-9)  # Avoid division by zero

            # Define new generators based on the optimal slope
            new_g = g * lambda_opt
            new_generators.append(new_g)

        return Zonotope(new_center, new_generators)

    def to_hyperplanes(self):
        """
        Convert the zonotope to a set of hyperplane inequalities.
        Each generator contributes two hyperplanes.
        """
        inequalities = []
        for g in self.generators:
            norm_positive = np.dot(g, self.center) + np.linalg.norm(g.numpy())  # Positive direction bound
            norm_negative = -np.dot(g, self.center) + np.linalg.norm(g.numpy())  # Negative direction bound
            
            # Append the inequality for the positive direction
            inequalities.append((g.numpy(), norm_positive))  # Ax <= b for positive direction
            # Append the inequality for the negative direction
            inequalities.append((-g.numpy(), norm_negative))  # Ax <= b for negative direction
        
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
        c_lp = np.zeros(num_generators + 1)
        c_lp[-1] = 1  # Minimize the last variable (t)
        
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
    
    def __repr__(self):
        """
        Return a string representation of the bounds.
        """
        return f"DeepPolyDomain(center={self.center.numpy()}, generators={self.generators})"

    
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

class DeepPoly:
    def __init__(self, lower_bounds, upper_bounds):
        """
        Initialize the DeepPoly domain with lower and upper bounds.
        """
        self.lower = torch.tensor(lower_bounds, dtype=torch.float32)
        self.upper = torch.tensor(upper_bounds, dtype=torch.float32)
        if self.lower.shape != self.upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
    
    def affine_transform(self, W, b, max_iter=10, tol=1e-6):
        """
        Apply affine transformation (W * x + b) iteratively to refine bounds.
        Incorporates features from the AffineTransform class.

        Args:
            W (torch.Tensor): Weight matrix of shape (output_dim, input_dim).
            b (torch.Tensor): Bias vector of shape (output_dim,).
            previous_transformer (DeepPoly): Transformer from the previous layer (optional).
            max_iter (int): Maximum number of iterations for refinement.
            tol (float): Tolerance for stabilization.

        Returns:
            DeepPoly: New DeepPoly domain with updated bounds.
        """
        W = torch.tensor(W, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)

        pos_w = W >= 0.0
        neg_w = W < 0.0

            # No backsubstitution
        ub = self.upper @ (pos_w.T * W.T) + self.lower @ (neg_w.T * W.T) + b
        lb = self.lower @ (pos_w.T * W.T) + self.upper @ (neg_w.T * W.T) + b

        return DeepPoly(lb, ub)



    def relu(self):
        """
        Apply ReLU activation, following the three cases:
        Case 1: u_j <= 0 -> l'_j = u'_j = 0
        Case 2: l_j >= 0 -> l'_j = l_j, u'_j = u_j
        Case 3: l_j < 0 < u_j -> l'_j = λ * l_j, u'_j = u_j
        """
        new_lower = self.lower.clone()
        new_upper = self.upper.clone()

        # Case 1: u_j <= 0 -> l'_j = u'_j = 0
        negative_mask = (self.upper <= 0)
        new_lower[negative_mask] = 0
        new_upper[negative_mask] = 0

        # Case 2: l_j >= 0 -> l'_j = l_j, u'_j = u_j (keep bounds as-is)
        # No change needed for positive_mask = (self.lower >= 0)

        # Case 3: l_j < 0 < u_j
        mixed_mask = (self.lower < 0) & (self.upper > 0)
        new_upper[mixed_mask] = self.upper[mixed_mask]  # u'_j = u_j

        # Compute λ = u_j / (u_j - l_j)
        lambda_val = torch.zeros_like(self.lower)
        lambda_val[mixed_mask] = self.upper[mixed_mask] / (self.upper[mixed_mask] - self.lower[mixed_mask])

        # l'_j = λ * l_j
        new_lower[mixed_mask] = lambda_val[mixed_mask] * self.lower[mixed_mask]

        return DeepPoly(new_lower, new_upper)

    def sigmoid(self):
        """
        Apply Sigmoid activation function, using the abstract transformer method.
        """
        return self.sigmoid_tanh_transform(torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)))

    def tanh(self):
        """
        Apply Tanh activation function, using the abstract transformer method.
        """
        return self.sigmoid_tanh_transform(torch.tanh, lambda x: 1 - torch.tanh(x) ** 2)

    def sigmoid_tanh_transform(self, func, func_prime):
        """
        Generalized abstract transformer for sigmoid and tanh functions.
        :param func: The activation function (sigmoid or tanh).
        :param func_prime: The derivative of the activation function.
        """
        new_lower = func(self.lower)
        new_upper = func(self.upper)

        # Handle the case where bounds are equal (no approximation needed)
        exact_mask = (self.lower == self.upper)
        new_lower[exact_mask] = func(self.lower[exact_mask])
        new_upper[exact_mask] = func(self.upper[exact_mask])

        # For non-equal bounds, compute approximations
        diff_mask = ~exact_mask
        lambda_prime = torch.minimum(func_prime(self.lower[diff_mask]), func_prime(self.upper[diff_mask]))

        new_lower[diff_mask] = new_lower[diff_mask]
        new_upper[diff_mask] = new_upper[diff_mask] + lambda_prime * (self.upper[diff_mask] - self.upper[diff_mask])

        return DeepPoly(new_lower, new_upper)

    def to_hyperplanes(self):
        """
        Convert the box domain to a set of hyperplane inequalities.
        Each dimension contributes two hyperplanes.
        """
        inequalities = []
        for i in range(self.lower.shape[0]):
            # Upper bound constraint: A[i] * x[i] <= u_i
            A_upper = np.zeros(self.lower.shape[0])
            A_upper[i] = 1
            inequalities.append(np.append(A_upper, -self.upper[i]))

            # Lower bound constraint: A[i] * x[i] >= l_i, or -A[i] * x[i] <= -l_i
            A_lower = np.zeros(self.lower.shape[0])
            A_lower[i] = -1
            inequalities.append(np.append(A_lower, self.lower[i]))

        return inequalities

    def __repr__(self):
        """
        Return a string representation of the bounds.
        """
        return f"DeepPolyDomain(lower={np.round(self.lower.numpy(), 2)}, upper={np.round(self.upper.numpy(), 2)})"

