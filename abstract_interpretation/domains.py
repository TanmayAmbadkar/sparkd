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

# class DeepPoly:
#     def __init__(self, lower_bounds, upper_bounds):
#         """
#         Initialize the DeepPoly domain with lower and upper bounds.
#         """
#         self.lower = torch.tensor(lower_bounds, dtype=torch.float32)
#         self.upper = torch.tensor(upper_bounds, dtype=torch.float32)
#         if self.lower.shape != self.upper.shape:
#             raise ValueError("Lower and upper bounds must have the same shape.")
    
#     def affine_transform(self, W, b):
#         """
#         Perform affine transformation and compute bounds.

#         Args:
#             W (torch.Tensor): Weight matrix of shape (output_dim, input_dim).
#             b (torch.Tensor): Bias vector of shape (output_dim,).

#         Returns:
#             DeepPoly: New DeepPoly domain with updated bounds.
#         """
#         W = torch.tensor(W, dtype=torch.float32)
#         b = torch.tensor(b, dtype=torch.float32)

#         pos_w = W >= 0.0
#         neg_w = W < 0.0

#             # No backsubstitution
#         ub = self.upper @ (pos_w.T * W.T) + self.lower @ (neg_w.T * W.T) + b
#         lb = self.lower @ (pos_w.T * W.T) + self.upper @ (neg_w.T * W.T) + b

#         return DeepPoly(lb, ub)


#     def relu(self):
#         """
#         Apply ReLU activation, following the three cases:
#         Case 1: u_j <= 0 -> l'_j = u'_j = 0
#         Case 2: l_j >= 0 -> l'_j = l_j, u'_j = u_j
#         Case 3: l_j < 0 < u_j -> l'_j = λ * l_j, u'_j = u_j
#         """
#         new_lower = self.lower.clone()
#         new_upper = self.upper.clone()

#         # Case 1: u_j <= 0 -> l'_j = u'_j = 0
#         negative_mask = (self.upper <= 0)
#         new_lower[negative_mask] = 0
#         new_upper[negative_mask] = 0

#         # Case 2: l_j >= 0 -> l'_j = l_j, u'_j = u_j (keep bounds as-is)
#         # No change needed for positive_mask = (self.lower >= 0)

#         # Case 3: l_j < 0 < u_j
#         mixed_mask = (self.lower < 0) & (self.upper > 0)
#         new_upper[mixed_mask] = self.upper[mixed_mask]  # u'_j = u_j

#         # Compute λ = u_j / (u_j - l_j)
#         lambda_val = torch.zeros_like(self.lower)
#         lambda_val[mixed_mask] = self.upper[mixed_mask] / (self.upper[mixed_mask] - self.lower[mixed_mask])

#         # l'_j = λ * l_j
#         new_lower[mixed_mask] = lambda_val[mixed_mask] * self.lower[mixed_mask]

#         return DeepPoly(new_lower, new_upper)

#     def sigmoid(self):
#         """
#         Apply Sigmoid activation function, using the abstract transformer method.
#         """
#         return self.sigmoid_tanh_transform(torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)))

#     def tanh(self):
#         """
#         Apply Tanh activation function, using the abstract transformer method.
#         """
#         return self.sigmoid_tanh_transform(torch.tanh, lambda x: 1 - torch.tanh(x) ** 2)

#     def sigmoid_tanh_transform(self, func, func_prime):
#         """
#         Generalized abstract transformer for sigmoid and tanh functions.
#         :param func: The activation function (sigmoid or tanh).
#         :param func_prime: The derivative of the activation function.
#         """
#         new_lower = func(self.lower)
#         new_upper = func(self.upper)

#         # Handle the case where bounds are equal (no approximation needed)
#         exact_mask = (self.lower == self.upper)
#         new_lower[exact_mask] = func(self.lower[exact_mask])
#         new_upper[exact_mask] = func(self.upper[exact_mask])

#         # For non-equal bounds, compute approximations
#         diff_mask = ~exact_mask
#         lambda_prime = torch.minimum(func_prime(self.lower[diff_mask]), func_prime(self.upper[diff_mask]))

#         new_lower[diff_mask] = new_lower[diff_mask]
#         new_upper[diff_mask] = new_upper[diff_mask] + lambda_prime * (self.upper[diff_mask] - self.upper[diff_mask])

#         return DeepPoly(new_lower, new_upper)

#     def to_hyperplanes(self):
#         """
#         Convert the box domain to a set of hyperplane inequalities.
#         Each dimension contributes two hyperplanes.
#         """
#         inequalities = []
#         for i in range(self.lower.shape[0]):
#             # Upper bound constraint: A[i] * x[i] <= u_i
#             A_upper = np.zeros(self.lower.shape[0])
#             A_upper[i] = 1
#             inequalities.append(np.append(A_upper, -self.upper[i]))

#             # Lower bound constraint: A[i] * x[i] >= l_i, or -A[i] * x[i] <= -l_i
#             A_lower = np.zeros(self.lower.shape[0])
#             A_lower[i] = -1
#             inequalities.append(np.append(A_lower, self.lower[i]))

#         return inequalities

#     def __repr__(self):
#         """
#         Return a string representation of the bounds.
#         """
#         return f"DeepPolyDomain(lower={np.round(self.lower.numpy(), 2)}, upper={np.round(self.upper.numpy(), 2)})"

#     def intersects(self, other):
#         """
#         Check if this box intersects with another box.
#         """
#         return torch.all(self.lower < other.upper) and torch.all(self.upper > other.lower)

#     def subtract(self, other):
#         """
#         Subtract another DeepPoly box from this box.
#         Returns a list of resulting DeepPoly boxes after subtraction.
#         """
#         if not self.intersects(other):
#             return [self]  # No intersection, return the original box

#         resulting_boxes = []
#         for dim in range(len(self.lower)):
#             if other.lower[dim] > self.lower[dim]:
#                 # Create a box below the intersection along this dimension
#                 new_lower = self.lower.clone()
#                 new_upper = self.upper.clone()
#                 new_upper[dim] = other.lower[dim]
#                 resulting_boxes.append(DeepPoly(new_lower.tolist(), new_upper.tolist()))

#             if other.upper[dim] < self.upper[dim]:
#                 # Create a box above the intersection along this dimension
#                 new_lower = self.lower.clone()
#                 new_upper = self.upper.clone()
#                 new_lower[dim] = other.upper[dim]
#                 resulting_boxes.append(DeepPoly(new_lower.tolist(), new_upper.tolist()))

#         return resulting_boxes

class DeepPoly:
    def __init__(self, lower_bounds, upper_bounds, A_L=None, c_L=None, A_U=None, c_U=None):
        """
        Initialize the DeepPoly domain with lower and upper bounds.
        No relational constraints are given.
        """
        self.lower = torch.tensor(lower_bounds, dtype=torch.float32)
        self.upper = torch.tensor(upper_bounds, dtype=torch.float32)
        if self.lower.shape != self.upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")

        input_size = self.lower.shape[0]
        if A_L is None:
            # Initialize affine expressions for input variables
            self.A_L = -torch.eye(input_size, dtype=torch.float32)
            self.A_U = torch.eye(input_size, dtype=torch.float32)
            self.c_L = -self.lower.clone()
            self.c_U = -self.upper.clone()
        else:
            self.A_L = A_L
            self.c_L = c_L
            self.A_U = A_U
            self.c_U = c_U

    def affine_transform(self, W, b):
        """
        Perform affine transformation and compute bounds using the abstract affine transformer
        with bounds substitution (recursive substitution of affine expressions).

        Args:
            W (torch.Tensor): Weight matrix of shape (output_dim, input_dim).
            b (torch.Tensor): Bias vector of shape (output_dim,).

        Returns:
            DeepPoly: New DeepPoly domain with updated bounds.
        """
        W = W.clone().detach().float()
        b = b.clone().detach().float()
        output_dim, input_dim = W.shape

        # Initialize new affine expressions
        new_A_L = []
        new_c_L = []
        new_A_U = []
        new_c_U = []
        new_lower = []
        new_upper = []

        # For each output neuron i
        for i in range(output_dim):
            # Initialize affine expressions for x_i
            # a'_i^≤ x = v_i + sum_j w_ij * x_j
            a_L_i = W[i, :].clone()
            c_L_i = b[i].clone()
            a_U_i = W[i, :].clone()
            c_U_i = b[i].clone()

            # Compute l'_i by substituting bounds recursively
            l_i = self.compute_bound(a_L_i, c_L_i, lower=True)

            # Compute u'_i by substituting bounds recursively
            u_i = self.compute_bound(a_U_i, c_U_i, lower=False)

            # Since we have substituted all variables, set affine expressions to zero
            # Append the constants as new affine expressions (which are constants)
            new_A_L.append(torch.zeros_like(a_L_i))
            new_c_L.append(torch.tensor(l_i))
            new_A_U.append(torch.zeros_like(a_U_i))
            new_c_U.append(torch.tensor(u_i))

            # Update bounds
            new_lower.append(l_i)
            new_upper.append(u_i)

        # Convert lists to tensors
        new_A_L = torch.stack(new_A_L)
        new_c_L = torch.stack(new_c_L)
        new_A_U = torch.stack(new_A_U)
        new_c_U = torch.stack(new_c_U)
        new_lower = torch.tensor(new_lower)
        new_upper = torch.tensor(new_upper)

        # Create new DeepPoly domain with updated variables
        return DeepPoly(new_lower, new_upper, new_A_L, new_c_L, new_A_U, new_c_U)

    def compute_bound(self, a_i, c_i, lower=True):
        """
        Recursively compute the bound (lower or upper) by substituting affine expressions.

        Args:
            a_i (torch.Tensor): Affine coefficients for the current neuron.
            c_i (float): Constant term for the current neuron.
            lower (bool): True for lower bound, False for upper bound.

        Returns:
            bound (float): Computed lower or upper bound.
        """
        # Initialize the current affine expression
        a_current = a_i.clone()
        c_current = c_i.clone()

        # Start substitution loop
        while True:
            # Indices of variables to substitute
            indices_to_substitute = (a_current != 0).nonzero(as_tuple=True)[0]
            new_indices = []

            substitution_made = False

            for idx in indices_to_substitute:
                if idx >= self.lower.shape[0]:
                    # Variable is from previous layer, substitute its affine expression
                    coeff = a_current[idx]
                    a_current[idx] = 0  # Remove the variable

                    if lower:
                        a_current += coeff * self.A_L[idx]
                        c_current += coeff * self.c_L[idx]
                    else:
                        a_current += coeff * self.A_U[idx]
                        c_current += coeff * self.c_U[idx]

                    substitution_made = True
                else:
                    # Input variable, cannot substitute further
                    new_indices.append(idx)

            if not substitution_made:
                # No more substitutions can be made
                break

        # Now, a_current contains only input variables
        if lower:
            # Compute lower bound
            pos_coeffs = torch.clamp(a_current, min=0)
            neg_coeffs = torch.clamp(a_current, max=0)
            bound = c_current + torch.dot(pos_coeffs, self.lower) + torch.dot(neg_coeffs, self.upper)
        else:
            # Compute upper bound
            pos_coeffs = torch.clamp(a_current, min=0)
            neg_coeffs = torch.clamp(a_current, max=0)
            bound = c_current + torch.dot(pos_coeffs, self.upper) + torch.dot(neg_coeffs, self.lower)

        return bound.item()


    
    def relu(self):
        """
        Apply ReLU activation, following the abstract transformer from the DeepPoly paper.
        """
        new_lower = self.lower.clone()
        new_upper = self.upper.clone()
        new_A_L = self.A_L.clone()
        new_c_L = self.c_L.clone()
        new_A_U = self.A_U.clone()
        new_c_U = self.c_U.clone()

        # Compute masks for the three cases
        case1 = self.upper <= 0  # u_j <= 0
        case2 = self.lower >= 0  # l_j >= 0
        case3 = (~case1) & (~case2)  # l_j < 0 < u_j

        # Handle Case 1: u_j <= 0
        idx_case1 = case1.nonzero(as_tuple=True)[0]
        new_lower[idx_case1] = 0
        new_upper[idx_case1] = 0
        new_A_L[idx_case1, :] = 0
        new_c_L[idx_case1] = 0
        new_A_U[idx_case1, :] = 0
        new_c_U[idx_case1] = 0

        # Handle Case 2: l_j >= 0
        # No changes needed; affine expressions and bounds remain the same

        # Handle Case 3: l_j < 0 < u_j
        idx_case3 = case3.nonzero(as_tuple=True)[0]
        l_j = self.lower[idx_case3]
        u_j = self.upper[idx_case3]

        # Upper bound: x_i ≤ (u_j / (u_j - l_j))(x_j - l_j)
        lambda_u = u_j / (u_j - l_j)
        new_A_U[idx_case3, :] = lambda_u.unsqueeze(1) * self.A_U[idx_case3, :]
        new_c_U[idx_case3] = lambda_u * (self.c_U[idx_case3] - l_j)

        # Update new_upper
        pos_coeffs = torch.clamp(new_A_U[idx_case3, :], min=0)
        neg_coeffs = torch.clamp(new_A_U[idx_case3, :], max=0)
        new_upper[idx_case3] = new_c_U[idx_case3] + pos_coeffs @ self.upper + neg_coeffs @ self.lower

        # Lower bound: Choose λ ∈ {0,1} that minimizes the area
        # According to the paper, we can choose λ = 0 when l_j < -u_j, else λ = 1
        # For simplicity, we'll choose λ = 0 since l_j < 0
        new_A_L[idx_case3, :] = 0
        new_c_L[idx_case3] = 0
        new_lower[idx_case3] = 0

        return DeepPoly(new_lower, new_upper, new_A_L, new_c_L, new_A_U, new_c_U)


    def sigmoid(self):
        """
        Apply the Sigmoid activation function using the abstract transformer.
        """
        return self.activation_transform(
            func=torch.sigmoid,
            func_prime=lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        )

    def tanh(self):
        """
        Apply the Tanh activation function using the abstract transformer.
        """
        return self.activation_transform(
            func=torch.tanh,
            func_prime=lambda x: 1 - torch.tanh(x) ** 2
        )

    def activation_transform(self, func, func_prime):
        """
        General method for applying activation functions using abstract transformers.
        """
        l_j = self.lower
        u_j = self.upper

        l_prime = func(l_j)
        u_prime = func(u_j)

        new_lower = l_prime.clone()
        new_upper = u_prime.clone()
        new_A_L = self.A_L.clone()
        new_c_L = self.c_L.clone()
        new_A_U = self.A_U.clone()
        new_c_U = self.c_U.clone()

        # Identify indices where l_j == u_j
        equal_mask = (l_j == u_j)
        idx_equal = equal_mask.nonzero(as_tuple=True)[0]

        # Handle the case where l_j == u_j
        new_A_L[idx_equal, :] = 0
        new_c_L[idx_equal] = l_prime[idx_equal]
        new_A_U[idx_equal, :] = 0
        new_c_U[idx_equal] = u_prime[idx_equal]

        # Indices where l_j != u_j
        idx_neq = (~equal_mask).nonzero(as_tuple=True)[0]

        if idx_neq.numel() > 0:
            l_j_neq = l_j[idx_neq]
            u_j_neq = u_j[idx_neq]
            A_L_neq = self.A_L[idx_neq, :]
            c_L_neq = self.c_L[idx_neq]
            A_U_neq = self.A_U[idx_neq, :]
            c_U_neq = self.c_U[idx_neq]

            # Compute lambda and lambda_prime
            denominator = u_j_neq - l_j_neq
            # Avoid division by zero
            denominator = torch.where(denominator == 0, torch.full_like(denominator, 1e-6), denominator)
            lambda_val = (func(u_j_neq) - func(l_j_neq)) / denominator
            lambda_prime = torch.min(func_prime(l_j_neq), func_prime(u_j_neq))

            # For lower affine expression
            l_positive_mask = (l_j_neq > 0)
            idx_l_positive = idx_neq[l_positive_mask]
            idx_l_nonpositive = idx_neq[~l_positive_mask]

            # Update lower affine expressions where l_j > 0
            if idx_l_positive.numel() > 0:
                lambda_lp = lambda_val[l_positive_mask]
                new_A_L[idx_l_positive, :] = lambda_lp.unsqueeze(1) * A_L_neq[l_positive_mask, :]
                new_c_L[idx_l_positive] = lambda_lp * c_L_neq[l_positive_mask] + \
                                          (func(l_j_neq[l_positive_mask]) - lambda_lp * l_j_neq[l_positive_mask])

            # Update lower affine expressions where l_j <= 0
            if idx_l_nonpositive.numel() > 0:
                lambda_lnp = lambda_prime[~l_positive_mask]
                new_A_L[idx_l_nonpositive, :] = lambda_lnp.unsqueeze(1) * A_L_neq[~l_positive_mask, :]
                new_c_L[idx_l_nonpositive] = lambda_lnp * c_L_neq[~l_positive_mask] + \
                                             (func(l_j_neq[~l_positive_mask]) - lambda_lnp * l_j_neq[~l_positive_mask])

            # For upper affine expression
            u_nonpositive_mask = (u_j_neq <= 0)
            idx_u_nonpositive = idx_neq[u_nonpositive_mask]
            idx_u_positive = idx_neq[~u_nonpositive_mask]

            # Update upper affine expressions where u_j <= 0
            if idx_u_nonpositive.numel() > 0:
                lambda_unp = lambda_prime[u_nonpositive_mask]
                new_A_U[idx_u_nonpositive, :] = lambda_unp.unsqueeze(1) * A_U_neq[u_nonpositive_mask, :]
                new_c_U[idx_u_nonpositive] = lambda_unp * c_U_neq[u_nonpositive_mask] + \
                                             (func(u_j_neq[u_nonpositive_mask]) - lambda_unp * u_j_neq[u_nonpositive_mask])

            # Update upper affine expressions where u_j > 0
            if idx_u_positive.numel() > 0:
                lambda_up = lambda_val[~u_nonpositive_mask]
                new_A_U[idx_u_positive, :] = lambda_up.unsqueeze(1) * A_U_neq[~u_nonpositive_mask, :]
                new_c_U[idx_u_positive] = lambda_up * c_U_neq[~u_nonpositive_mask] + \
                                          (func(u_j_neq[~u_nonpositive_mask]) - lambda_up * u_j_neq[~u_nonpositive_mask])

        # Return the new DeepPoly domain
        return DeepPoly(new_lower, new_upper, new_A_L, new_c_L, new_A_U, new_c_U)

    def __repr__(self):
        """
        Return a string representation of the bounds.
        """
        return f"DeepPolyDomain(lower={np.round(self.lower.numpy(), 2)}, upper={np.round(self.upper.numpy(), 2)})"



def recover_safe_region(observation_box, unsafe_boxes):
    """
    Recover the safe region by subtracting unsafe boxes from the observation boundary.
    
    Args:
        obs_lower: Lower bounds of the observation boundary (list of floats).
        obs_upper: Upper bounds of the observation boundary (list of floats).
        unsafe_boxes: List of DeepPoly objects representing the unsafe region.
    
    Returns:
        A list of DeepPoly objects representing the safe region.
    """
    # Initialize the observation boundary as a single DeepPoly box

    # Initialize the safe region with the observation boundary
    safe_regions = [observation_box]

    # Iteratively subtract each unsafe box from the safe regions
    for unsafe_box in unsafe_boxes:
        new_safe_regions = []
        for safe_box in safe_regions:
            new_safe_regions.extend(safe_box.subtract(unsafe_box))
        safe_regions = new_safe_regions

    
    return safe_regions

def get_unsafe_region(obs_space, safe_space):
    """
    Find unsafe regions by dividing the observation space into smaller boxes outside the safe space.
    
    Args:
        obs_space: A DeepPoly object representing the observation space.
        safe_space: A DeepPoly object representing the safe space.
    
    Returns:
        A list of DeepPoly boxes representing the unsafe regions.
    """
    
    
    dimensions = obs_space.lower.shape[0]  # Number of dimensions
    unsafe_regions = obs_space.subtract(safe_space)
    # # Iterate over each dimension to create complementary boxes
    # for idx in range(dimensions):
    #     # Create a box below the safe region in this dimension
    #     if obs_space.lower[idx] < safe_space.lower[idx]:
    #         new_low = obs_space.lower.clone()
    #         new_high = obs_space.upper.clone()
    #         new_high[idx] = safe_space.lower[idx]  # Adjust the upper bound of this dimension
    #         unsafe_regions.append(DeepPoly(new_low, new_high))

    #     # Create a box above the safe region in this dimension
    #     if obs_space.upper[idx] > safe_space.upper[idx]:
    #         new_low = obs_space.lower.clone()
    #         new_high = obs_space.upper.clone()
    #         new_low[idx] = safe_space.upper[idx]  # Adjust the lower bound of this dimension
    #         unsafe_regions.append(DeepPoly(new_low, new_high))

    return unsafe_regions


if __name__ ==  "__main__":
    
    poly = DeepPoly(-torch.ones(4), torch.ones(4))
    W = torch.randn(4, 2)
    b = torch.rand(2)

    print(poly.affine_transform(W, b))
        