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
    def __init__(self, lower_bounds, upper_bounds, parent = None, A_L = None, A_U = None):
        """
        
        Initialize the DeepPoly domain with lower and upper bounds.
        No relational constraints are given.
        """
        self.lower = torch.tensor(lower_bounds, dtype=torch.float64)
        self.upper = torch.tensor(upper_bounds, dtype=torch.float64)
        self.parent = parent
        self.name = None
        if self.lower.shape != self.upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")

        if A_L is None:
            input_size = self.lower.shape[0]
            self.A_L = torch.ones((input_size, input_size+1)).double()
            self.A_U = torch.ones((input_size, input_size+1)).double()
            self.A_L[:, :-1] = 0
            self.A_U[:, :-1] = 0
            self.A_L[:,-1] = self.lower
            self.A_U[:,-1] = self.upper
        else:
            self.A_L = A_L
            self.A_U = A_U
        
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
        W = W
        b = b
        output_dim, input_dim = W.shape

        new_A_L = np.hstack([W, b.reshape(-1, 1)])
        new_A_U = np.hstack([W, b.reshape(-1, 1)])
        
        pos_w = W >= 0.0
        neg_w = W < 0.0

            # No backsubstitution
        ub = self.upper @ (pos_w.T * W.T) + self.lower @ (neg_w.T * W.T) + b
        lb = self.lower @ (pos_w.T * W.T) + self.upper @ (neg_w.T * W.T) + b
        
        # Create new DeepPoly domain with updated variables
        self.name = "AFFINE"
        # print("BOUNDS", lb, ub)

        return DeepPoly(lb, ub, self, torch.tensor(new_A_L).double(), torch.tensor(new_A_U).double())

    
    def relu(self):
        """
        Apply ReLU activation, following the abstract transformer from the DeepPoly paper.
        """
        # Create new DeepPoly domain with updated variables
        self.name = "RELU"
        
        
        new_lower = self.lower.clone().detach()
        new_upper = self.upper.clone().detach()
        new_A_L = torch.zeros((self.lower.shape[0], self.lower.shape[0] + 1)).double()
        new_A_U = torch.zeros((self.lower.shape[0], self.lower.shape[0] + 1)).double()
        new_A_L[:,-1] = 0
        new_A_U[:,-1] = 0

        # Compute masks for the three cases
        case1 = self.upper <= 0  # u_j <= 0
        case2 = self.lower >= 0  # l_j >= 0
        case3 = (~case1) & (~case2)  # l_j < 0 < u_j

        # Handle Case 1: u_j <= 0
        idx_case1 = case1.nonzero(as_tuple=True)[0]
        new_lower[idx_case1] = 0
        new_upper[idx_case1] = 0

        # Handle Case 2: l_j >= 0
        
        idx_case2 = case2.nonzero(as_tuple=True)[0]
        if idx_case2.numel() > 0:
            new_A_L[idx_case2, idx_case2] = torch.ones(len(idx_case2)).double()
            new_A_U[idx_case2, idx_case2] = torch.ones(len(idx_case2)).double()
        # No changes needed; affine expressions and bounds remain the same

        # Handle Case 3: l_j < 0 < u_j
        idx_case3 = case3.nonzero(as_tuple=True)[0]
        l_j = self.lower[idx_case3]
        u_j = self.upper[idx_case3]

        # Upper bound: x_i ≤ (u_j / (u_j - l_j))(x_j - l_j)
        lambda_u = u_j / (u_j - l_j)
        new_A_U[idx_case3, idx_case3] = lambda_u * torch.ones(len(idx_case3)).double()
        new_A_U[idx_case3, -1] = lambda_u * (- l_j)
        
        
        # Update new_upper
        pos_coeffs = torch.clamp(new_A_U[idx_case3, idx_case3], min=0)
        neg_coeffs = torch.clamp(new_A_U[idx_case3, idx_case3], max=0)
        new_upper[idx_case3] = new_A_U[idx_case3, -1] + self.upper[idx_case3] * pos_coeffs  + self.lower[idx_case3] * neg_coeffs

        # Lower bound: Choose λ ∈ {0,1} that minimizes the area
        # According to the paper, we can choose λ = 0 when l_j < -u_j, else λ = 1
        # For simplicity, we'll choose λ = 0 since l_j < 0
        new_A_L[idx_case3] = 0
        new_lower[idx_case3] = 0
        # print("BOUNDS", new_lower, new_upper)

        return DeepPoly(new_lower, new_upper, self, new_A_L.double(), new_A_U.double())


    def sigmoid(self):
        """
        Apply the Sigmoid activation function using the abstract transformer.
        """
        self.name = "RELU"
        return self.activation_transform(
            func=torch.sigmoid,
            func_prime=lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        )

    def tanh(self):
        """
        Apply the Tanh activation function using the abstract transformer.
        """
        self.name = "TANH"
        return self.activation_transform(
            func=torch.tanh,
            func_prime=lambda x: 1 - torch.tanh(x) ** 2
        )

    def activation_transform(self, func, func_prime):
        """
        General method for applying activation functions using abstract transformers.
        """
        l_j = self.lower.clone()
        u_j = self.upper.clone()
        new_A_L = torch.zeros((self.lower.shape[0], self.lower.shape[0] + 1)).double()
        new_A_U = torch.zeros((self.lower.shape[0], self.lower.shape[0] + 1)).double()


        l_prime = func(l_j).double()
        u_prime = func(u_j).double()

        # Identify indices where l_j == u_j
        equal_mask = (l_j == u_j)
        idx_equal = equal_mask.nonzero(as_tuple=True)[0]

        # Handle the case where l_j == u_j
        new_A_L[idx_equal, :-1] = 0
        new_A_L[idx_equal, -1] = l_prime[idx_equal]
        new_A_U[idx_equal, :-1] = 0
        new_A_U[idx_equal, -1] = u_prime[idx_equal]

        # Indices where l_j != u_j
        idx_neq = (~equal_mask).nonzero(as_tuple=True)[0]

        if idx_neq.numel() > 0:
            l_j_neq = l_j[idx_neq]
            u_j_neq = u_j[idx_neq]
            A_L_neq = self.A_L[idx_neq]
            A_U_neq = self.A_U[idx_neq]

            # Compute lambda and lambda_prime
            denominator = u_j_neq - l_j_neq
            # Avoid division by zero
            denominator = torch.where(denominator == 0, torch.full_like(denominator, 1e-6), denominator)
            lambda_val = (func(u_j_neq) - func(l_j_neq)) / denominator
            lambda_prime = torch.min(func_prime(l_j_neq), func_prime(u_j_neq)).double()

            # For lower affine expression
            l_positive_mask = (l_j_neq > 0)
            idx_l_positive = idx_neq[l_positive_mask]
            idx_l_nonpositive = idx_neq[~l_positive_mask]

            # Update lower affine expressions where l_j > 0
            if idx_l_positive.numel() > 0:
                lambda_lp = lambda_val[l_positive_mask]
                new_A_L[idx_l_positive, idx_l_positive] = lambda_lp * torch.ones(len(idx_l_positive))
                new_A_L[idx_l_positive, -1] = lambda_lp * new_A_L[idx_l_positive, -1] + \
                                          (func(l_j_neq[idx_l_positive]) - lambda_lp * l_j_neq[idx_l_positive])

            # Update lower affine expressions where l_j <= 0
            if idx_l_nonpositive.numel() > 0:
                lambda_lnp = lambda_prime[~l_positive_mask]
                new_A_L[idx_l_nonpositive, idx_l_nonpositive] = lambda_lnp * torch.ones(len(idx_l_nonpositive))
                new_A_L[idx_l_nonpositive, -1] = lambda_lnp * new_A_L[idx_l_nonpositive, -1] + \
                                             (func(l_j_neq[idx_l_nonpositive]) - lambda_lnp * l_j_neq[idx_l_nonpositive])

            # For upper affine expression
            u_nonpositive_mask = (u_j_neq <= 0)
            idx_u_nonpositive = idx_neq[u_nonpositive_mask]
            idx_u_positive = idx_neq[~u_nonpositive_mask]

            # Update upper affine expressions where u_j <= 0
            if idx_u_nonpositive.numel() > 0:
                lambda_unp = lambda_prime[u_nonpositive_mask]
                new_A_U[idx_u_nonpositive, idx_u_nonpositive] = lambda_unp * torch.ones(len(idx_u_nonpositive))
                new_A_U[idx_u_nonpositive, -1] = lambda_unp * new_A_U[idx_u_nonpositive, -1] + \
                                             (func(u_j_neq[idx_u_nonpositive]) - lambda_unp * u_j_neq[idx_u_nonpositive])

            # Update upper affine expressions where u_j > 0
            if idx_u_positive.numel() > 0:
                lambda_up = lambda_val[~u_nonpositive_mask]
                new_A_U[idx_u_positive, idx_u_positive] = lambda_up * torch.ones(len(idx_u_positive))
                new_A_U[idx_u_positive, -1] = lambda_up * new_A_U[idx_u_positive, -1] + \
                                          (func(u_j_neq[idx_u_positive]) - lambda_up * u_j_neq[idx_u_positive])

        
        # print("BOUNDS", l_prime, u_prime)
        # print(new_A_L)
        # print(new_A_U)

        # Return the new DeepPoly domain
        return DeepPoly(l_prime, u_prime, self, new_A_L.double(), new_A_U.double())

    def __repr__(self):
        """
        Return a string representation of the bounds.
        """
        return f"DeepPolyDomain(lower={np.round(self.lower.numpy(), 4)}, upper={np.round(self.upper.numpy(), 4)})"

    def calculate_bounds(self, A_L = None, A_U = None):
        """
        Compute the concrete bounds for the current DeepPoly domain by recursively
        backsubstituting through the parent domains until we reach the input domain.

        Returns:
            (lower, upper): Two torch.Tensor vectors containing the concrete lower 
                            and upper bounds of the current domain's variables.
        """
        # Base case: If no parent, we are at the input domain with concrete bounds
        # print(A_L, A_U, self.parent)
        
        if A_L is None and self.parent is None:
            return self.lower, self.upper
        
        if self.parent is None:
            
            # LOWER BOUND
            
            
            pos_w = torch.clamp(A_L, min = 0.0).double()
            neg_w = torch.clamp(A_L, max = 0.0).double()
            lower_bound = torch.sum(pos_w @ self.A_L, axis = 1) + torch.sum(neg_w @ self.A_U, axis = 1)
            
            # UPPER BOUND
            pos_w = torch.clamp(A_U, min = 0).double()
            neg_w = torch.clamp(A_U, max = 0.0).double()
            upper_bound = torch.sum(pos_w @ self.A_U, axis = 1) + torch.sum(neg_w @ self.A_L, axis = 1)

            
            return lower_bound.reshape(-1, ), upper_bound.reshape(-1, )
        
        else:
            if A_L is None:
                
                lower_bound, upper_bound = self.parent.calculate_bounds(self.A_L[:,:-1].double(), self.A_U[:,:-1].double())
                
                return lower_bound + self.A_L[:,-1], upper_bound + self.A_U[:, -1]
            
            else:
                # LOWER BOUND
                # print(A_U)
                # print(self.A_U)
                pos_w = torch.clamp(A_L, min = 0.0).double()
                neg_w = torch.clamp(A_L, max = 0.0).double()
                new_A_L = pos_w @ self.A_L+ neg_w @ self.A_U 
                
                
                pos_w = torch.clamp(A_U, min = 0).double()
                neg_w = torch.clamp(A_U, max = 0.0).double()
                new_A_U = pos_w @ self.A_U + neg_w @ self.A_L
                
                lower_bound, upper_bound = self.parent.calculate_bounds(new_A_L[:,:-1].double(), new_A_U[:,:-1].double())
                
                return lower_bound + new_A_L[:, -1], upper_bound + new_A_U[:, -1]
            
        
    def to_hyperplanes(self):
        """
        Convert the box domain to a set of hyperplane inequalities.
        Each dimension contributes two hyperplanes.
        """
        inequalities = []
        self.lower, self.upper = self.calculate_bounds()
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
    
    def intersects(self, other):
        """
        Check if this box intersects with another box.
        """
        # self.lower, self.upper = self.calculate_bounds()
        return torch.all(self.lower < other.upper) and torch.all(self.upper > other.lower)

    def subtract(self, other):
        """
        Subtract another DeepPoly box from this box.
        Returns a list of resulting DeepPoly boxes after subtraction.
        """
        # self.lower, self.upper = self.calculate_bounds()
        if not self.intersects(other):
            return [self]  # No intersection, return the original box

        resulting_boxes = []
        for dim in range(len(self.lower)):
            if torch.round(other.lower[dim], decimals = 4) > torch.round(self.lower[dim], decimals = 4):
                # Create a box below the intersection along this dimension
                new_lower = self.lower.clone()
                new_upper = self.upper.clone()
                new_upper[dim] = other.lower[dim]
                if not torch.equal(new_upper, new_lower):
                    resulting_boxes.append(DeepPoly(new_lower.tolist(), new_upper.tolist()))

            if torch.round(other.upper[dim], decimals = 4) < torch.round(self.upper[dim], decimals = 4):
                # Create a box above the intersection along this dimension
                new_lower = self.lower.clone()
                new_upper = self.upper.clone()
                new_lower[dim] = other.upper[dim]
                if not torch.equal(new_upper, new_lower):
                    resulting_boxes.append(DeepPoly(new_lower.tolist(), new_upper.tolist()))

        return resulting_boxes 
    
    def __hash__(self):
        return hash(self.__repr__)

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

def intersect_domains(domains):
    """
    Compute the intersection of a list of DeepPoly domains.
    
    Args:
        domains (list of DeepPoly): A list of DeepPoly domains representing boxes.
        
    Returns:
        A list of DeepPoly domains that encompass the intersection. 
        For box domains, there will be either one resulting domain if the intersection is non-empty, 
        or an empty list if no intersection exists.
    """
    if not domains:
        return []
    
    # Start with the first domain as our intersection candidate
    intersection_lower = domains[0].lower.clone()
    intersection_upper = domains[0].upper.clone()
    
    # Iteratively intersect with the remaining domains
    for d in domains[1:]:
        intersection_lower = torch.max(intersection_lower, d.lower)
        intersection_upper = torch.min(intersection_upper, d.upper)
        
        # If at any point the intersection is empty, return an empty list
        if torch.any(intersection_lower > intersection_upper):
            return []
    
    # If the intersection is valid, return a single DeepPoly representing that intersection
    return DeepPoly(intersection_lower.tolist(), intersection_upper.tolist())

def can_merge(box_a, box_b):
    """
    Check if two DeepPoly boxes can be merged without overestimation.
    They can be merged if the union forms a single continuous box 
    in all dimensions (no gaps).
    """
    # Extract lower and upper bounds
    lA, uA = box_a.lower, box_a.upper
    lB, uB = box_b.lower, box_b.upper
    
    # Check for continuity in all dimensions
    # The union of intervals [lA_i, uA_i] and [lB_i, uB_i] must be contiguous:
    # max(lA_i, lB_i) <= min(uA_i, uB_i)
    # If this fails for any dimension, they cannot be merged without overestimation.
    if torch.any(torch.max(lA, lB) > torch.min(uA, uB)):
        return False
    
    return True

def merge_two_boxes(box_a, box_b):
    """
    Merge two DeepPoly boxes into one.
    Assumes can_merge(box_a, box_b) is True.
    """
    new_lower = torch.min(box_a.lower, box_b.lower)
    new_upper = torch.max(box_a.upper, box_b.upper)
    return DeepPoly(new_lower.tolist(), new_upper.tolist())

def merge_boxes(domains):
    """
    Given a list of DeepPoly domains (boxes), iteratively merge any that can be combined
    without overestimation until no more merges are possible.
    
    Returns:
        A list of DeepPoly domains, possibly reduced in number due to merging.
    """
    domains = list(domains)  # Copy to avoid modifying the original input
    merged = True
    
    while merged:
        merged = False
        new_domains = []
        
        # We'll use a simple O(n^2) approach: try to merge any pair
        # If a merge is successful, restart the process with the updated list.
        i = 0
        used = set()
        while i < len(domains):
            if i in used:
                i += 1
                continue
            
            merged_this_round = False
            for j in range(i+1, len(domains)):
                if j in used:
                    continue
                if can_merge(domains[i], domains[j]):
                    # Merge them
                    merged_box = intersect_domains([domains[i], domains[j]])
                    new_domains.append(merged_box)
                    used.add(i)
                    used.add(j)
                    merged_this_round = True
                    merged = True
                    break
            
            if not merged_this_round and i not in used:
                # No merge found for domains[i], keep it as is
                new_domains.append(domains[i])
            
            i += 1
        
        domains = new_domains
    
    return domains

# Example usage:
# Suppose we have several boxes that can be merged:
# Box 1: [0, 0] to [2, 2]
# Box 2: [2, 0] to [3, 2] (touching Box 1 on the right boundary)
# Box 3: [0, 2] to [2, 3] (touching Box 1 on the upper boundary)
# These three boxes can be merged into a single box [0,0] to [3,3].

# example_domains = [
#     DeepPoly([0,0],[2,2]),
#     DeepPoly([2,0],[3,2]),
#     DeepPoly([0,2],[2,3]),
# ]
# merged_result = merge_boxes(example_domains)
# for box in merged_result:
#     print(box)
