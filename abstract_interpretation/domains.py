import numpy as np
import torch
from scipy.optimize import linprog
import torch
import numpy as np
from itertools import product


class DeepPoly:
    def __init__(self, lower_bounds, upper_bounds, parent=None, A_L=None, A_U=None):
        """
        Initialize the DeepPoly domain with lower and upper bounds.
        The bounds are expected to be either 1D (a single domain) or 2D (batch_size x feature_dim).
        """
        # Convert inputs to torch tensors with type float64.
        if not torch.is_tensor(lower_bounds):
            lower_bounds = torch.tensor(lower_bounds, dtype=torch.float64)
        else:
            lower_bounds = lower_bounds.double()
        if not torch.is_tensor(upper_bounds):
            upper_bounds = torch.tensor(upper_bounds, dtype=torch.float64)
        else:
            upper_bounds = upper_bounds.double()

        # Ensure there is a batch dimension.
        if lower_bounds.dim() == 1:
            lower_bounds = lower_bounds.unsqueeze(0)
            upper_bounds = upper_bounds.unsqueeze(0)

        self.lower = lower_bounds  # Shape: (B, n)
        self.upper = upper_bounds  # Shape: (B, n)
        self.parent = parent
        self.name = None

        if self.lower.shape != self.upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")

        if lower_bounds.dim() != 1:
           batch_size, input_size = self.lower.shape

        if A_L is None:
            # Create initial affine coefficient matrices.
            # They have shape (B, input_size, input_size+1) where the last column is the constant.
            self.A_L = torch.zeros((batch_size, input_size, input_size + 1), dtype=torch.float64)
            self.A_U = torch.zeros((batch_size, input_size, input_size + 1), dtype=torch.float64)
            self.A_L[:, :, -1] = self.lower
            self.A_U[:, :, -1] = self.upper
        else:
            self.A_L = A_L
            self.A_U = A_U

    def affine_transform(self, W, b):
        """
        Perform an affine transformation on the batched DeepPoly domain using CROWN linearization.
        Args:
            W (torch.Tensor): Weight matrix of shape (output_dim, input_dim).
            b (torch.Tensor): Bias vector of shape (output_dim,).
        Returns:
            DeepPoly: New DeepPoly domain with updated bounds.
        """
        batch_size, input_dim = self.lower.shape
        output_dim = W.shape[0]

        # Build the new affine coefficients for the layer.
        base_A = torch.cat([W, b.unsqueeze(1)], dim=1)  # (output_dim, input_dim+1)
        new_A_L = base_A.unsqueeze(0).expand(batch_size, -1, -1).clone().double()
        new_A_U = base_A.unsqueeze(0).expand(batch_size, -1, -1).clone().double()

        # Compute new bounds with CROWN linearization
        pos_w = (W >= 0.0).double()  # (output_dim, input_dim)
        neg_w = (W < 0.0).double()   # (output_dim, input_dim)

        # Apply CROWN linearization (linearize activation at bounds)
        ub = torch.matmul(self.upper, (pos_w.T * W.T)) + torch.matmul(self.lower, (neg_w.T * W.T)) + b
        lb = torch.matmul(self.lower, (pos_w.T * W.T)) + torch.matmul(self.upper, (neg_w.T * W.T)) + b

        self.name = "CROWN_AFFINE"
        return DeepPoly(lb, ub, parent=self, A_L=new_A_L, A_U=new_A_U)


    def relu(self):
        """
        Apply ReLU activation with CROWN linearization.
        Use piecewise linear approximation around the input bounds.
        """
        self.name = "CROWN_RELU"
        batch_size, n = self.lower.shape

        new_lower = self.lower.clone().detach()
        new_upper = self.upper.clone().detach()
        new_A_L = torch.zeros((batch_size, n, n+1), dtype=torch.float64)
        new_A_U = torch.zeros((batch_size, n, n+1), dtype=torch.float64)

        # Compute masks for the three cases.
        case1 = self.upper <= 0      # Completely inactive.
        case2 = self.lower >= 0      # Completely active.
        case3 = (~case1) & (~case2)   # Mixed.

        # Case 1: Neurons that are always inactive become 0.
        idx_case1 = case1.nonzero(as_tuple=True)
        if idx_case1[0].numel() > 0:
            new_lower[idx_case1] = 0
            new_upper[idx_case1] = 0

        # Case 2: Neurons that are always active pass through exactly.
        idx_case2 = case2.nonzero(as_tuple=True)
        if idx_case2[0].numel() > 0:
            new_A_L[idx_case2[0], idx_case2[1], idx_case2[1]] = 1.0
            new_A_U[idx_case2[0], idx_case2[1], idx_case2[1]] = 1.0

        # Case 3: Mixed neurons using piecewise linear relaxation.
        idx_case3 = case3.nonzero(as_tuple=True)
        if idx_case3[0].numel() > 0:
            batch_idxs, feat_idxs = idx_case3
            l_vals = self.lower[batch_idxs, feat_idxs]
            u_vals = self.upper[batch_idxs, feat_idxs]
            
            # CROWN relaxation: piecewise linear approximation
            lambda_u = u_vals / (u_vals - l_vals)
            new_A_U[batch_idxs, feat_idxs, feat_idxs] = lambda_u
            new_A_U[batch_idxs, feat_idxs, -1] = lambda_u * (-l_vals)
            pos_coeffs = torch.clamp(new_A_U[batch_idxs, feat_idxs, feat_idxs], min=0)
            neg_coeffs = torch.clamp(new_A_U[batch_idxs, feat_idxs, feat_idxs], max=0)
            new_upper[batch_idxs, feat_idxs] = new_A_U[batch_idxs, feat_idxs, -1] + u_vals * pos_coeffs + l_vals * neg_coeffs

            # Lower bound relaxation
            lambda_choice = torch.where(u_vals <= -l_vals, torch.zeros_like(u_vals), torch.ones_like(u_vals))
            new_A_L[batch_idxs, feat_idxs, feat_idxs] = lambda_choice
            new_lower[batch_idxs, feat_idxs] = lambda_choice * l_vals

        return DeepPoly(new_lower, new_upper, parent=self, A_L=new_A_L, A_U=new_A_U)


    def sigmoid(self):
        """
        Apply the Sigmoid activation function using the abstract transformer.
        """
        self.name = "SIGMOID"
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
        General method for applying a non-linear activation function in a batched fashion.
        """
        batch_size, n = self.lower.shape
        new_A_L = torch.zeros((batch_size, n, n+1), dtype=torch.float64)
        new_A_U = torch.zeros((batch_size, n, n+1), dtype=torch.float64)

        l_j = self.lower.clone()
        u_j = self.upper.clone()
        l_prime = func(l_j).double()
        u_prime = func(u_j).double()

        # For coordinates where the bounds are equal.
        equal_mask = (l_j == u_j)
        idx_equal = equal_mask.nonzero(as_tuple=True)
        if idx_equal[0].numel() > 0:
            new_A_L[idx_equal[0], idx_equal[1], :-1] = 0
            new_A_L[idx_equal[0], idx_equal[1], -1] = l_prime[idx_equal[0], idx_equal[1]]
            new_A_U[idx_equal[0], idx_equal[1], :-1] = 0
            new_A_U[idx_equal[0], idx_equal[1], -1] = u_prime[idx_equal[0], idx_equal[1]]

        # For coordinates where l_j != u_j.
        idx_neq = (~equal_mask).nonzero(as_tuple=True)
        if idx_neq[0].numel() > 0:
            batch_idxs, feat_idxs = idx_neq
            l_j_neq = l_j[batch_idxs, feat_idxs]
            u_j_neq = u_j[batch_idxs, feat_idxs]
            denominator = u_j_neq - l_j_neq
            denominator = torch.where(denominator == 0, torch.full_like(denominator, 1e-6), denominator)
            lambda_val = (func(u_j_neq) - func(l_j_neq)) / denominator
            lambda_prime = torch.min(func_prime(l_j_neq), func_prime(u_j_neq)).double()

            # Update lower affine expressions.
            l_positive_mask = l_j_neq > 0
            if l_positive_mask.sum() > 0:
                idx_l_positive = (batch_idxs[l_positive_mask], feat_idxs[l_positive_mask])
                lambda_lp = lambda_val[l_positive_mask]
                new_A_L[idx_l_positive[0], idx_l_positive[1], idx_l_positive[1]] = lambda_lp
                new_A_L[idx_l_positive[0], idx_l_positive[1], -1] = func(l_j_neq[l_positive_mask]) - lambda_lp * l_j_neq[l_positive_mask]
            if (~l_positive_mask).sum() > 0:
                idx_l_nonpositive = (batch_idxs[~l_positive_mask], feat_idxs[~l_positive_mask])
                lambda_lnp = lambda_prime[~l_positive_mask]
                new_A_L[idx_l_nonpositive[0], idx_l_nonpositive[1], idx_l_nonpositive[1]] = lambda_lnp
                new_A_L[idx_l_nonpositive[0], idx_l_nonpositive[1], -1] = func(l_j_neq[~l_positive_mask]) - lambda_lnp * l_j_neq[~l_positive_mask]

            # Update upper affine expressions.
            u_nonpositive_mask = u_j_neq <= 0
            if u_nonpositive_mask.sum() > 0:
                idx_u_nonpositive = (batch_idxs[u_nonpositive_mask], feat_idxs[u_nonpositive_mask])
                lambda_unp = lambda_prime[u_nonpositive_mask]
                new_A_U[idx_u_nonpositive[0], idx_u_nonpositive[1], idx_u_nonpositive[1]] = lambda_unp
                new_A_U[idx_u_nonpositive[0], idx_u_nonpositive[1], -1] = func(u_j_neq[u_nonpositive_mask]) - lambda_unp * u_j_neq[u_nonpositive_mask]
            if (~u_nonpositive_mask).sum() > 0:
                idx_u_positive = (batch_idxs[~u_nonpositive_mask], feat_idxs[~u_nonpositive_mask])
                lambda_up = lambda_val[~u_nonpositive_mask]
                new_A_U[idx_u_positive[0], idx_u_positive[1], idx_u_positive[1]] = lambda_up
                new_A_U[idx_u_positive[0], idx_u_positive[1], -1] = func(u_j_neq[~u_nonpositive_mask]) - lambda_up * u_j_neq[~u_nonpositive_mask]

        return DeepPoly(l_prime, u_prime, parent=self, A_L=new_A_L, A_U=new_A_U)

    
    def calculate_bounds(self):
        """
        Recursively compute the concrete bounds for the current DeepPoly domain using CROWN.
        """
        if self.parent is None:
            return self.lower, self.upper
        else:
            # Get the parent's bounds
            parent_lower, parent_upper = self.parent.calculate_bounds()
            
            # Get the current layer's affine coefficients
            weight_L = self.A_L[..., :-1]  # (B, current_n, parent_n)
            bias_L   = self.A_L[..., -1]   # (B, current_n)
            weight_U = self.A_U[..., :-1]  # (B, current_n, parent_n)
            bias_U   = self.A_U[..., -1]   # (B, current_n)
            
            # Use CROWN-based linearization for both lower and upper bounds
            pos_weight_L = torch.clamp(weight_L, min=0.0)
            neg_weight_L = torch.clamp(weight_L, max=0.0)
            new_lower = (torch.bmm(pos_weight_L, parent_lower.unsqueeze(-1)).squeeze(-1) +
                        torch.bmm(neg_weight_L, parent_upper.unsqueeze(-1)).squeeze(-1) +
                        bias_L)
            
            pos_weight_U = torch.clamp(weight_U, min=0.0)
            neg_weight_U = torch.clamp(weight_U, max=0.0)
            new_upper = (torch.bmm(pos_weight_U, parent_upper.unsqueeze(-1)).squeeze(-1) +
                        torch.bmm(neg_weight_U, parent_lower.unsqueeze(-1)).squeeze(-1) +
                        bias_U)
            
            return new_lower, new_upper


    def __repr__(self):
        """
        Return a string representation that shows the batch shape and the bounds of the first sample.
        """
        lower_np = np.round(self.lower.numpy(), 4)
        upper_np = np.round(self.upper.numpy(), 4)
        return (f"DeepPolyDomain(batch_shape={self.lower.shape}, "
                f"first_sample_lower={lower_np[0]}, first_sample_upper={upper_np[0]})")



    
    def batch_split_and_join_bounds_all_dims(self, propagate_fn, parts_per_dim=1, batch_size=1000):
        """
        Perform trace partitioning over all dimensions with batching to avoid the exponential
        blowup of analyzing every subdomain individually. This version accumulates subdomains,
        propagates them in batches through the network (using propagate_fn), and joins the bounds.
        GPU is used if available.

        Args:
            propagate_fn (callable): Function taking a DeepPoly element and returning the propagated DeepPoly.
            parts_per_dim (int): Number of subintervals per input dimension.
            batch_size (int): Number of subdomains to process in each batch.

        Returns:
            (joined_lower, joined_upper): Tensors representing the joined (refined) lower and upper bounds.
        """
        # Use GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        # Assume self.lower is of shape (1, n) for a single domain; remove batch dimension.
        lower_np = self.lower.cpu().numpy().squeeze(0)
        upper_np = self.upper.cpu().numpy().squeeze(0)
        dims = lower_np.shape[0]

        # Create partitions for each input dimension.
        partitions = []
        for d in range(dims):
            lb = lower_np[d]
            ub = upper_np[d]
            step = (ub - lb) / parts_per_dim
            partitions.append([(lb + i * step, lb + (i + 1) * step) for i in range(parts_per_dim)])

        # Generator for subdomains: each yields (sub_lb, sub_ub) as numpy arrays of shape (n,).
        def gen_subdomains():
            for sub in product(*partitions):
                sub_lb = np.array([interval[0] for interval in sub])
                sub_ub = np.array([interval[1] for interval in sub])
                yield sub_lb, sub_ub

        batch_results = []   # Will store a DeepPoly for each processed batch.
        batch_lower_list = []  # Temporary storage for lower bounds of current batch.
        batch_upper_list = []  # Temporary storage for upper bounds of current batch.
        count = 0
        total_subdomains = parts_per_dim ** dims

        # Iterate over all subdomains.
        for sub_lb, sub_ub in gen_subdomains():
            count += 1
            batch_lower_list.append(sub_lb)
            batch_upper_list.append(sub_ub)
            if len(batch_lower_list) == batch_size:
                # Build batched tensors on the correct device.
                batched_lower = torch.tensor(np.array(batch_lower_list), dtype=torch.float64, device=device)
                batched_upper = torch.tensor(np.array(batch_upper_list), dtype=torch.float64, device=device)
                # Create a batched DeepPoly domain.
                dp_batch = DeepPoly(batched_lower, batched_upper)
                # Propagate the entire batch through the network.
                dp_batch = propagate_fn(dp_batch)
                # Compute the concrete bounds for each subdomain in the batch.
                lower_batch, upper_batch = dp_batch.calculate_bounds()
                # Join over the batch dimension.
                joined_lower_batch, _ = torch.min(lower_batch, dim=0)
                joined_upper_batch, _ = torch.max(upper_batch, dim=0)
                # Store the joined batch result as a DeepPoly (with a singleton batch dimension).
                batch_results.append(DeepPoly(joined_lower_batch.unsqueeze(0), joined_upper_batch.unsqueeze(0)))
                batch_lower_list = []
                batch_upper_list = []
                print("\rProcessed {} / {} subdomains".format(count, total_subdomains), end="")
        print()

        # Process any remaining subdomains.
        if batch_lower_list:
            batched_lower = torch.tensor(batch_lower_list, dtype=torch.float64, device=device)
            batched_upper = torch.tensor(batch_upper_list, dtype=torch.float64, device=device)
            dp_batch = DeepPoly(batched_lower, batched_upper)
            dp_batch = propagate_fn(dp_batch)
            lower_batch, upper_batch = dp_batch.calculate_bounds()
            joined_lower_batch, _ = torch.min(lower_batch, dim=0)
            joined_upper_batch, _ = torch.max(upper_batch, dim=0)
            batch_results.append(DeepPoly(joined_lower_batch.unsqueeze(0), joined_upper_batch.unsqueeze(0)))
            print("\rProcessed {} / {} subdomains".format(count, total_subdomains), end="")
        print()
        # Join the bounds from all batch results.
        all_lower_batches = torch.stack([b.lower.squeeze(0) for b in batch_results])
        all_upper_batches = torch.stack([b.upper.squeeze(0) for b in batch_results])
        joined_lower, _ = torch.min(all_lower_batches, dim=0)
        joined_upper, _ = torch.max(all_upper_batches, dim=0)

        return joined_lower, joined_upper

    def to_hyperplanes(self):
        """
        Convert the DeepPoly domain into a set of hyperplane inequalities.
        Each dimension yields two hyperplanes.
        
        If the domain is batched (shape (B, n)), the function returns a list of length B,
        where each entry is a list of 2*n hyperplane inequalities (each a NumPy array).
        """
        lower, upper = self.calculate_bounds()
        if lower.ndim == 1:
            dims = lower.shape[0]
            inequalities = []
            for i in range(dims):
                A_upper = np.zeros(dims)
                A_upper[i] = 1
                inequalities.append(np.append(A_upper, -upper[i].item()))
                A_lower = np.zeros(dims)
                A_lower[i] = -1
                inequalities.append(np.append(A_lower, lower[i].item()))
            return inequalities
        else:
            B, dims = lower.shape
            all_inequalities = []
            for b in range(B):
                inequalities = []
                for i in range(dims):
                    A_upper = np.zeros(dims)
                    A_upper[i] = 1
                    inequalities.append(np.append(A_upper, -upper[b, i].item()))
                    A_lower = np.zeros(dims)
                    A_lower[i] = -1
                    inequalities.append(np.append(A_lower, lower[b, i].item()))
                all_inequalities.append(np.array(inequalities))
            return all_inequalities
        
    def invert_polytope(self):

        """
        Invert the DeepPoly domain to obtain the set of hyperplanes.
        This is done by negating the lower and upper bounds.
        """
        lower, upper = self.calculate_bounds()
        if lower.ndim == 1:
            dims = lower.shape[0]
            inequalities = []
            for i in range(dims):
                A_upper = np.zeros(dims)
                A_upper[i] = -1
                inequalities.append(np.append(A_upper, upper[i].item()))
                A_lower = np.zeros(dims)
                A_lower[i] = 1
                inequalities.append(np.append(A_lower, -lower[i].item()))
            return inequalities
        else:
            B, dims = lower.shape
            all_inequalities = []
            for b in range(B):
                inequalities = []
                for i in range(dims):
                    A_upper = np.zeros(dims)
                    A_upper[i] = -1
                    all_inequalities.append(np.array([np.append(A_upper, upper[b, i].item())]))
                    A_lower = np.zeros(dims)
                    A_lower[i] = 1
                    all_inequalities.append(np.array([np.append(A_lower, -lower[b, i].item())]))
                # all_inequalities.append(np.array([inequalities]))3
            return all_inequalities



    def intersects(self, other):
        """
        Check whether this DeepPoly domain (or each element in a batched domain) intersects with another.
        If both domains are batched, they are compared elementwise (and a boolean tensor is returned).
        If one of them is a single box, it is unsqueezed to have a batch dimension.
        """
        if self.lower.ndim == 1:
            return torch.all(self.lower < other.upper) and torch.all(self.upper > other.lower)
        else:
            # Ensure both operands have a batch dimension.
            if other.lower.ndim == 1:
                other_lower = other.lower.unsqueeze(0)
                other_upper = other.upper.unsqueeze(0)
            else:
                other_lower = other.lower
                other_upper = other.upper
            # Intersection holds if for every feature, self.lower < other.upper and self.upper > other.lower.
            return torch.all(self.lower < other_upper, dim=1) & torch.all(self.upper > other_lower, dim=1)

    def subtract(self, other):
        """
        Subtract another DeepPoly box (other) from this one.
        Returns a list of DeepPoly boxes representing the result.
        If self is batched, the subtraction is applied elementwise and all resulting boxes are collected.
        """
        # If batched (batch_size > 1), process each element independently.
        if self.lower.ndim > 1 and self.lower.shape[0] > 1:
            results = []
            B = self.lower.shape[0]
            # If 'other' is a single box, broadcast it.
            if other.lower.ndim == 1:
                other_dp = DeepPoly(other.lower.unsqueeze(0), other.upper.unsqueeze(0))
            else:
                other_dp = other
            for b in range(B):
                dp_single = DeepPoly(self.lower[b].unsqueeze(0), self.upper[b].unsqueeze(0))
                # If other is batched with the same size, subtract elementwise.
                if other_dp.lower.shape[0] == 1:
                    sub_res = dp_single.subtract(other_dp)
                else:
                    sub_res = dp_single.subtract(DeepPoly(other_dp.lower[b].unsqueeze(0), other_dp.upper[b].unsqueeze(0)))
                results.extend(sub_res)
            return results
        else:
            # self is a single box (1D lower/upper).
            if not self.intersects(other):
                return [self]
            resulting_boxes = []
            dims = self.lower.shape[1]  # since self.lower is of shape (1, n)
            # Work with 1D tensors.
            l = self.lower.squeeze(0)
            u = self.upper.squeeze(0)
            other_l = other.lower.squeeze(0) if other.lower.ndim > 1 else other.lower
            other_u = other.upper.squeeze(0) if other.upper.ndim > 1 else other.upper
            for dim in range(dims):
                if torch.round(other_l[dim], decimals=4) > torch.round(l[dim], decimals=4):
                    new_lower = l.clone()
                    new_upper = u.clone()
                    new_upper[dim] = other_l[dim]
                    if not torch.equal(new_upper, new_lower):
                        resulting_boxes.append(DeepPoly(new_lower.tolist(), new_upper.tolist()))
                if torch.round(other_u[dim], decimals=4) < torch.round(u[dim], decimals=4):
                    new_lower = l.clone()
                    new_upper = u.clone()
                    new_lower[dim] = other_u[dim]
                    if not torch.equal(new_upper, new_lower):
                        resulting_boxes.append(DeepPoly(new_lower.tolist(), new_upper.tolist()))
            return resulting_boxes

    def sample(self, size=1):
        """
        Uniformly sample points from the DeepPoly domain.
        
        If the domain is batched (shape (B, n)), returns a tensor of shape:
            - (B, size, n) if size > 1, or
            - (B, n) if size == 1.
        If the domain is a single box (shape (n,)), returns a tensor of shape (size, n).
        """
        if self.lower.ndim == 1:
            n = self.lower.shape[0]
            samples = torch.rand((size, n), dtype=self.lower.dtype, device=self.lower.device) * (self.upper - self.lower) + self.lower
            return samples
        else:
            B, n = self.lower.shape
            samples = torch.rand((B, size, n), dtype=self.lower.dtype, device=self.lower.device) * (self.upper.unsqueeze(1) - self.lower.unsqueeze(1)) + self.lower.unsqueeze(1)
            if size == 1:
                samples = samples.squeeze(1)
            return samples

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        """
        Return a string representation showing the batch shape and the bounds for the first sample.
        """
        lower_np = np.round(self.lower.cpu().numpy(), 4)
        upper_np = np.round(self.upper.cpu().numpy(), 4)
        return (f"DeepPolyDomain(batch_shape={self.lower.shape}, "
                f"first_sample_lower={lower_np[0]}, first_sample_upper={upper_np[0]})")

    

def recover_safe_region(observation_box, unsafe_boxes):
    """
    Recover the safe region by subtracting unsafe boxes from the observation boundary.
    
    Args:
        observation_box: A DeepPoly object representing the observation boundary.
        unsafe_boxes: List of DeepPoly objects representing the unsafe region.
    
    Returns:
        A list of DeepPoly objects representing the safe region.
    """
    # Start with the observation boundary.
    safe_regions = [observation_box]
    # Iteratively subtract each unsafe box.
    for unsafe_box in unsafe_boxes:
        new_safe_regions = []
        for safe_box in safe_regions:
            new_safe_regions.extend(safe_box.subtract(unsafe_box))
        safe_regions = new_safe_regions


    lower = []
    upper = []

    for i in range(len(safe_regions)):
        lower.append(safe_regions[i].lower)
        upper.append(safe_regions[i].upper)

    lower = torch.vstack(lower)
    upper = torch.vstack(upper)
    safe_regions = DeepPoly(lower, upper)
    return safe_regions