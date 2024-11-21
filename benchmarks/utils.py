def normalize_constraints(target, a, b, target_range=(-1, 1)):
    """
    Normalize the constraints (lower and upper bounds) to a target range.

    Args:
        lower (torch.Tensor): Original lower bounds.
        upper (torch.Tensor): Original upper bounds.
        target_range (tuple): Target range (c, d) for normalization, e.g., (-1, 1).

    Returns:
        normalized_lower (torch.Tensor): Normalized lower bounds.
        normalized_upper (torch.Tensor): Normalized upper bounds.
    """
    c, d = target_range  # Target range

    # Scale bounds to the target range
    normalized_target = (target - a  + 0.0000000001) / (b - a  + 0.0000000001) * (d - c) + c
    # normalized_upper = (upper - a) / (b - a) * (d - c) + c

    return normalized_target

def denormalize_constraints(target, a, b, target_range=(-1, 1)):
    """
    Denormalize constraints from a target range back to the original range.

    Args:
        normalized_lower (torch.Tensor): Normalized lower bounds.
        normalized_upper (torch.Tensor): Normalized upper bounds.
        original_lower (torch.Tensor): Original lower bounds.
        original_upper (torch.Tensor): Original upper bounds.
        target_range (tuple): The target range used for normalization.

    Returns:
        denormalized_lower (torch.Tensor): Denormalized lower bounds.
        denormalized_upper (torch.Tensor): Denormalized upper bounds.
    """
    c, d = target_range

    denormalized_target = (target - c) / (d - c) * (b - a) + a

    return denormalized_target
