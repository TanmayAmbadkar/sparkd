import numpy as np
import json
import io

def parse_dynamics_from_json(filepath):
    """
    Parses a JSON file to extract the linear dynamics matrices A, B, c, and the error bound vector eps.

    The file is expected to be a JSON object with keys "A", "B", "c", and "eps".

    Args:
        filepath (str or file-like object): The path to the dynamics JSON file or a file-like object.

    Returns:
        tuple: A tuple containing (A, B, c, eps) where A, B, c, and eps are NumPy arrays.
    """
    # Check if filepath is a path or an already opened file
    is_path = isinstance(filepath, str)
    f = open(filepath, 'r') if is_path else filepath

    try:
        data = json.load(f)
    finally:
        if is_path:
            f.close()

    # --- Convert to NumPy arrays ---
    A = np.array(data['A'], dtype=np.float32)
    B = np.array(data['B'], dtype=np.float32)
    c = np.array(data['c'], dtype=np.float32)
    eps = np.array(data['eps'], dtype=np.float32)

    # Ensure B, c, and eps are column vectors if they are not already
    if len(B.shape) == 1:
        B = B.reshape(-1, 1)
    if len(c.shape) == 1:
        c = c.reshape(-1, 1)
    if len(eps.shape) == 1:
        eps = eps.reshape(-1, 1)

    return A, B, c, eps
