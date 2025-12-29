import numpy as np
import gymnasium as gym
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.getcwd())

from src.policy import ProjectionPolicy

# Mock Environment Model
class MockKoopmanModel:
    def __init__(self, s_dim, u_dim):
        self.adaptive_error = False
    
    def get_matrix_at_point(self, point, s_dim):
        # Identity dynamics: x' = x + u (simple integrator)
        A = np.eye(s_dim)
        B = np.eye(s_dim) 
        c = np.zeros(s_dim)
        M = np.hstack((A, B, c[:, None]))
        return M, (None, None)

def create_poly(low, high):
    """Creates a polytope Px <= b for a box [low, high]"""
    dim = len(low)
    # [I; -I] x <= [high; -low]
    P = np.vstack([np.eye(dim), -np.eye(dim)])
    b = np.concatenate([high, -np.array(low)])
    return np.hstack([P, b[:, None]])

def test_runaway_backup():
    print("\n--- Test 1: Run Away Backup Logic ---")
    s_dim = 2
    u_dim = 2
    env = MockKoopmanModel(s_dim, u_dim)
    
    # Safe Poly (Ignored by new backup, but needed for init)
    safe_polys = [create_poly([0, 0], [10, 10])]
    
    # Unsafe Poly: Box [4, 5]
    unsafe_polys = [create_poly([4, 4], [5, 5])]
    
    state_space = gym.spaces.Box(low=0, high=10, shape=(s_dim,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(u_dim,))
    
    policy = ProjectionPolicy(env, state_space, action_space, horizon=5,
                              unsafe_polys=unsafe_polys, safe_polys=safe_polys)
    policy.update_model()
    
    # State effectively INSIDE unsafe region (or very close)
    # Center of [4,5] is 4.5. Let's be at 4.5.
    state = np.array([4.5, 4.5])
    
    # The geometric center of unsafe is 4.5.
    # The direction "away" is ambiguous if exactly at center, so let's be slightly off center
    # State = 4.8. Nearest boundary is 5. Direction away is NEGATIVE (towards 0).
    # Wait, if we are at 4.8, the interval is [4, 5].
    # Distance to 5 is 0.2. Distance to 4 is 0.8.
    # "Escape direction" from QP stage 1 depends on finding shortest vector TO unsafe.
    # If we are INSIDE, the distance is 0. 
    # The implementation calculates vector d such that z+d is inside? No, existing logic was:
    # "find shortest vector... to inside". If inside, it's 0.
    
    # Let's check logic:
    # backup_qp_stage1: A * (z+d) <= b. Minimize ||d||.
    # If z is inside, d=0 is valid.
    # Then best_proj = 0.
    # Then escape_dir = 0.
    # And it returns zero action.
    
    # So "Run Away" only triggers if we are slightly outside? 
    # Or does it need to project to *boundary*?
    # The user logic was "shortest vector from state to unsafe".
    # If we are OUTSIDE, say at 3.0. Unsafe starts at 4.0.
    # Shortest vector is +1.0 (pointing to 4.0).
    # Escape direction is -1.0 (Run away from 4.0).
    
    state = np.array([3.9, 3.9]) # Near 4.0
    u_backup = policy.backup(state)
    print(f"State: {state}, Backup Action: {u_backup}")
    
    # We expect action to be NEGATIVE (running away from 4.0, towards 0.0)
    if np.any(u_backup < -0.1):
        print("SUCCESS: Action is negative (running away).")
    else:
        print("FAILURE: Action is not running away effectively.")

def test_caching_bug():
    print("\n--- Test 2: Caching Bug Fix ---")
    s_dim = 2
    u_dim = 2
    env = MockKoopmanModel(s_dim, u_dim)
    safe_polys = [create_poly([0, 0], [10, 10])]
    unsafe_polys = []
    
    state_space = gym.spaces.Box(low=0, high=10, shape=(s_dim,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(u_dim,))
    
    policy = ProjectionPolicy(env, state_space, action_space, horizon=5,
                              unsafe_polys=unsafe_polys, safe_polys=safe_polys)
    policy.update_model()
    
    state = np.array([5.0, 5.0])
    action_risky = np.array([0.9, 0.9])
    
    # 1. Call solve/unsafe with action_risky
    print("Calling with risky action...")
    _, _ = policy.solve(state, action=action_risky)
    
    # 2. Call policy(state) which implies action=0 (or whatever default)
    print("Calling with default action (None)...")
    res_default, _ = policy(state, action=None)
    
    print(f"Default Result: {res_default}")
    
    # If it returns approx action_risky, the cache is still broken.
    if np.allclose(res_default, action_risky):
        print("FAILURE: Caching bug persists.")
    else:
        print("SUCCESS: Cache respects input action.")

if __name__ == "__main__":
    test_runaway_backup()
    test_caching_bug()
