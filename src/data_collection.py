import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from src.envs import envs
from tqdm import tqdm

def compute_horizon_labels(
    states, next_states, costs, dones, horizon=10, gamma=0.95, alpha=2.0
):
    """
    Computes the Horizon Oracle target y_t^H.
    
    S_t = 1.0 - alpha * Cost_t  (Clipped [-1, 1])
    If done and cost > 0, S_t = -1.0
    
    y_t^H = min_{k=0..H} (gamma^k * S_{t+k})
    """
    num_samples = len(states)
    safety_scores = np.zeros(num_samples)
    horizon_targets = np.zeros(num_samples)
    
    # 1. Compute Immediate Safety Scores S_t
    for t in range(num_samples):
        c = costs[t]
        d = dones[t]
        
        # Base score
        s_val = 1.0 - alpha * c
        s_val = np.clip(s_val, -1.0, 1.0)
        
        # Hard failure override
        if d and c > 0:
            s_val = -1.0
            
        safety_scores[t] = s_val
        
    # 2. Compute Horizon Targets y_t^H (Backwards pass is usually dynamic programming, 
    # but here we need a window. We can just loop forward with a lookahead or backward.)
    # Since episodes are somewhat contiguous but might break, we need to respect 'dones'.
    # Actually, the buffer might be mixed episodes. We should process episode by episode.
    # But for simplicity in a flat buffer, let's assume we handle boundaries.
    
    # A robust way: Iterate through the buffer using indices.
    # We must be careful about episode boundaries. 
    # Current datasets are often just lists of transitions.
    
    # Let's simple iterate. If we hit a 'done' within the horizon, we stop looking ahead 
    # or assume the state stays terminal (if failure, -1 forever).
    
    for t in tqdm(range(num_samples), desc="Labeling Horizon Oracle"):
        min_score = 100.0 # Large init
        
        # Look ahead up to H steps
        for k in range(horizon + 1):
            idx = t + k
            
            # Check bounds
            if idx >= num_samples:
                break
                
            # Discounted score
            score = (gamma ** k) * safety_scores[idx]
            
            if score < min_score:
                min_score = score
            
            # If this step was terminal, we can't look past it for this trajectory
            if dones[idx]:
                break
                
        horizon_targets[t] = min_score
        
    return safety_scores, horizon_targets

def collect_data(env_name, num_steps, output_file, seed=42):
    env = envs.get_env_from_name(env_name)
    
    # Seeding
    # env.seed(seed) # Gym new API
    # env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Collecting {num_steps} steps from {env_name}...")
    
    states = []
    actions = []
    next_states = []
    rewards = []
    costs = []
    dones = []
    
    obs, _ = env.reset()
    
    for _ in tqdm(range(num_steps), desc="Collecting Data"):
        # Random Policy
        action = env.action_space.sample()
        
        next_obs, reward, cost, done, trunc, info = env.step(action)
        
        # Check for cost (env specific, usually info['cost'] or similar)
        # using cost from env.step() directly now

        
        states.append(obs)
        actions.append(action)
        next_states.append(next_obs)
        rewards.append(reward)
        costs.append(cost)
        dones.append(done or trunc)
        
        obs = next_obs
        if done or trunc:
            obs, _ = env.reset()
            
    # Convert to arrays
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    costs = np.array(costs)
    dones = np.array(dones)
    
    print("Labeling data...")
    safety_scores, horizon_targets = compute_horizon_labels(
        states, next_states, costs, dones, horizon=10, gamma=0.95
    )
    
    print(f"Saving to {output_file}...")
    data = {
        'states': torch.tensor(states, dtype=torch.float32),
        'actions': torch.tensor(actions, dtype=torch.float32),
        'next_states': torch.tensor(next_states, dtype=torch.float32),
        'costs': torch.tensor(costs, dtype=torch.float32),
        'dones': torch.tensor(dones, dtype=torch.float32),
        'safety_scores': torch.tensor(safety_scores, dtype=torch.float32),
        'horizon_targets': torch.tensor(horizon_targets, dtype=torch.float32)
    }
    
    torch.save(data, output_file)
    print("Done.")