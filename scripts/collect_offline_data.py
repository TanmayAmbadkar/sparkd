import argparse
import os
import sys
import random
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
import gymnasium as gym
import safety_gymnasium

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs import envs
from src.agent_factory import create_agent
from src.algorithms.sac.replay_memory import ReplayMemory

def collect_data(args):
    """
    Collects data using a Random Policy (or loaded agent) and saves to buffer.
    """
    print(f"Collecting Offline Data for env: {args.env_name}")
    
    # 1. Setup Env
    # env = envs.get_env_from_name(args.env_name)
    # Direct creation to avoid PassiveEnvChecker issues with 6-element tuple
    env = safety_gymnasium.make(args.env_name)
    
    # 2. Setup Memory
    action_dim = env.action_space.shape[0]
    buffer = ReplayMemory(args.buffer_size, env.observation_space, action_dim, args.seed)
    
    # 3. Collection Loop
    state, _ = env.reset(seed=args.seed)
    steps = 0
    episodes = 0
    
    while steps < args.num_steps:
        # Random Action for exploration / diversity
        # Ideally we want a mixture of random and partially trained policies for D4RL-style
        # For now, pure random is a good "Medium-Replay" equivalent baseline
        action = env.action_space.sample()
        
        next_state, reward, cost, done, trunc, _ = env.step(action)
        
        buffer.push(state, action, reward, next_state, done or trunc, cost)
        
        state = next_state
        steps += 1
        
        if done or trunc:
            state, _ = env.reset()
            episodes += 1
            if episodes % 10 == 0:
                print(f"Collected {steps}/{args.num_steps} steps...")

    print(f"Collection Complete. Buffer Size: {len(buffer)}")
    
    # 4. Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"offline_buffer_{args.env_name}.npz")
    buffer.save_buffer(save_path)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="SafetyHalfCheetahVelocity-v1")
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_dir", type=str, default="data")
    args = parser.parse_args()
    
    collect_data(args)
