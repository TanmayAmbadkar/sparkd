import argparse
import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
import safety_gymnasium

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs import envs
from src.agent_factory import create_agent
from src.algorithms.sac.replay_memory import ReplayMemory

def train_offline(args):
    print(f"Starting Offline Training for {args.env_name}...")
    
    # 1. Setup Env (Just for dimensions)
    env = envs.get_env_from_name(args.env_name)
    
    # 2. Config Construction (Mimic main.py)
    # We need a proper DictConfig for the agent
    agent_params = {
        "name": "all_c",
        "gamma": 0.99,
        "lr": 3e-4,
        "batch_size": 256,
        "hidden_size": 256,
        "red_dim": 32, # Latent Dim
        "replay_size": 1000000,
        "seed": args.seed,
        "cuda": torch.cuda.is_available(),
        "horizon": 5,
        "action_dim": env.action_space.shape[0],
        "obs_shape": env.observation_space.shape,
        # Default All-C params
        "r_cost": 2.0,
        "value_hidden_dim": 256,
        "exploration_noise": 0.2,
        "gae_lambda": 0.95,
        "value_epochs": 10,
        "value_batch_size": 64,
        "vll_epochs_dyn": 20, # Used in init
        "vll_finetune_epochs_dyn": 20,
        "vll_finetune_epochs_cbf": 20,
        "on_policy_batch_size": 2048, # used for batching offline
        "num_steps": args.steps, # Anneal R over the training duration
        "start_steps": 0, # We have data immediately
        "tau": 0.005
    }
    
    conf = OmegaConf.create(agent_params)
    # Add 'agent' substructure pointing to same params for compatibility
    conf.agent = OmegaConf.create(agent_params)
    
    # 3. Create Agent
    agent = create_agent(conf, env)
    
    # 4. Load Data
    buffer_path = os.path.join(args.data_dir, f"offline_buffer_{args.env_name}.npz")
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"Data not found at {buffer_path}. Run collect_offline_data.py first.")
        
    agent.memory.load_buffer(buffer_path)
    
    # 5. Offline Train Loop
    agent.train_offline(steps=args.steps, log_every=100)
    
    # 6. Save Agent
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "all_c_offline_agent.pth")
    agent.save_checkpoint(save_path)
    print(f"Offline Agent saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="SafetyHalfCheetahVelocity-v1")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="runs_offline")
    args = parser.parse_args()
    
    train_offline(args)
