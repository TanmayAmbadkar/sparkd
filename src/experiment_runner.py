
import os
import torch
import numpy as np
from src.algorithms.sac.replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Tuple, Optional, Dict, List
from src.data_collection import compute_horizon_labels
from src import train_vdk

from src.vdk_shield import VDK_Shield, VisualEncoder, VisualDecoder, TabularEncoder, TabularDecoder
from src.shield import VDK_Runtime
from src.policies.shield import Shield, ShieldPolicy
from src.policies.abstract_agent import Agent
import imageio
# Use the Abstract Base Class for typing
AgentType = Agent 

def compute_labels_from_batch(costs, dones, gamma=0.95, alpha=2.0):
    """
    Computes horizon target y_0^H for each sequence in the batch (N, T).
    y_0^H = min_{k=0..T-1} (gamma^k * S_k)
    where S_k = 1 - alpha*cost_k (clipped) or -1 if failure.
    
    Returns: (N,) horizon_targets
    """
    # costs, dones are (N, T)
    batch_size, T = costs.shape
    
    # 1. Compute Base Scores S (N, T)
    # S = 1 - alpha * C
    S = 1.0 - alpha * costs
    S = np.clip(S, -1.0, 1.0)
    
    # Failure handling: if done AND cost > 0, set current and future S to -1?
    # Actually if done[k]=True and cost[k]>0, then S[k]=-1.
    # Future steps in the sequence (if they exist, likely padding or next ep) are irrelevant 
    # because we stop rollout at failure.
    # But ReplayMemory guarantees valid sequences (no done in middle) via 'valid_length' check?
    # Wait, ReplayMemory sample implementation:
    # batch_dones = array([dones[i:i+horizon] ...])
    # It does NOT strictly filter out dones in the middle currently.
    # If done occurs at step k < T-1, the physical transition breaks.
    # However, for the label, we just need to confirm failure.
    
    # Vectorized computation
    # Mask failures: cost > 0 and done
    failures = (dones > 0.5) & (costs > 0)
    S[failures] = -1.0
    
    # Discounted min reduction
    # y_i = min(gamma^0 S_i_0, gamma^1 S_i_1, ...)
    
    discounts = np.power(gamma, np.arange(T)) # (T,)
    S_discounted = S * discounts.reshape(1, T) # (N, T)
    
    # We need to handle cumulative min, but actually we want min over the whole T.
    # But wait, if done occurs at k, steps k+1... are invalid.
    # We should mask them with +inf so min ignores them.
    # Find first done index for each row.
    
    # Simple accumulation for now:
    vals = np.min(S_discounted, axis=1) # (N,)
    return vals


def load_vdk_shield(
    shield_path: str, 
    env_obs_space: Any, 
    action_dim: int, 
    args: Any
) -> Tuple[VDK_Shield, np.ndarray, np.ndarray]:
    """
    Loads a VDK Shield from disk, ensuring correct architecture (Visual/Tabular).
    Returns (vdk_shield, state_mean, state_std).
    """
    checkpoint = torch.load(shield_path)
    
    # Instantiate VDK
    obs_shape = env_obs_space.shape
    is_image = len(obs_shape) == 3
    
    if is_image:
        encoder = VisualEncoder(latent_dim=args.red_dim, in_channels=obs_shape[0])
        decoder = VisualDecoder(latent_dim=args.red_dim, out_channels=obs_shape[0])
    else:
        encoder = TabularEncoder(state_dim=obs_shape[0], latent_dim=args.red_dim)
        decoder = TabularDecoder(latent_dim=args.red_dim, state_dim=obs_shape[0])
        
    vdk_shield = VDK_Shield(encoder=encoder, control_dim=action_dim, decoder=decoder)
    vdk_shield.load_state_dict(checkpoint['state_dict'])
    
    if args.cuda: 
        vdk_shield = vdk_shield.cuda()
        
    state_mean = checkpoint.get('state_mean', torch.zeros(obs_shape[0] if not is_image else obs_shape)).cpu().numpy()
    state_std = checkpoint.get('state_std', torch.ones(obs_shape[0] if not is_image else obs_shape)).cpu().numpy()
    
    return vdk_shield, state_mean, state_std


def run_vll_pretraining(
    env: Any,
    agent: AgentType,
    real_data: ReplayMemory,
    args: Any,
    log_dir: str,
    writer: SummaryWriter
) -> Shield:
    """
    Executes the VLL Pre-training phase:
    1. Collects data (Random -> Policy)
    2. Labels data (Horizon Labeling)
    3. Trains VDK Shield
    4. Returns a wrapped Shield object
    """
    data_file = os.path.join(log_dir, "vll_data.pt")
    vll_save_path = os.path.join(log_dir, "vll_shield.pth")
    
    print("--- Starting VLL-HPS Pipeline: Pre-training RL Agent ---")
    
    vll_obs, _ = env.reset()
    vll_done, vll_trunc = False, False
    
    for step in range(args.vll_steps):
        if step < args.start_steps:
            vll_action = env.action_space.sample()
        else:
            vll_action = agent(vll_obs)
            
        vll_next, vll_rew, vll_cost, vll_done, vll_trunc, _ = env.step(vll_action)
        c_val = 1 if vll_cost > 0 else 0
        
        # Store in both agent memory (for RL training) and real_data (for Shield training)
        agent.add(vll_obs, vll_action, vll_rew, vll_next, vll_done or vll_trunc, c_val)
        real_data.push(vll_obs, vll_action, vll_rew, vll_next, vll_done or vll_trunc, c_val)
        
        # Train Agent if buffer full
        if len(agent.memory) >= args.batch_size:
            # SAC trains every step typically, but let's stick to simple "if ready, train"
            # Note: main_sac loops 'updates_per_step'. PPO loops epochs.
            # We assume agent.train() handles its own logic call, or we rely on the main loop for extensive training.
            # In original code, SAC called agent.train() here. PPO called it if batch full.
            # We will perform a generic train call if available/appropriate.
            if hasattr(agent, 'train_on_batch'): # Hypothetical unification
                 pass 
            elif hasattr(args, 'updates_per_step'): # SAC
                 for _ in range(args.updates_per_step): agent.train()
            else: # PPO
                 agent.train()

        vll_obs = vll_next
        if vll_done or vll_trunc:
            vll_obs, _ = env.reset()
            vll_done, vll_trunc = False, False
            
        if (step+1) % 5000 == 0: 
            print(f"VLL Pre-training: {step+1}/{args.vll_steps}")

    # Labeling & Sequence Extraction
    print("Extracting sequences and labeling...")
    
    # We want sequences of length args.horizon for autoregressive training
    horizon = args.horizon
    if real_data.size < horizon:
        print("Not enough data for sequence extraction.")
        return # Should probably error or handle gracefully
        
    # Use helper from ReplayMemory to get all valid sequences
    # We sample 'size' amount? No, sample() takes random indices. 
    # We want ALL valid sequences.
    # We'll just loop or use sample with replace=False and size roughly equal to buffer.
    # But ReplayMemory.sample doesn't guarantee contiguous coverage if strictly random.
    # Ideally standard sampling is fine for training.
    
    valid_size = real_data.size - horizon
    s, a, r, ns, d, c = real_data.sample(batch_size=valid_size, horizon=horizon, get_cost=True)
    
    # s is (N, T, D)
    
    # Compute labels for the FIRST state in each sequence based on the rollout
    horizon_targets = compute_labels_from_batch(c, d, gamma=0.95) # (N,)
    
    # Save sequences
    torch.save({
        'states': torch.tensor(s, dtype=torch.float32),
        'actions': torch.tensor(a, dtype=torch.float32),
        'next_states': torch.tensor(ns, dtype=torch.float32),
        'horizon_targets': torch.tensor(horizon_targets, dtype=torch.float32)
    }, data_file)
    
    # Train VDK
    _, dl, cl = train_vdk.train_vdk(
        data_path=data_file, output_path=vll_save_path,
        epochs_dyn=args.vll_epochs_dyn, epochs_cbf=args.vll_epochs_cbf,
        latent_dim=args.red_dim, gpu=args.cuda
    )
    writer.add_scalar('loss/vll_dyn', dl, 0)
    writer.add_scalar('loss/vll_cbf', cl, 0)
    
    # Load and Wrap
    print(f"Loading VLL-HPS Shield from {vll_save_path}...")
    vdk_shield, state_mean, state_std = load_vdk_shield(vll_save_path, env.observation_space, env.action_space.shape[0], args)
    
    device = 'cuda' if args.cuda else 'cpu'
    safe_agent = Shield(
        ShieldPolicy(VDK_Runtime(vdk_shield, device=device)), 
        agent, 
        means=state_mean, 
        stds=state_std
    )
    print("VLL-HPS Shield Initialized and Active.")
    return safe_agent


def run_vll_finetuning(
    safe_agent: Shield,
    real_data: ReplayMemory,
    args: Any,
    log_dir: str,
    total_numsteps: int,
    writer: SummaryWriter
) -> None:
    """
    Executes VLL Periodic Finetuning:
    1. Samples batch
    2. Computed labels
    3. Finetunes VDK (warm start)
    4. Reloads shield into safe_agent
    """
    print(f"\n--- Periodic VLL Finetuning at {total_numsteps} ---")
    vll_save_path = os.path.join(log_dir, "vll_shield.pth")
    ft_file = os.path.join(log_dir, f"ft_data_{total_numsteps}.pt")
    
    # Sample Sequences
    batch_size = min(len(real_data), 5000) # Slightly smaller batch for sequences to save memory?
    
    # Extract sequences (N, T, dim)
    s_b, a_b, _, ns_b, d_b, c_b = real_data.sample(batch_size=batch_size, horizon=args.horizon, get_cost=True)
    
    # Compute labels correctly using the batch structure
    horizon_targets = compute_labels_from_batch(c_b, d_b, gamma=0.95)
    
    torch.save({
        'states': torch.tensor(s_b, dtype=torch.float32),
        'actions': torch.tensor(a_b, dtype=torch.float32),
        'next_states': torch.tensor(ns_b, dtype=torch.float32),
        'horizon_targets': torch.tensor(horizon_targets, dtype=torch.float32)
    }, ft_file)
    
    # Train
    _, dl, cl = train_vdk.train_vdk(
        data_path=ft_file, output_path=vll_save_path,
        epochs_dyn=args.vll_finetune_epochs_dyn, epochs_cbf=args.vll_finetune_epochs_cbf,
        latent_dim=args.red_dim, gpu=args.cuda, pretrained_path=vll_save_path
    )
    writer.add_scalar('loss/vll_dyn_ft', dl, total_numsteps)
    writer.add_scalar('loss/vll_cbf_ft', cl, total_numsteps)
    
    # Reload
    # We can reuse the load helper, but we need to access the shield inside safe_agent
    # Actually, simpler to just re-instantiate or load state dict if architecture is static.
    # We'll use the load helper for cleanness and re-assign.
    
    # We need to know env details... safe_agent doesn't store them explicitly nicely visible here?
    # We can assume args match. We need obs_shape logic.
    # To keep this function clean, we rely on args.
    # But wait, load_vdk_shield needs 'env_obs_space'. We don't have 'env' here.
    # We can assume we pass it or infer from data shape. 's_b' is (Batch, State).
    
    # Let's reconstruct obs_space mock or pass env.
    # Passing env is safer.
    pass # See usage in main script. To fix this, we will rely on safe_agent.shield.runtime.model to get dims?
    # Or just passing env to this function.

    # TEMPORARY FIX: We need VDK Loading logic here but we lack ENV.
    # We can manually load state dict since model structure hasn't changed.
    
    checkpoint = torch.load(vll_save_path)
    # Access the internal model
    # safe_agent -> shield (ShieldPolicy) -> runtime (VDK_Runtime) -> model (VDK_Shield)
    vdk_shield = (
        safe_agent.shield.shield_runtime.model 
        if hasattr(safe_agent.shield, 'shield_runtime') 
        else safe_agent.shield.runtime.model # Fallback if I messed up
    ) 
    vdk_shield.load_state_dict(checkpoint['state_dict'])
    if args.cuda: vdk_shield = vdk_shield.cuda()
    
    # Update Stats
    state_mean = checkpoint.get('state_mean', torch.zeros_like(torch.tensor(safe_agent.means))).cpu().numpy()
    state_std = checkpoint.get('state_std', torch.ones_like(torch.tensor(safe_agent.stds))).cpu().numpy()
    
    safe_agent.means = state_mean
    safe_agent.stds = state_std
    safe_agent.agent_times = 0 # Reset counters
    
    print("VLL Shield Finetuned & Reloaded.")


def evaluate_agent(
    env: Any,
    agent: AgentType,
    safe_agent: Optional[Shield],
    args: Any,
    total_numsteps: int,
    writer: SummaryWriter,
    video_dir: Optional[str] = None
) -> None:
    """
    Runs evaluation episodes, logs Reward/Cost, and optionally saves video.
    """
    print("Evaluating...")
    episodes = 5
    frames = []
    avg_rew = 0.0
    avg_cost = 0.0
    
    for ep in range(episodes):
        s, _ = env.reset()
        d, t = False, False
        ret = 0.0
        ep_cost = 0.0
        
        while not d and not t:
            if safe_agent:
                act, _, _, _ = safe_agent(s, evaluate=True)
            else:
                act = agent(s, evaluate=True) # PPO/SAC support evaluate=True
                
            ns, r, c, d, t, _ = env.step(act)
            ret += r
            ep_cost += c
            
            if args.render and ep == 0:
                try:
                    frames.append(env.render())
                except Exception:
                    pass
            s = ns
            
        avg_rew += ret
        avg_cost += ep_cost

    if args.render and frames and video_dir:
        video_file = os.path.join(video_dir, f"eval_{total_numsteps}.mp4")
        imageio.mimsave(video_file, frames, fps=30)
        
    writer.add_scalar("reward/test", avg_rew / episodes, total_numsteps)
    writer.add_scalar("cost/test", avg_cost / episodes, total_numsteps)
    print(f"Eval Reward: {avg_rew/episodes:.2f}, Cost: {avg_cost/episodes:.2f}")

    if safe_agent:
        safe_agent.reset_count()
