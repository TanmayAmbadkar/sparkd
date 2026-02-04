import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import random
import datetime
import itertools
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from src.policy import Shield, PPOPolicy, ShieldPolicy
from src.shield import VDK_Runtime
from src.experiment_runner import (
    run_vll_pretraining, 
    run_vll_finetuning, 
    evaluate_agent, 
    load_vdk_shield
)
from src.data_collection import compute_horizon_labels

from benchmarks import envs
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
import imageio
import time

torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="conf", config_name="ppo_config")
def main(args: DictConfig):
    # --- 1. Setup Logging & Directories ---
    if args.load_dir:
        pass
    else:
        # Hydra changes CWD to output dir, but user wants logs in project root
        import hydra.utils
        orig_cwd = hydra.utils.get_original_cwd()
        
        # Construct Name based on User Request
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Helper to get value or empty string if missing (using getattr for safety or explicit check)
        # Note: DictConfig supports .get()
        cbf_gamma = args.get("cbf_gamma", "N/A") 
        percentile = args.get("percentile", "N/A")
        safe_str = "_safe" if args.get("train_vll", False) else "" # Assuming train_vll implies safety or check strict args
        # Actually user example used args.no_safety. We check if shield logic is active.
        # Let's try to match as best as possible with available fields.
        
        name = f"{timestamp}_PPO_{args.env_name}_H{args.horizon}_D{args.red_dim}_S{args.seed}"
        
        log_dir = os.path.join(orig_cwd, "runs_ppo", name)
        video_dir = os.path.join(log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        OmegaConf.save(args, os.path.join(log_dir, "config.yaml"))
        print(f"--- Experiment Log Directory: {log_dir} ---")

    # --- 2. Environment Setup ---
    render_mode = "rgb_array" if args.render else None
    env = envs.get_env_from_name(args.env_name, render_mode=render_mode)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # --- 3. Agent & Shield Initialization ---
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # PPO Agent
    # Note: args.batch_size in PPO context usually refers to update_batch_size (e.g. 2048)
    # args.mini_batch_size is optimization batch
    agent = PPOPolicy(env, args.replay_size, args.seed, args.batch_size, args)

    vdk_shield = None
    safe_agent = None

    # --- 4. Loading Mode ---
    if args.load_dir:
        print(f"\n=== Loading Experiment from {args.load_dir} ===")
        if not os.path.exists(args.load_dir):
            raise FileNotFoundError(f"Load directory not found: {args.load_dir}")

        # Load Agent
        agent_path = os.path.join(args.load_dir, "ppo_agent.pth")
        print(f"Loading Agent from {agent_path}...")
        agent.load_checkpoint(agent_path)

        # Load Shield
        shield_path = os.path.join(args.load_dir, "vll_shield.pth")

        if os.path.exists(shield_path):
            print(f"Loading Shield from {shield_path}...")
            vdk_shield, state_mean, state_std = load_vdk_shield(
                shield_path, env.observation_space, action_dim, args
            )

            device = "cuda" if args.cuda else "cpu"
            safe_agent = Shield(
                ShieldPolicy(VDK_Runtime(vdk_shield, device=device)),
                agent,
                means=state_mean,
                stds=state_std,
            )
            print("Shield loaded and active.")
        else:
            print("No shield found. Running Unsafe.")

        # Eval
        print("\n--- Starting Evaluation ---")
        avg_reward = 0.0
        episodes = 10
        frames = []
        for ep in range(episodes):
            s, _ = env.reset()
            d, t = False, False
            ret = 0
            while not d and not t:
                if safe_agent:
                    act, _, _, _ = safe_agent(s)
                else:
                    act = agent(s, evaluate=True)
                ns, r, _, d, t, _ = env.step(act)
                ret += r
                if args.render and ep == 0:
                    try:
                        frames.append(env.render())
                    except:
                        pass
                s = ns
            print(f"Eval Episode {ep+1}: Reward {ret:.2f}")
            avg_reward += ret

        if args.render and frames:
            video_path = os.path.join(args.load_dir, "eval_playback.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved video to {video_path}")

        print(f"Average Reward: {avg_reward/episodes:.2f}")
        return

    # --- 5. Training Mode Setup ---
    agent_save_path = os.path.join(log_dir, "ppo_agent.pth")
    vll_save_path = os.path.join(log_dir, "vll_shield.pth")

    # Replay Memory (for VLL training & buffering PPO batch)
    real_data = ReplayMemory(
        args.replay_size, env.observation_space, action_dim, args.seed
    )

    # VLL Pretraining
    if args.train_vll:
        safe_agent = run_vll_pretraining(
            env, agent, real_data, args, log_dir, writer
        )
        agent.agent.save_checkpoint(agent_save_path)

    # --- 6. Main Loop ---
    total_numsteps = 0
    total_real_episodes = 0
    real_unsafe_episodes = 0

    last_vll_finetune_step = 0
    last_eval_step = 0
    iterator_loop = itertools.count(1)

    while total_numsteps < args.num_steps:
        i_episode = next(iterator_loop)
        ep_rew = 0
        ep_steps = 0
        state, _ = env.reset()
        done = False
        trunc = False
        unsafe_flag = False
        episode_cost = 0

        while not done and not trunc:
            if safe_agent:
                # Shield returns: act, shielded, diff, proposed
                action, shielded, _, policy_act = safe_agent(state)
            else:
                policy_act = agent(state)
                action = policy_act
                shielded = "N"

            next_state, reward, cost, done, trunc, _ = env.step(action)
            ep_steps += 1
            total_numsteps += 1
            ep_rew += reward

            if cost > 0:
                if not unsafe_flag:
                    real_unsafe_episodes += 1
                unsafe_flag = True

            # PPO Storage
            # Note: PPOPolicy stores (state, action, reward, next_state, done, cost)
            # Make sure we store the POLICY action unless we want to learn from shielded action?
            # Standard shield: learn from proposed (safe exploration) or shielded (imitation)?
            # Legacy code stored 'policy_act' in agent, 'action' (execution) in real_data.
            # agent.add(state, policy_act...)
            agent.add(state, policy_act, reward, next_state, done or trunc, cost)
            real_data.push(state, action, reward, next_state, done or trunc, cost)

            if len(agent.memory) >= args.batch_size:
                losses = agent.train()
                writer.add_scalar(
                    "loss/policy", losses["avg_policy_loss"], total_numsteps
                )
                writer.add_scalar(
                    "loss/value", losses["avg_value_loss"], total_numsteps
                )
                writer.add_scalar(
                    "loss/entropy", losses["avg_entropy_loss"], total_numsteps
                )
                writer.add_scalar(
                    "loss/total", losses["avg_total_loss"], total_numsteps
                )
                writer.add_scalar(
                    "ppo/clip_fraction", losses["avg_clip_fraction"], total_numsteps
                )
                writer.add_scalar(
                    "ppo/kl_divergence", losses["avg_kl_divergence"], total_numsteps
                )
                writer.add_scalar(
                    "ppo/explained_variance",
                    losses["avg_explained_variance"],
                    total_numsteps,
                )

            state = next_state
            episode_cost += cost

        total_real_episodes += 1
        writer.add_scalar("reward/train", ep_rew, total_numsteps)
        print(
            f"Ep {i_episode}: Steps {ep_steps}, Reward {ep_rew:.2f}, Cost {episode_cost}, Total {total_numsteps}"
        )

        if safe_agent:
            s, a, b, _ = safe_agent.report()
            print(f"Shield: {s}, Neural: {a}")
            safe_agent.reset_count()

        # Periodic VLL
        if (
            args.train_vll
            and total_numsteps >= args.start_steps
            and (total_numsteps - last_vll_finetune_step) >= args.vll_finetune_steps
        ):
            last_vll_finetune_step = total_numsteps
            agent.agent.save_checkpoint(agent_save_path)
            run_vll_finetuning(
                safe_agent, real_data, args, log_dir, total_numsteps, writer
            )

        # Eval
        if total_numsteps - last_eval_step >= args.eval_steps:
            last_eval_step = total_numsteps
            agent.agent.save_checkpoint(agent_save_path)
            evaluate_agent(
                env, agent, safe_agent, args, total_numsteps, writer, video_dir
            )

    print("Training Complete.")
    agent.agent.save_checkpoint(agent_save_path)


if __name__ == "__main__":
    main()
