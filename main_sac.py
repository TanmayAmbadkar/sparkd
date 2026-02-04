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
from src.policy import Shield, SACPolicy, ShieldPolicy
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


@hydra.main(version_base=None, config_path="conf", config_name="sac_config")
def main(args: DictConfig):
    # --- 1. Setup Logging & Directories ---
    if args.load_dir:
        pass
    else:
        # Hydra changes CWD to output dir, but user wants logs in project root
        import hydra.utils
        orig_cwd = hydra.utils.get_original_cwd()
        
        # Construct Name
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        name = f"{timestamp}_SAC_{args.env_name}_H{args.horizon}_D{args.red_dim}_S{args.seed}"
        
        log_dir = os.path.join(orig_cwd, "runs_sac", name)
        video_dir = os.path.join(log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        # Save resolved config
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

    # SAC Agent
    agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)

    # Shield Components
    vdk_shield = None
    safe_agent = None

    # --- 4. Loading Mode (Load -> Eval -> Exit) ---
    if args.load_dir:
        print(f"\n=== Loading Experiment from {args.load_dir} ===")
        if not os.path.exists(args.load_dir):
            raise FileNotFoundError(f"Load directory not found: {args.load_dir}")

        # Load Agent
        agent_path = os.path.join(args.load_dir, "sac_agent.pth")
        print(f"Loading Agent from {agent_path}...")
        agent.load_checkpoint(agent_path)  # SACPolicy has load_checkpoint

        # Load Shield (if it exists)
        # Try finding the VLL model file. Previous convention: model_vll_{env}.pth or internal name.
        # Ideally we saved it as 'vll_shield.pth' or similar.
        # Based on refactor below, we will save it as `vll_shield.pth` inside log_dir.
        shield_path = os.path.join(args.load_dir, "vll_shield.pth")

        if os.path.exists(shield_path):
            print(f"Loading Shield from {shield_path}...")
            vdk_shield, state_mean, state_std = load_vdk_shield(
                shield_path, env.observation_space, action_dim, args
            )

            device = "cuda" if args.cuda else "cpu"
            runtime = VDK_Runtime(vdk_shield, device=device)
            shield_adapter = ShieldPolicy(runtime)
            safe_agent = Shield(shield_adapter, agent, means=state_mean, stds=state_std)
            print("Shield loaded and active.")
        else:
            print("No shield model found in directory. Running with Unsafe Agent.")

        # Evaluation Loop
        print("\n--- Starting Evaluation ---")
        avg_reward = 0.0
        episodes = 10
        frames = []

        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            trunc = False
            ep_ret = 0

            while not done and not trunc:
                if safe_agent:
                    action, _, _, _ = safe_agent(state)
                else:
                    action = agent(state, evaluate=True)

                next_state, r, _, done, trunc, _ = env.step(action)
                ep_ret += r

                if args.render and ep == 0:
                    try:
                        frames.append(env.render())
                    except:
                        pass

                state = next_state

            print(f"Eval Episode {ep+1}: Reward {ep_ret:.2f}")
            avg_reward += ep_ret

        if args.render and frames:
            video_path = os.path.join(args.load_dir, "eval_playback.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved eval video to {video_path}")

        print(f"Average Reward: {avg_reward / episodes:.2f}")
        return  # Exit after eval

    # --- 5. Training Mode Setup ---
    # Define save paths inside log_dir
    agent_save_path = os.path.join(log_dir, "sac_agent.pth")
    vll_save_path = os.path.join(log_dir, "vll_shield.pth")

    # Replay Memory
    real_data = ReplayMemory(
        args.replay_size, env.observation_space, action_dim, args.seed
    )

    # --- VLL-HPS Integration ---
    if args.train_vll:
        safe_agent = run_vll_pretraining(
            env, agent, real_data, args, log_dir, writer
        )
        # Save Initial Agent
        agent.agent.save_checkpoint(env_name=args.env_name, ckpt_path=agent_save_path)

    # --- 6. Main Training Loop ---
    total_numsteps = 0
    total_real_episodes = 0
    real_unsafe_episodes = 0

    last_vll_finetune_step = 0
    last_eval_step = 0

    iterator_loop = itertools.count(1)

    while total_numsteps < args.num_steps:
        i_episode = next(iterator_loop)
        episode_reward = 0
        episode_steps = 0
        done = False
        trunc = False
        state, _ = env.reset()
        unsafe_flag = False
        episode_cost = 0

        while not done and not trunc:
            if safe_agent:
                action, shielded, _, _ = safe_agent(state)
            else:
                action = agent(state)
                shielded = "N"

            next_state, reward, cost, done, trunc, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += cost

            if cost > 0:
                if not unsafe_flag:
                    real_unsafe_episodes += 1
                unsafe_flag = True
                reward = reward - 100  # Penalty

            # Training
            agent.add(state, action, reward, next_state, done or trunc, cost)
            real_data.push(state, action, reward, next_state, done or trunc, cost)

            if len(agent.memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    c1l, c2l, pl, el, al = agent.train()
                    writer.add_scalar("loss/critic_1", c1l, total_numsteps)
                    writer.add_scalar("loss/critic_2", c2l, total_numsteps)
                    writer.add_scalar("loss/policy", pl, total_numsteps)
                    writer.add_scalar("loss/entropy_loss", el, total_numsteps)
                    writer.add_scalar("loss/alpha_value", al, total_numsteps)

            state = next_state

        total_real_episodes += 1
        writer.add_scalar("reward/train", episode_reward, total_numsteps)
        print(
            f"Ep {i_episode}: Steps {episode_steps}, Reward {episode_reward:.2f}, Cost {episode_cost}, Total {total_numsteps}"
        )

        if safe_agent:
            s, a, b, _ = safe_agent.report()
            print(f"Shield: {s}, Neural: {a}")
            safe_agent.reset_count()

        # --- Periodic VLL Finetune ---
        if (
            args.train_vll
            and total_numsteps >= args.start_steps
            and (total_numsteps - last_vll_finetune_step) >= args.vll_finetune_steps
        ):
            last_vll_finetune_step = total_numsteps
            
            # Save Agent checkpoint
            agent.agent.save_checkpoint(
                env_name=args.env_name, ckpt_path=agent_save_path
            )
            
            run_vll_finetuning(
                safe_agent, real_data, args, log_dir, total_numsteps, writer
            )

        # --- Evaluation ---
        if total_numsteps - last_eval_step >= args.eval_steps:
            last_eval_step = total_numsteps
            # Save Checkpoint
            agent.agent.save_checkpoint(
                env_name=args.env_name, ckpt_path=agent_save_path
            )
            evaluate_agent(
                env, agent, safe_agent, args, total_numsteps, writer, video_dir
            )

    print("Training Complete.")
    agent.agent.save_checkpoint(env_name=args.env_name, ckpt_path=agent_save_path)


if __name__ == "__main__":
    main()
