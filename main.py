import os
import argparse
import random
import datetime
import itertools
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

# Imports from new structure
from src.policies.shield import Shield, ShieldPolicy
from src.policies.abstract_agent import Agent
from src.policies.sac_agent import SACPolicy
from src.policies.ppo_agent import PPOPolicy
from src.policies.all_c_agent import ALLCAgent
from src.agent_factory import create_agent
from src.shield import VDK_Runtime
from src.experiment_runner import (
    run_vll_pretraining, 
    run_vll_finetuning, 
    evaluate_agent, 
    load_vdk_shield
)
from src.data_collection import compute_horizon_labels
from src.envs import envs
from src.algorithms.sac.replay_memory import ReplayMemory

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.set_num_threads(1)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig):
    # --- 1. Setup Logging & Directories ---
    if args.load_dir:
        pass
    else:
        # Hydra changes CWD to output dir, but user wants logs in project root
        import hydra.utils
        orig_cwd = hydra.utils.get_original_cwd()

        # --- Flatten Config for Compatibility ---
        # Merge 'agent' fields into root to support legacy code access (e.g. args.start_steps)
        # We create a new unrestricted config from the container
        root_container = OmegaConf.to_container(args, resolve=True)
        if 'agent' in root_container:
            agent_conf = root_container.pop('agent')
            # Merge agent params ON TOP of root (or vice versa? Agent usually more specific)
            # But wait, env params like 'start_steps' in ant.yaml (root) should win over default sac.yaml?
            # Hydra already merged defaults. 'start_steps' in ant.yaml is at root. 'start_steps' in sac.yaml is at agent.start_steps.
            # We want args.start_steps to exist.
            # If ant.yaml defined start_steps at root, it exists!
            # Ah, SAC/PPO yaml defines defaults like gamma, batch_size in 'agent'.
            # We want those to be at root IF NOT defined.
            
            # Merge strategy: Root (Env/Global) > Agent Defaults
            # So we take Agent, update with Root.
            merged_dict = agent_conf.copy()
            merged_dict.update(root_container)
            args = OmegaConf.create(merged_dict)
            
            # Re-add agent sub-struct just in case something needs it explicitly (legacy hybrid)
            args.agent = OmegaConf.create(agent_conf)
        
        print(f"Active Agent: {args.name}") 
        
    # --- 1. Setup Logging & Directories (Continued) ---
        from hydra.core.hydra_config import HydraConfig
        log_dir = HydraConfig.get().runtime.output_dir
        print(f"--- Experiment Log Directory: {log_dir} ---")
        
        # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # agent_name = args.name.upper() # args.agent.name -> args.name after flatten
        # name = f"{timestamp}_{agent_name}_{args.env_name}_H{args.horizon}_D{args.red_dim}_S{args.seed}"
        
        # # Route to runs_sac or runs_ppo based on agent type to maintain legacy structure
        # # or just use runs/ ? User specifically asked for runs_ppo / runs_sac.
        # base_log_dir = f"runs_{args.name}"
        
        # log_dir = os.path.join(orig_cwd, base_log_dir, name)
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
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- 3. Agent & Shield Initialization ---
    action_dim = env.action_space.shape[0]
    
    # Factory Creation
    agent = create_agent(args, env)

    vdk_shield = None
    safe_agent = None

    # --- 4. Loading Mode ---
    if args.load_dir:
        print(f"\n=== Loading Experiment from {args.load_dir} ===")
        if not os.path.exists(args.load_dir):
            raise FileNotFoundError(f"Load directory not found: {args.load_dir}")

        # Load Agent
        # Filename depends on agent type? "sac_agent.pth" vs "ppo_agent.pth"
        agent_filename = f"{args.name}_agent.pth"
        agent_path = os.path.join(args.load_dir, agent_filename)
        print(f"Loading Agent from {agent_path}...")
        agent.load_checkpoint(agent_path)

        # --- 4. Load Existing Shield (If Applicable) ---
        # Check standard location and new subdirectory location
        p1 = os.path.join(args.load_dir, "vll_shield.pth")
        p2 = os.path.join(args.load_dir, "dynamics_weights", "vll_shield.pth")
        
        if os.path.exists(p2):
            shield_path = p2
        elif os.path.exists(p1):
            shield_path = p1
        else:
            shield_path = p1 # Default fallback for printing error

        if os.path.exists(shield_path):
            print(f"Loading Shield from {shield_path}...")
            vdk_shield, state_mean, state_std = load_vdk_shield(
                shield_path, env.observation_space, action_dim, args
            )

            if args.cuda and torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

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
        evaluate_agent(
                env, agent, safe_agent, args, 0, None, args.load_dir
        )
        return

    # --- 5. Training Mode Setup ---
    agent_save_path = os.path.join(log_dir, f"{args.name}_agent.pth")
    # vll_save_path unused here, defined in experiment runner inside weights dir

    # Replay Memory (for VLL training & buffering)
    # Note: PPO has internal memory too, but we use this 'real_data' for VLL training
    buffer_size = args.replay_size
        
    real_data = ReplayMemory(
        buffer_size, env.observation_space, action_dim, args.seed
    )

    # VLL Pretraining
    safe_agent = None
    if args.name == 'all_c':
        args.train_vll = False
        
    if args.train_vll:
        safe_agent = run_vll_pretraining(
            env, agent, real_data, args, log_dir, writer
        )
        if hasattr(real_data, 'is_image') and real_data.is_image:
             print("Discarding Pretraining Data (Image Env)...")
             real_data.clear_memory()
        agent.save_checkpoint(agent_save_path)

    # --- 6. Main Loop ---
    total_numsteps = 0
    total_real_episodes = 0
    real_unsafe_episodes = 0

    last_vll_finetune_step = 0
    last_eval_step = 0
    iterator_loop = itertools.count(1)
    
    # Determine Training frequency
    is_sac = (args.name == "sac")
    is_all_c = (args.name == "all_c")
    is_ppo = (args.name == "ppo")

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
                reward = reward - 100 # Penalty

            # Store in Agent Memory (for its own training)
            agent.add(state, policy_act, reward, next_state, done or trunc, cost)
            
            # Store in Real Data Memory (for VLL training)
            real_data.push(state, action, reward, next_state, done or trunc, cost)

            # --- ALG SPECIFIC TRAINING ---
            if is_sac:
                 if len(agent.memory) > args.batch_size:
                    # Number of updates per step (normally 1)
                    for i in range(args.updates_per_step):    
                    
                        # Standard Agent Training
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.train()
                        
                        writer.add_scalar('loss/critic_1', critic_1_loss, total_numsteps)
                        writer.add_scalar('loss/critic_2', critic_2_loss, total_numsteps)
                        writer.add_scalar('loss/policy', policy_loss, total_numsteps)
                        writer.add_scalar('loss/entropy_loss', ent_loss, total_numsteps)
                        writer.add_scalar('entropy_temprature/alpha', alpha, total_numsteps)
        
            if is_all_c:
                if len(agent.memory) > args.batch_size:
                    # Number of updates per step (normally 1)
                    for i in range(args.updates_per_step):    
                        # Hybrid ALL-C Training
                        # This call internally handles:
                        # 1. SAC Teacher update (every call)
                        # 2. VDK/Student On-Policy update (if buffer full)
                        sac_stats = agent.teacher.train()

                        # Convert sac_stats tuple to dict for consistent return
                        train_stats = {
                            'critic_1': sac_stats[0],
                            'critic_2': sac_stats[1],
                            'policy': sac_stats[2],
                            'entropy': sac_stats[3],
                            'alpha': sac_stats[4],
                        }
                        
                        if isinstance(train_stats, dict) and train_stats:
                            # Log SAC Stats
                            writer.add_scalar("loss/critic_1", train_stats.get('critic_1', 0), total_numsteps)
                            writer.add_scalar("loss/critic_2", train_stats.get('critic_2', 0), total_numsteps)
                            writer.add_scalar("loss/policy", train_stats.get('policy', 0), total_numsteps)
                            writer.add_scalar("loss/entropy_loss", train_stats.get('entropy', 0), total_numsteps)
                            writer.add_scalar("loss/alpha_value", train_stats.get('alpha', 0), total_numsteps)
                            
                            # Log VDK/Student stats if they occurred
                train_stats = agent.train()
                if 'vdk_loss' in train_stats:
                    writer.add_scalar('loss/vdk_dyn', train_stats['vdk_loss'], total_numsteps)
                    writer.add_scalar('loss/student_val', train_stats['student_loss'], total_numsteps)
            

            elif is_ppo:
                # Access internal memory size via agent (PPOPolicy wrappers)
                if len(agent.memory) >= args.batch_size:
                    losses = agent.train()
                    writer.add_scalar("loss/policy", losses["avg_policy_loss"], total_numsteps)
                    writer.add_scalar("loss/value", losses["avg_value_loss"], total_numsteps)
                    writer.add_scalar("loss/entropy", losses["avg_entropy_loss"], total_numsteps)
                    writer.add_scalar("loss/total", losses["avg_total_loss"], total_numsteps)
                    writer.add_scalar("ppo/clip_fraction", losses["avg_clip_fraction"], total_numsteps)
                    writer.add_scalar("ppo/kl_divergence", losses["avg_kl_divergence"], total_numsteps)
                    writer.add_scalar("ppo/explained_variance", losses["avg_explained_variance"], total_numsteps)

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
            agent.save_checkpoint(agent_save_path)
            run_vll_finetuning(
                safe_agent, real_data, args, log_dir, total_numsteps, writer
            )
            if hasattr(real_data, 'is_image') and real_data.is_image:
                 print("Discarding Finetuning Data (Image Env)...")
                 real_data.clear_memory()

        # Eval
        if total_numsteps - last_eval_step >= args.eval_steps:
            last_eval_step = total_numsteps
            agent.save_checkpoint(agent_save_path)
            evaluate_agent(
                env, agent, safe_agent, args, total_numsteps, writer, video_dir
            )

    print("Training Complete.")
    agent.save_checkpoint(agent_save_path)

if __name__ == "__main__":
    main()
