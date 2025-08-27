import argparse
import datetime
import itertools
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from src.policy import CPOPolicy, PCRPOPolicy, CUPPolicy, P3OPolicy, PPOPolicy
from src.cost_function import CostFunction
from koopman.env_model import get_environment_model
from abstract_interpretation import domains
from benchmarks import envs
from src.replay_memory import ReplayMemory
import gymnasium as gym 
import matplotlib.pyplot as plt

import traceback

parser = argparse.ArgumentParser(description='Safe CPO/PCRPO/CUP Args')

# --- ENV/GENERAL ---
parser.add_argument('--env_name', default="hopper")
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--cuda', action="store_true", default=False, help="Use CUDA if available")
parser.add_argument('--save_dir', type=str, default='runs_analysis', help="Directory to save logs/models")
parser.add_argument('--log_interval', type=int, default=10, help="Logging interval in episodes")
parser.add_argument('--eval_episodes', type=int, default=10, help="Number of episodes to use for evaluation")

# --- RL TRAINING LOOP ---
parser.add_argument('--num_steps', type=int, default=200000)
parser.add_argument('--start_steps', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--mini_batch_size', type=int, default=64)
parser.add_argument('--replay_size', type=int, default=200000)
parser.add_argument('--horizon', type=int, default=5)

# --- POLICY/CRITIC NETWORK ---
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--red_dim', type=int, default=20)
parser.add_argument('--log_std_min', type=float, default=-20)
parser.add_argument('--log_std_max', type=float, default=1)

# --- OPTIMIZATION/LOSS ---
parser.add_argument('--actor_lr', type=float, default=3e-4)
parser.add_argument('--critic_lr', type=float, default=3e-4)
parser.add_argument('--max_grad_norm', type=float, default=0.5, help="Gradient clipping max norm")
parser.add_argument('--value_coeff', type=float, default=0.5, help="Coefficient for value loss")
parser.add_argument('--entropy_coeff', type=float, default=0.00, help="Coefficient for entropy loss")

# --- ADVANTAGE/GAE ---
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--cost_gamma', type=float, default=0.95, help="Discount factor for costs")
parser.add_argument('--lam', type=float, default=0.995, help="GAE lambda")
parser.add_argument('--gae_bias_correction', action='store_true', help="Enable bias-corrected GAE (CUP)")

# --- PPO-STYLE / CONSTRAINTS ---
parser.add_argument('--eps_clip', type=float, default=0.2, help="Clipping parameter for PPO/CPO objectives")

# --- CONSTRAINTS (Multi-constraint support) ---
parser.add_argument('--num_costs', type=int, default=1, help="Number of cost constraints")
parser.add_argument('--cost_limit', type=float, default=0, help="Cost constraint (max cost per episode)")
parser.add_argument('--lagrange_init', type=float, default=1.0,
                    help="Initial values for Lagrange multipliers (list or single value)")
parser.add_argument('--lagrange_lr', type=float, default=0.01, help="Learning rate for Lagrange multiplier update")
parser.add_argument('--no_safety', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')

# --- PCRPO-specific ---
parser.add_argument('--switching_temp', type=float, default=1.0, help="Switching temperature for PCRPO gradient blending")
parser.add_argument('--slack_coef', type=float, default=0, help="Slack coefficient (PCRPO, if using slacks)")

# --- CUP-specific ---
parser.add_argument('--cup_trust_region', type=float, default=0.1, help="Trust region size (KL bound) for CUP")

# --- P3O-specific arguments ---

parser.add_argument('--p3o_penalty_init', type=float, default=1.0,
                   help="Initial penalty multiplier (lambda) for P3O (default: 1.0)")
parser.add_argument('--p3o_penalty_lr', type=float, default=0.01,
                   help="Learning rate for penalty multiplier update in P3O dual ascent")
parser.add_argument('--p3o_penalty_learn', action='store_true', default=False,
                   help="Whether to adaptively learn the penalty multiplier during training (dual ascent). If not set, penalty is fixed.")

# --- MISC ---
parser.add_argument('--save_model', action='store_true', help="Save model after training")
parser.add_argument('--load_model_path', type=str, default=None, help="Path to load a checkpoint")
parser.add_argument('--eval_only', action='store_true', help="Run evaluation only")

# --- EVAL/LOGGING ---
parser.add_argument('--log_std_eval', type=float, default=None, help="Set a fixed log std for evaluation policy")

# --- LEGACY/DEPRECATED ---
# parser.add_argument('--legacy_flag', action='store_true')


parser.add_argument('--policy', type=str, default="cup", help="Policy class to choose from: cpo or pcrpo or cup")
args = parser.parse_args()

# Setup environment
env = envs.get_env_from_name(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

hyperparams = vars(args)

# Tensorboard
if not os.path.exists("runs"):
    os.makedirs("runs")
writer = SummaryWriter('runs/{}_{}_{}_{}H{}_D{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, 
    ("safecost" if not args.no_safety else "nocost"),
    args.horizon, args.red_dim))

print(hyperparams)
file = open('runs/{}_{}_{}_{}H{}_D{}/log.txt'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy,
    ("safecost" if not args.no_safety else "nocost"),
    args.horizon, args.red_dim), "w+")

# CPO agent setup

# Replay memories
real_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)

iterator_loop = itertools.count(1)

real_unsafe_episodes = 0
total_real_episodes = 0
total_numsteps = 0
if args.policy == "cpo":
    agent = CPOPolicy(env, args.batch_size, args.seed, args.batch_size, args)
elif args.policy == "cup":
    agent = CUPPolicy(env, args.batch_size, args.seed, args.batch_size, args)
elif args.policy == "pcrpo":
    agent = PCRPOPolicy(env, args.batch_size, args.seed, args.batch_size, args)
elif args.policy == "p3o":
    agent = P3OPolicy(env, args.batch_size, args.seed, args.batch_size, args)
elif args.policy == "ppo":
    agent = PPOPolicy(env, args.batch_size, args.seed, args.batch_size, args)
cost_function = None

# Training loop

env_model = None
unsafe_test_episodes = 0
total_test_episodes = 0
unsafe_sim_episodes = 0
total_sim_episodes = 0
train_steps = 1
while True:
    i_episode = next(iterator_loop)
    episode_reward = 0
    episode_cost = 0
    episode_steps = 0
    done = False
    trunc = False
    state, info = env.reset()
    unsafe_flag = False
    trajectory = [state]
    if True:

        print(i_episode, ": Real data")
        tmp_buffer = []
        real_buffer = []
        
        flags = []
        
        while not done and not trunc:
            
            action = agent(state)
            
            next_state, reward, done, trunc, info = env.step(action)
                
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            

            cost = 0
            
            trajectory.append(next_state)
            
            if cost_function is not None:
                cost = cost_function(state, action)

            if env.unsafe(next_state, False):

                real_unsafe_episodes += 1 * (not unsafe_flag)
                episode_reward -= 100 * (not unsafe_flag)
                reward -=100
                print("UNSAFE (outside testing)")
                print(f"{np.round(state, 2)}", "\n", action, "\n", f"{np.round(next_state, 2)}")
                done = done or (True if cost_function is not None else False) or args.no_safety
                cost += 1

                unsafe_flag = True or unsafe_flag
                
            episode_cost += cost
            agent.add(state, action, reward, next_state, done or trunc, cost)
            real_data.push(state, action, reward, next_state, done or trunc, cost)
            
            
            if len(agent.memory) >= args.batch_size:
                losses = agent.train()
                for key, value in losses.items():
                    writer.add_scalar(f'loss/{key}', value, total_numsteps)

            state = next_state
        total_real_episodes += 1 

    
    
    if total_numsteps >= args.start_steps * train_steps and args.no_safety is False:
    # if False:
        train_steps*=2
        try:
            
            states, actions, rewards, next_states, dones, costs = \
                real_data.sample(batch_size = min(len(real_data), 100000), get_cost=True, remove_samples=False, horizon = args.horizon)
                
            
        except Exception as e:
            
            print(traceback.format_exc())
            print("Error in sampling")
            exit()
        
        if env_model is not None:
            env_model.mars.koopman_model.lr = 0.001
            koopman_model = env_model.mars.koopman_model
            epochs = 50
        else:
            koopman_model = None
            epochs = 100

        
        env_model, ev_score, r2_score, mean, std = get_environment_model(
                states, actions, next_states, rewards,
                domains.DeepPoly(env.observation_space.low, env.observation_space.high),
                seed=args.seed, koopman_model = koopman_model, latent_dim=args.red_dim, horizon = args.horizon, epochs= epochs)
        
        writer.add_scalar(f'loss/ev_koopman', ev_score, total_numsteps)   
        writer.add_scalar(f'loss/r2_score', r2_score, total_numsteps)   
            
        # new_obs_space_domain = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.koopman_model.embedding_net.embed_net, domains.DeepPoly(env.observation_space.low, env.observation_space.high)))
        
        
        # safety_domain = domains.DeepPoly(env.safety.lower, env.safety.upper)
        
        # safety = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.koopman_model.embedding_net.embed_net, safety_domain))
        
        
        
        
        safety = domains.DeepPoly(torch.hstack([env.safety.lower, -torch.ones(env.safety.lower.shape[0], args.red_dim)]), torch.hstack([env.safety.upper, torch.ones(env.safety.upper.shape[0], args.red_dim)]))

        if args.red_dim != 0:
            safety.lower[:, :-args.red_dim] = (safety.lower[:, :-args.red_dim] - mean)/(std + 1e-8)
            safety.upper[:, :-args.red_dim] = (safety.upper[:, :-args.red_dim] - mean)/(std + 1e-8)
        else:
            safety.lower = (safety.lower - mean)/(std + 1e-8)
            safety.upper = (safety.upper - mean)/(std + 1e-8)

        if args.red_dim != 0:
            new_obs_space = gym.spaces.Box(low=np.concatenate([np.nan_to_num(env.observation_space.low, nan=-9999, posinf=33333333, neginf=-33333333), -np.ones(args.red_dim, )]), high=np.concatenate([np.nan_to_num(env.observation_space.high, nan=-9999, posinf=33333333, neginf=-33333333), np.ones(args.red_dim, )]), shape=(args.red_dim + env.observation_space.shape[0],))
            new_obs_space.low[:-args.red_dim] = (new_obs_space.low[:-args.red_dim] - mean)/(std + 1e-8)
            new_obs_space.high[:-args.red_dim] = (new_obs_space.high[:-args.red_dim] - mean)/(std + 1e-8)            
        else:
            new_obs_space = gym.spaces.Box(low=np.nan_to_num(env.observation_space.low, nan=-9999, posinf=33333333, neginf=-33333333), high=np.nan_to_num(env.observation_space.high, nan=-9999, posinf=33333333, neginf=-33333333), shape=(env.observation_space.shape[0],))
            new_obs_space.low = (new_obs_space.low - mean)/(std + 1e-8)
            new_obs_space.high = (new_obs_space.high - mean)/(std + 1e-8)
        
        polys = safety.to_hyperplanes(new_obs_space)
        print(polys)
        unsafe_domains = safety.invert_polytope(new_obs_space)
        env.transformed_safe_polys = polys
        env.transformed_polys = unsafe_domains
        
        print("LATENT SAFETY", safety, len(polys[0]))
        print("LATENT UNSAFETY",  len(env.transformed_polys))
        cost_function = CostFunction(
            env_model.get_symbolic_model(), new_obs_space, env.observation_space,
            env.action_space, args.horizon, env.transformed_polys, env.transformed_safe_polys, env_model.mars.koopman_model.transform,
            mean, std)
        

    # Test the agent periodically
    
    writer.add_scalar(f'reward/train', episode_reward, total_numsteps)
    writer.add_scalar(f'cost/train', episode_cost, total_numsteps)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}"
          .format(i_episode, total_numsteps,
                  episode_steps, round(episode_reward, 2), round(episode_cost, 2)))
   
    if i_episode % 10 == 0:
        print("starting testing...")
        avg_reward = 0.
        avg_cost = 0.
        episodes = 1
        unsafe_episodes = 0
        avg_length = 0.
        shield_count = 0
        backup_count = 0
        neural_count = 0
        t = 0

        for episode_num in range(episodes):
            # record_video = episode_num % 2 == 0  # Record every alternate episode (example condition)
            custom_filename = f"videos/episode_{i_episode}.mp4"
            
            # video_env.video_recorder.file_prefix = os.path.join("videos/", f"{custom_filename.split('.')[0]}")
            
            state, info = env.reset()
            episode_reward = 0
            episode_cost = 0
            done = False
            trunc = False
            episode_steps = 0
            trajectory = [state]
            # frames  = [env.render()]

            while not done and not trunc:
                # Decide action
                action = agent(state)
                

                next_state, reward, done, trunc, info = env.step(action)
                episode_reward += reward
                episode_cost += 0 if cost_function is None else cost_function(state, action)
                episode_steps += 1

                if episode_steps >= env._max_episode_steps:
                    done = True
                if env.unsafe(next_state, False):
                    print("UNSAFE Inside testing")
                    episode_reward += -100
                    print(f"{np.round(state, 2)}", "\n", action, "\n", f"{np.round(next_state, 2)}")
                    unsafe_episodes += 1
                    episode_cost += 1
                    done = True


                state = next_state
                trajectory.append(state)
                # frames.append(env.render())

            # imageio.mimsave(custom_filename, frames, fps=30)
            avg_reward += episode_reward
            avg_cost += episode_cost
            avg_length += episode_steps

            avg_reward /= episodes
            avg_cost /= episodes
            avg_length /= episodes
            shield_count /= episodes
            neural_count /= episodes
            backup_count /= episodes
            unsafe_test_episodes+=unsafe_episodes
            total_test_episodes+=episodes
            writer.add_scalar(f'agent/shield', shield_count, total_numsteps)
            writer.add_scalar(f'agent/neural', neural_count, total_numsteps)
            writer.add_scalar(f'agent/backup', backup_count, total_numsteps)
            writer.add_scalar(f'agent/unsafe_real_episodes', real_unsafe_episodes, total_numsteps)
            writer.add_scalar(f'agent/unsafe_real_episodes_ratio', real_unsafe_episodes/total_real_episodes, total_numsteps)
            writer.add_scalar(f'agent/unsafe_test_episodes', unsafe_test_episodes, total_numsteps)
            writer.add_scalar(f'agent/unsafe_test_episodes_ratio', (unsafe_test_episodes+0.0000001)/(total_test_episodes + 0.0000001), total_numsteps)
            writer.add_scalar(f'reward/test', avg_reward, total_numsteps)
            writer.add_scalar(f'cost/test', avg_cost, total_numsteps)
            print("----------------------------------------")
            print("Test Episodes: {}, Unsafe: {}, Avg. Length: {}, Avg. Reward: {}, Avg. Cost: {}"
                .format(episodes, unsafe_episodes, round(avg_length, 2),
                        round(avg_reward, 2), round(avg_cost, 2)))
            print("----------------------------------------")
            if (i_episode - 99) % 100 == 0:
                print("Trajectory:")
                print(trajectory)    
            # total_episodes += 1 
        
        
    if total_numsteps > args.num_steps:
        break
    
total_episodes = next(iterator_loop) - 1
print("Total unsafe real:", real_unsafe_episodes, "/", total_real_episodes)
print("Total unsafe real:", real_unsafe_episodes, "/", total_real_episodes, file=file)
print("Total unsafe sim:", unsafe_sim_episodes, "/", total_sim_episodes)
print("Total unsafe sim:", unsafe_sim_episodes, "/", total_sim_episodes, file=file)
print("Total unsafe Test:", unsafe_test_episodes, "/", total_test_episodes)
print("Total unsafe Test:", unsafe_test_episodes, "/", total_test_episodes, file=file)
print("Using SPICE:", not args.no_safety)
print("Using SPICE:", not args.no_safety, file=file)

writer.add_hparams(
    hparam_dict = hyperparams, 
    metric_dict = {
        "Unsafe Real Episodes": real_unsafe_episodes, 
        "Unsafe Sim Episodes": unsafe_sim_episodes, 
        "Unsafe Test Episodes": unsafe_test_episodes,
        "Total Real Episodes": total_real_episodes, 
        "Total Sim Episodes": total_sim_episodes, 
        "Total Test Episodes":total_test_episodes
    }
)
