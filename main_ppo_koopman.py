import argparse
import datetime
import itertools
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO, ActorCritic
from src.policy import Shield, PPOPolicy, ProjectionPolicy
from koopman.env_model import get_environment_model
from abstract_interpretation import domains, verification
from benchmarks import envs
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
import gymnasium as gym 

import traceback

parser = argparse.ArgumentParser(description='Safe PPO Args')
parser.add_argument('--env_name', default="lunar_lander")
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--mini_batch_size', type=int, default=64)
parser.add_argument('--num_steps', type=int, default=200000)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--replay_size', type=int, default=200000)
parser.add_argument('--start_steps', type=int, default=5000)
parser.add_argument('--cuda', action="store_true")
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--red_dim', type=int, default = 20)
parser.add_argument('--no_safety', default=False, action='store_true')
args = parser.parse_args()

# Setup environment
env = envs.get_env_from_name(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

hyperparams = vars(args)

# Tensorboard
if not os.path.exists("runs_ppo"):
    os.makedirs("runs_ppo")
writer = SummaryWriter('runs_ppo/{}_PPO_{}_H{}_D{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.horizon, args.red_dim))

print(hyperparams)
if not os.path.exists("logs_ppo"):
    os.makedirs("logs_ppo")

file = open('runs_ppo/{}_PPO_{}_H{}_D{}/log.txt'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.horizon, args.red_dim), "w+")

# PPO agent setup

# Replay memories
real_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)


iterator_loop = itertools.count(1)

real_unsafe_episodes = 0
total_real_episodes = 0
total_numsteps = 0
agent = PPOPolicy(env, args.batch_size, args.seed, args.batch_size, args)
safe_agent = None

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
    episode_steps = 0
    done = False
    trunc = False
    state, info = env.reset()
    unsafe_flag = False
    if True:

        print(i_episode, ": Real data")
        tmp_buffer = []
        real_buffer = []
        
        flags = []
        
        while not done and not trunc:
            if safe_agent is not None:
                action, shielded = safe_agent(state)
                flags.append(shielded[0])
            else:
                action = agent(state)
                shielded = None

            next_state, reward, done, trunc, info = env.step(action)
                
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            

            cost = 0
            
            if env.unsafe(next_state, False):

                real_unsafe_episodes += 1 * (not unsafe_flag)
                episode_reward -= 100 * (not unsafe_flag)
                reward = -100
                print("UNSAFE (outside testing)", shielded)
                print(f"{np.round(state, 2)}", "\n", action, "\n", f"{np.round(next_state, 2)}")
                done = done or (True if safe_agent is not None else False)
                cost = 1

                unsafe_flag = True or unsafe_flag
            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py

            if cost > 0:
                agent.add(state, action, reward, next_state, done or trunc, 1)
                real_data.push(state, action, reward, next_state, done or trunc, 1)
            else:
                agent.add(state, action, reward, next_state, done or trunc, 0)
                real_data.push(state, action, reward, next_state, done or trunc, 0)
            
            
            if len(agent.memory) >= args.batch_size:
                losses = agent.train()

                writer.add_scalar(f'loss/policy_loss', losses['avg_policy_loss'], total_numsteps)
                writer.add_scalar(f'loss/value_loss', losses['avg_value_loss'], total_numsteps)
                writer.add_scalar(f'loss/entropy_loss', losses['avg_entropy_loss'], total_numsteps)
                writer.add_scalar(f'loss/clip_fraction', losses['avg_clip_fraction'], total_numsteps)
                writer.add_scalar(f'loss/kl_div', losses['avg_kl_divergence'], total_numsteps)
                writer.add_scalar(f'loss/total_loss', losses['avg_total_loss'], total_numsteps)
                writer.add_scalar(f'loss/explained_variance', losses['avg_explained_variance'], total_numsteps)


            state = next_state

            
        print("Sequence", "".join(flags))
        if safe_agent is not None:
            try:
                s, a, b, t = safe_agent.report()
                print("Shield steps:", s, "  Neural steps:", a, "  Backup steps:", b)
                print("Average time:", t / (s + a + b))
                safe_agent.reset_count()
            except Exception:
                pass
        
        total_real_episodes += 1 

    if total_numsteps >= args.start_steps * train_steps:
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
            env_model.mars.koopman_model.lr = 0.0001
            koopman_model = env_model.mars.koopman_model
            epochs = 80
        else:
            koopman_model = None
            epochs = 150
    
        env_model, ev_score, r2_score = get_environment_model(
                states, actions, next_states, rewards,
                domains.DeepPoly(env.observation_space.low, env.observation_space.high),
                seed=args.seed, koopman_model = koopman_model, latent_dim=args.red_dim, horizon = args.horizon, epochs= epochs)
        
        writer.add_scalar(f'loss/ev_koopman', ev_score, total_numsteps)   
        writer.add_scalar(f'loss/r2_score', r2_score, total_numsteps)   
            
        # new_obs_space_domain = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.koopman_model.embedding_net.embed_net, domains.DeepPoly(env.observation_space.low, env.observation_space.high)))
        
        
        # safety_domain = domains.DeepPoly(env.safety.lower, env.safety.upper)
        
        # safety = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.koopman_model.embedding_net.embed_net, safety_domain))
        
        
        
            
        safety = domains.DeepPoly(torch.cat([env.safety.lower[0], -torch.ones(args.red_dim, )]), torch.cat([env.safety.upper[0], torch.ones(args.red_dim, )]))

        

        new_obs_space = gym.spaces.Box(low=np.concatenate([np.nan_to_num(env.observation_space.low, nan=-9999, posinf=33333333, neginf=-33333333), -np.ones(args.red_dim, )]), high=np.concatenate([np.nan_to_num(env.observation_space.high, nan=-9999, posinf=33333333, neginf=-33333333), np.ones(args.red_dim, )]), shape=(args.red_dim + env.observation_space.shape[0],))
        
        polys = safety.to_hyperplanes(new_obs_space)
        print("LATENT OBS SPACE", new_obs_space)
        
        unsafe_domains = safety.invert_polytope(new_obs_space)
        env.transformed_safe_polys = polys
        # env.state_processor = env_model.mars.koopman_model.transform
        env.transformed_polys = unsafe_domains
        
        print("LATENT SAFETY", safety, len(polys[0]))
        print("LATENT UNSAFETY",  len(env.transformed_polys))
        shield = ProjectionPolicy(
            env_model.get_symbolic_model(), new_obs_space, env.observation_space,
            env.action_space, args.horizon, env.transformed_polys, env.transformed_safe_polys, env_model.mars.koopman_model.transform)
        safe_agent = Shield(shield, agent)
        
        

    # Test the agent periodically
    
    writer.add_scalar(f'reward/train', episode_reward, total_numsteps)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
          .format(i_episode, total_numsteps,
                  episode_steps, round(episode_reward, 2)))
    if safe_agent is not None:
        safe_agent.reset_count()

    if i_episode % 10 == 0:
        print("starting testing...")
        avg_reward = 0.
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
            done = False
            trunc = False
            episode_steps = 0
            trajectory = [state]
            # frames  = [env.render()]

            while not done and not trunc:
                # Decide action
                if safe_agent is not None:
                    action, shielded = safe_agent(state)
                else:
                    action = agent(state)
                    shielded = None


                next_state, reward, done, trunc, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                if episode_steps >= env._max_episode_steps:
                    done = True
                if env.unsafe(next_state, False):
                    print("UNSAFE Inside testing", shielded)
                    episode_reward += -100
                    print(f"{np.round(state, 2)}", "\n", action, "\n", f"{np.round(next_state, 2)}")
                    unsafe_episodes += 1
                    done = True

                if done and safe_agent is not None:
                    try:
                        s, a, b, t = safe_agent.report()
                        print("Finished test episode:", s, "shield and", b, "backup and", a, "neural")
                        shield_count += s
                        backup_count += b
                        neural_count += a

                        print("Average time:", t / (s + a + b))
                        safe_agent.reset_count()
                    except Exception as e:
                        print(e)
                        pass

                state = next_state
                trajectory.append(state)
                # frames.append(env.render())

            # imageio.mimsave(custom_filename, frames, fps=30)
            avg_reward += episode_reward
            avg_length += episode_steps

            avg_reward /= episodes
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
            writer.add_scalar(f'agent/unsafe_sim_episodes', unsafe_sim_episodes, total_numsteps)
            writer.add_scalar(f'agent/unsafe_sim_episodes_ratio', (unsafe_sim_episodes+0.0000001)/(total_sim_episodes+0.0000001), total_numsteps)
            writer.add_scalar(f'agent/unsafe_test_episodes', unsafe_test_episodes, total_numsteps)
            writer.add_scalar(f'agent/unsafe_test_episodes_ratio', (unsafe_test_episodes+0.0000001)/(total_test_episodes + 0.0000001), total_numsteps)
            writer.add_scalar(f'reward/test', avg_reward, total_numsteps)

            print("----------------------------------------")
            print("Test Episodes: {}, Unsafe: {}, Avg. Length: {}, Avg. Reward: {}"
                .format(episodes, unsafe_episodes, round(avg_length, 2),
                        round(avg_reward, 2)))
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