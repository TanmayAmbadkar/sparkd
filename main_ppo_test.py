import argparse
import datetime
import itertools
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO, ActorCritic
from src.policy import Shield, PPOPolicy, ProjectionPolicy
from e2c.env_model import get_environment_model
from abstract_interpretation import domains, verification
from benchmarks import envs
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
import gymnasium as gym 
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, mean_absolute_error

import traceback

parser = argparse.ArgumentParser(description='Safe PPO Args')
parser.add_argument('--env_name', default="lunar_lander")
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--mini_batch_size', type=int, default=64)
parser.add_argument('--num_steps', type=int, default=200000)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--replay_size', type=int, default=1000000)
parser.add_argument('--start_steps', type=int, default=10000)
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

file = open('logs_ppo/{}_PPO_{}_H{}_D{}.txt'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.horizon, args.red_dim), "w+")

# PPO agent setup

# Replay memories
real_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)
e2c_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)


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
while True:
    i_episode = next(iterator_loop)
    episode_reward = 0
    episode_steps = 0
    done = False
    trunc = False
    while True:
        state, info = env.reset()
        if not env.unsafe(state, False):
            break

    if (i_episode // 20) % 20 == 0:

        print(i_episode, ": Real data")
        tmp_buffer = []
        real_buffer = []
        
        while not done and not trunc:
            if safe_agent is not None:
                action = safe_agent(state)
            else:
                action = agent(state)

            next_state, reward, done, trunc, info = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            cost = 0
            if env.unsafe(next_state, False):
                real_unsafe_episodes += 1
                episode_reward -= 100
                reward = -100
                print("UNSAFE (outside testing)")
                print(f"{np.round(state, 2)}", "\n", action, "\n", f"{np.round(next_state, 2)}")
                done = True
                cost = 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py

            if cost > 0:
                agent.add(state, action, reward, next_state, done, 1)
                real_data.push(state, action, reward, next_state, done, 1)
            else:
                agent.add(state, action, reward, next_state, done, 0)
                real_data.push(state, action, reward, next_state, done, 0)
            
            state = next_state

            
        
        if safe_agent is not None:
            try:
                s, a, b, t = safe_agent.report()
                print("Shield steps:", s, "  Neural steps:", a, "  Backup steps:", b)
                print("Average time:", t / (s + a + b))
                safe_agent.reset_count()
            except Exception:
                pass
        
        total_real_episodes += 1 

    elif env_model is not None:
        
        print(i_episode, ": Simulated data")
        original_states = [state]
        predicted_states = [state]
        while not done:
            if episode_steps % 100 == 0:
                print(i_episode, episode_steps, total_numsteps)
            
            action = agent(state)  # Sample action from policy

            next_state, reward = env_model(state, action,
                                           use_neural_model=True)
            original_next_state, _, _, _, _ = env.step(action)
            original_states.append(original_next_state)
            predicted_states.append(next_state[0])
            
            done = not np.all(np.abs(next_state) < 1e5) and \
                not np.any(np.isnan(next_state))
            # done = done or env.predict_done(next_state)
            done = done or episode_steps == env._max_episode_steps or \
                not np.all(np.abs(next_state) < 1e5)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            cost = 0
            if env.unsafe(next_state, False):
                print("UNSAFE SIM", next_state)
                unsafe_sim_episodes += 1
                reward =-100
                episode_reward -= 100
                done = True
                cost = 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py

            agent.add(state.reshape(-1, ), action.reshape(-1, ), reward, next_state.reshape(-1, ), done, cost)
                    
            state = next_state.reshape(-1,)

        
        
        total_sim_episodes += 1 
        original_states = np.array(original_states)
        predicted_states = np.array(predicted_states)
        print("Original states", np.round(original_states[:10], 3))
        print("Predicted states", np.round(predicted_states[:10], 3))

        print("EV Score", explained_variance_score(original_states, predicted_states), mean_absolute_percentage_error(original_states, predicted_states), mean_absolute_error(original_states, predicted_states))
        print("EV Score 5", explained_variance_score(original_states[:5], predicted_states[:5]),  mean_absolute_percentage_error(original_states[:5], predicted_states[:5]),  mean_absolute_error(original_states[:5], predicted_states[:5]))
        print("EV Score 10", explained_variance_score(original_states[:10], predicted_states[:10]),  mean_absolute_percentage_error(original_states[:10], predicted_states[:10]),  mean_absolute_error(original_states[:10], predicted_states[:10]))
        print("EV Score 15", explained_variance_score(original_states[:15], predicted_states[:15]),  mean_absolute_percentage_error(original_states[:15], predicted_states[:15]),  mean_absolute_error(original_states[:15], predicted_states[:15]))
        print("EV Score 20", explained_variance_score(original_states[:20], predicted_states[:20]),  mean_absolute_percentage_error(original_states[:20], predicted_states[:20]),  mean_absolute_error(original_states[:20], predicted_states[:20]))

    if ((i_episode-1) // 20) % 20 == 0 and ((i_episode) // 20) % 20 != 0 and not args.no_safety:
    # if False:
        try:
            
            print("E2C DATA", len(e2c_data))
            states, actions, rewards, next_states, dones, costs = \
                real_data.sample(min(len(real_data), 70000), get_cost=True, remove_samples = True)
                
            states_e2c, actions_e2c, rewards_e2c, next_states_e2c, dones_e2c, costs_e2c = \
                e2c_data.sample(min(len(e2c_data), 30000), get_cost=True, remove_samples = False)
                
            for idx in range(states.shape[0]):
                state = states[idx]
                action = actions[idx]
                reward = rewards[idx]
                next_state = next_states[idx]
                mask = dones[idx]
                cost = costs[idx]
                e2c_data.push(state, action, reward, next_state, mask, cost)

            
            states = np.vstack([states, states_e2c])
            actions = np.vstack([actions, actions_e2c])
            rewards = np.concatenate([rewards, rewards_e2c])
            next_states = np.vstack([next_states, next_states_e2c])
            dones = np.concatenate([dones, dones_e2c])
            costs = np.concatenate([costs, costs_e2c])
            
        except Exception as e:
            
            print(traceback.format_exc())
            print("Error in sampling")
            continue
        
        if env_model is not None:
            env_model.mars.e2c_predictor.lr = 0.00003
            e2c_predictor = env_model.mars.e2c_predictor
            epochs = 50
        else:
            e2c_predictor = None
            epochs = 150
    
        env_model = get_environment_model(
                states, actions, next_states, rewards,
                domains.DeepPoly(env.original_observation_space.low, env.original_observation_space.high),
                seed=args.seed, e2c_predictor = e2c_predictor, latent_dim=args.red_dim, horizon = args.horizon, epochs= epochs)
        
            
            
            
        e2c_mean = env_model.mars.e2c_predictor.mean
        e2c_std = env_model.mars.e2c_predictor.std
        new_obs_space_domain = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, domains.DeepPoly((env.original_observation_space.low - e2c_mean)/e2c_std, (env.original_observation_space.high - e2c_mean)/e2c_std)))
        new_obs_space = gym.spaces.Box(low=new_obs_space_domain.lower[0].detach().numpy(), high=new_obs_space_domain.upper[0].detach().numpy(), shape=(args.red_dim,))
        
        safety_domain = domains.DeepPoly((env.original_safety.lower - e2c_mean)/e2c_std, (env.original_safety.upper - e2c_mean)/e2c_std)
        
        safety = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, safety_domain))
        
        
        
        unsafe_domains_list = domains.recover_safe_region(new_obs_space_domain, [safety])
            
        
        polys = safety.to_hyperplanes()

        env.transformed_safe_polys = polys
        # env.state_processor = env_model.mars.e2c_predictor.transform
        env.transformed_polys = [domain.to_hyperplanes() for domain in unsafe_domains_list]

        print("LATENT SAFETY", safety)
        print("LATENT OBS SPACE", new_obs_space_domain)

        shield = ProjectionPolicy(
            env_model.get_symbolic_model(), new_obs_space,
            env.action_space, args.horizon, env.transformed_polys, env.transformed_safe_polys, env_model.mars.e2c_predictor.transform)
        safe_agent = Shield(shield, agent)
        
        
        
