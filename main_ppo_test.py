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

import traceback

parser = argparse.ArgumentParser(description='Safe PPO Args')
parser.add_argument('--env_name', default="lunar_lander")
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=2048)
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

while total_numsteps < args.start_steps:
    i_episode = next(iterator_loop)
    state, _ = env.reset()
    episode_steps = 0
    episode_reward = 0
    print(i_episode, ": Real data")
    real_buffer = []
    done = False
    trunc = False
    
    while not done and not trunc:

        action = env.action_space.sample()
        next_state, reward, done, trunc, _ = env.step(action)
        cost = int(env.unsafe(next_state, False))
        state = next_state
        total_numsteps += 1

        if env.unsafe(next_state, False):
            real_unsafe_episodes += 1
            episode_reward -= 10
            reward = -10
            print("UNSAFE (outside testing)", np.round(next_state, 2))
            done = True
            cost = 1


        episode_steps += 1
        episode_reward += reward
        
        real_data.push(state, action, reward, next_state, done, cost)

    
    
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
          .format(i_episode, total_numsteps,
                  episode_steps, round(episode_reward, 2)))
    
    
    total_real_episodes += 1


# Creation of environment model
if not args.no_safety:
    states, actions, rewards, next_states, dones, costs = \
        real_data.sample(args.start_steps, get_cost=True, remove_samples = True)
    
    for idx in range(states.shape[0]):
        state = states[idx]
        action = actions[idx]
        reward = rewards[idx]
        next_state = next_states[idx]
        mask = dones[idx]
        cost = costs[idx]
        e2c_data.push(state, action, reward, next_state, mask, cost)

    # print("E2C DATA", len(e2c_data))

    states, actions, rewards, next_states, dones, costs = \
        e2c_data.sample(args.start_steps, get_cost=True, remove_samples = False)

    env_model = get_environment_model(
            states, actions, next_states, rewards,
            domains.DeepPoly(env.observation_space.low, env.observation_space.high), seed=args.seed, e2c_predictor = None, latent_dim=args.red_dim, horizon = args.horizon, epochs= 80)

    
    
    e2c_mean = env_model.mars.e2c_predictor.mean
    e2c_std = env_model.mars.e2c_predictor.std
    new_obs_space = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, domains.DeepPoly((env.observation_space.low - e2c_mean)/e2c_std, (env.observation_space.high - e2c_mean)/e2c_std)))
    env.observation_space = gym.spaces.Box(low=new_obs_space.lower.detach().numpy(), high=new_obs_space.upper.detach().numpy(), shape=(args.red_dim,))
    
    safety_domain = domains.DeepPoly((env.original_safety.lower - e2c_mean)/e2c_std, (env.original_safety.upper - e2c_mean)/e2c_std)
    
    env.safety = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, safety_domain))
    
    
    unsafe_domains_list = domains.recover_safe_region(new_obs_space, [env.safety])
        
    
    
    polys = [np.array(env.safety.to_hyperplanes())]

    env.safe_polys = polys
    env.state_processor = env_model.mars.e2c_predictor.transform
    env.polys = [np.array(domain.to_hyperplanes()) for domain in unsafe_domains_list]
    
    print(env.safety)
    print(env.observation_space)
    print(domains.DeepPoly(*verification.get_variational_bounds(env_model.mars.e2c_predictor, safety_domain)))
    func = lambda x: verification.get_constraints(env_model.mars.e2c_predictor.encoder.fc_mu, verification.get_constraints(env_model.mars.e2c_predictor.encoder.shared_net, x))
    print(func(safety_domain))
    
    agent = PPOPolicy(env, args.batch_size, args.seed, args.batch_size, args)
    shield = ProjectionPolicy(
        env_model.get_symbolic_model(), env.observation_space,
        env.action_space, args.horizon, env.polys, env.safe_polys)
    safe_agent = Shield(shield, agent)

    # Push collected training data to agent


    print(env_model.mars.e2c_predictor.transform(env.original_safety.lower))
    print(env_model.mars.e2c_predictor.transform(env.original_safety.upper))