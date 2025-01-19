import numpy as np
np.int = np.int64
np.float = np.float64

import argparse
import datetime
import itertools
import torch
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from pytorch_soft_actor_critic.replay_memory import ReplayMemory

from benchmarks import envs
from e2c.env_model import get_environment_model
from src.policy import Shield, SACPolicy, ProjectionPolicy, CSCShield
from abstract_interpretation import domains, verification
import gymnasium as gym
from sklearn.metrics import classification_report, r2_score
import itertools
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='SPICE Args')
parser.add_argument('--env_name', default="lunar_lander_R",
                    help='Environment (default: acc)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every few episodes (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient (tau) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter alpha determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', default=False, action='store_true',
                    help='Automaically adjust alpha (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size (default: 1024)')
parser.add_argument('--num_steps', type=int, default=10000000, metavar='N',
                    help='maximum number of steps (default: 10000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=40, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--horizon', type=int, default=5,
                    help='The safety horizon')
parser.add_argument('--neural_safety', default=False, action='store_true',
                    help='Use a neural safety signal')
parser.add_argument('--neural_threshold', type=float, default=0.1,
                    help='Safety threshold for the neural model')
parser.add_argument('--red_dim', type=int, default=4,
                    help='Reduced dimension size')
parser.add_argument('--no_safety', default=False, action='store_true',
                    help='To use safety or no safety')
args = parser.parse_args()

print("Arguments:")
print(args)
hyperparams = vars(args)


env = envs.get_env_from_name(args.env_name)
env.seed(args.seed)


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
# agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)
# safe_agent = None

# Tensorboard
writer = SummaryWriter('runs/temp/{}_SAC_{}_{}_{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.policy, "autotune" if args.automatic_entropy_tuning else ""))

print(hyperparams)
if not os.path.exists("logs"):
    os.makedirs("logs")

file = open('logs/temp/{}_SAC_{}_{}_{}.txt'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.policy, "autotune" if args.automatic_entropy_tuning else ""), "w+")

# Memory
real_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)
e2c_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)

real_unsafe_episodes = 0
total_real_episodes = 0
total_numsteps = 0

update_steps = 10


iterator_loop = itertools.count(1)

agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)


# PRETRAINING 
original_obs_space = env.observation_space
updates = 0
while total_numsteps < args.start_steps:
    
    i_episode = next(iterator_loop)
    state, _ = env.reset()
    episode_steps = 0
    episode_reward = 0
    print(i_episode, ": Real data")
    real_buffer = []
    done = False
    trunc = False
    while not done and not trunc :
        
        if not args.no_safety:
            action = env.action_space.sample()
        else:
            action = agent(state)
        # if len(agent.memory) > args.batch_size and args.no_safety:
        if total_numsteps % update_steps == 0  and len(agent.memory) > args.batch_size and args.no_safety:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_l, alph = \
                    agent.train()

                writer.add_scalar(f'loss/critic_1', critic_1_loss, updates)
                writer.add_scalar(f'loss/critic_2', critic_2_loss, updates)
                writer.add_scalar(f'loss/policy', policy_loss, updates)
                writer.add_scalar(f'loss/entropy_loss', ent_l, updates)
                writer.add_scalar(f'loss/alpha', alph, updates)
                updates += 1

        next_state, reward, done, trunc, info = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        cost = 0
        if env.unsafe(next_state, False):
            real_unsafe_episodes += 1
            episode_reward -= 10
            reward+=-10
            print("UNSAFE (outside testing)", np.round(next_state, 2))
            done = True
            cost = 1

        real_buffer.append((state, action, reward, next_state, done,
            cost))


        state = next_state

    
    for (state, action, reward, next_state, mask, cost) in real_buffer:
        if cost > 0:
            real_data.push(state, action, reward, next_state, mask, 1)
        else:
            real_data.push(state, action, reward, next_state, mask, 0)
            
    for (state, action, reward, next_state, mask, cost) in real_buffer:
        if cost > 0:
            agent.add(state, action, reward, next_state, mask, 1)
        else:
            agent.add(state, action, reward, next_state, mask, 0)

    # total_episodes += 1 
    
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
          .format(i_episode, total_numsteps,
                  episode_steps, round(episode_reward, 2)))
    
    total_real_episodes += 1

if not args.no_safety:     
    states, actions, rewards, next_states, dones, costs = \
        real_data.sample(len(real_data), get_cost=True, remove_samples = True)

    for (state, action, reward, next_state, mask, cost) in zip(states, actions, rewards, next_states, dones, costs):
        e2c_data.push(state, action, reward, next_state, mask, cost)


    if args.neural_safety:
        env_model, cost_model = get_environment_model(
                states, actions, next_states, rewards, costs,
                torch.tensor(np.concatenate([env.observation_space.low, env.action_space.low])),
                torch.tensor(np.concatenate([env.observation_space.high, env.action_space.high])),
                model_pieces=20, seed=args.seed, policy=agent,
                use_neural_model=False, cost_model=None, e2c_predictor = None, latent_dim=args.red_dim)
    else:
        env_model, cost_model = get_environment_model(
                states, actions, next_states, rewards, costs,
                domains.DeepPoly(env.observation_space.low, env.observation_space.high),
                model_pieces=20, seed=args.seed, policy=None,
                use_neural_model=False, cost_model=None, e2c_predictor = None, latent_dim=args.red_dim, horizon = args.horizon)

        
    new_obs_space = domains.DeepPoly(*verification.get_variational_bounds(env_model.mars.e2c_predictor, domains.DeepPoly(env.observation_space.low, env.observation_space.high)))
    env.observation_space = gym.spaces.Box(low=new_obs_space.lower.detach().numpy(), high=new_obs_space.upper.detach().numpy(), shape=(args.red_dim,))
    agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)
    
    env.safety = domains.DeepPoly(*verification.get_variational_bounds(env_model.mars.e2c_predictor, env.original_safety))
    
    
    unsafe_domains_list = [verification.get_variational_bounds(env_model.mars.e2c_predictor, unsafe_dom) for unsafe_dom in env.unsafe_domains]
    unsafe_domains_list = [domains.DeepPoly(*unsafe_dom) for unsafe_dom in unsafe_domains_list]
        
    
    domain = verification.get_constraints(env_model.mars.e2c_predictor.encoder.shared_net, domains.DeepPoly(env.original_observation_space.low, env.original_observation_space.high))
    mu_domain = verification.get_constraints(env_model.mars.e2c_predictor.encoder.fc_mu, domain)
    recovered_dom = verification.get_constraints(env_model.mars.e2c_predictor.decoder.net, mu_domain)
    print("Recovered domain", recovered_dom.calculate_bounds())
    print("Original domain",env.original_observation_space)
    
    
    print(unsafe_domains_list)
    print("SAFETY: ", env.safety)
    print("OBS SPACE: ", env.observation_space)
    
    polys = [np.array(env.safety.to_hyperplanes())]

    env.safe_polys = polys
    env.state_processor = env_model.mars.e2c_predictor.transform
    env.polys = [np.array(domain.to_hyperplanes()) for domain in unsafe_domains_list]
    
    if args.neural_safety:
        safe_agent = CSCShield(agent, cost_model,
                                threshold=args.neural_threshold)
    else:
        shield = ProjectionPolicy(
            env_model.get_symbolic_model(), env.observation_space,
            env.action_space, args.horizon, env.polys, env.safe_polys)
        safe_agent = Shield(shield, agent)

    # Push collected training data to agent


    # for (state, action, rewards, next_state, mask, cost) in zip(states, actions, rewards, next_states, dones, costs):
    #     agent.add(env_model.mars.e2c_predictor.transform(state[0]), action[0], rewards[0], env_model.mars.e2c_predictor.transform(next_state[0]), mask[0], cost[0])

else:
    safe_agent=None
    
    
next_states = []
predictions = []

rewards_true = []
rewards_pred = []

# Test env loop
for i in range(10):
    state, info = env.reset()
    done = False
    trunc = False

    while not done and not trunc:
        
        action = env.action_space.sample()
        
        next_state, reward_true, done, trunc, info = env.step(action)
        
        
        next_state_pred, reward_pred= env_model(state, action, use_neural_model=False)
        
        if i == 1:
            print("Next state", next_state)
            print("Next state pred", next_state_pred)
            # print("Reward true", reward_true)
            # print("Reward pred", reward_pred)
            # print(f"True state {np.round(info['state_original'], 3)}")
            # print(f"Predicted state {np.round(env_model.mars.e2c_predictor.inverse_transform(next_state), 3)}")
        
        state = next_state_pred
        
        predictions.append(next_state_pred)
        next_states.append(next_state)
        
        # if i == 1:
            # print("TRUE UNSAFE", env.unsafe(info['state_original'], False))
            # print("Predicted UNSAFE", env.unsafe(next_state_pred, True))
        
        if env.unsafe(info['state_original'], False):
            print("UNSAFE")
            break
        
        rewards_pred.append(reward_pred)
        rewards_true.append(reward_true)

predictions = np.array(predictions)
next_states = np.array(next_states)
rewards_pred = np.array(rewards_pred)
rewards_true = np.array(rewards_true)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(np.vstack([next_states, predictions.reshape(-1, args.red_dim)]))
# X_2d = X_2d[:len(next_states)]
plt.scatter(X_2d[:len(next_states), 0], X_2d[:len(next_states), 1], c='r', label='True')
plt.savefig("scatter_e2c.png")


tsne = TSNE(n_components=2, random_state=0)
# X_2d = tsne.fit_transform(predictions.reshape(-1, args.red_dim))
# X_2d = tsne.transform(predictions.reshape(-1, args.red_dim))
plt.scatter(X_2d[len(next_states):, 0], X_2d[len(next_states):, 1], c='g', label='Pred')
plt.savefig("scatter_e2c_pred.png")

print("MSE, MAE for states", np.mean((predictions - next_states)**2), np.mean(np.abs(predictions - next_states)))
print("MSE, MAE for rewards", np.mean((rewards_pred - rewards_true)**2), np.mean(np.abs(rewards_pred - rewards_true)))

print("r2 score", r2_score(predictions.reshape(predictions.shape[0], -1), next_states.reshape(next_states.shape[0], -1)))
        

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(next_states.shape)
print(predictions.shape)
predictions = predictions.reshape(-1, 3)
ax.scatter(next_states[:, 0], next_states[:, 1], next_states[:, 2], c='g', label="True")
ax.scatter(predictions[:, 0], next_states[:, 1], predictions[:, 2], c='r', label="Pred")
plt.savefig("scatter_e2c_3d.png")