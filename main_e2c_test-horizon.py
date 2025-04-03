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
from src.policy import Shield, SACPolicy, ProjectionPolicy
from abstract_interpretation import domains, verification
import gymnasium as gym
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
import traceback
import imageio
from stable_baselines3 import SAC



parser = argparse.ArgumentParser(description='SPICE Args')
parser.add_argument('--env_name', default="lunar_lander",
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
parser.add_argument('--batch_size', type=int, default=4096, metavar='N',
                    help='batch size (default: 1024)')
parser.add_argument('--num_steps', type=int, default=10000000, metavar='N',
                    help='maximum number of steps (default: 10000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=40, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--horizon', type=int, default=5,
                    help='The safety horizon')
parser.add_argument('--neural_safety', default=False, action='store_true',
                    help='Use a neural safety signal')
parser.add_argument('--neural_threshold', type=float, default=0.1,
                    help='Safety threshold for the neural model')
parser.add_argument('--red_dim', type=int, default=8,
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

print(hyperparams)
if not os.path.exists("logs"):
    os.makedirs("logs")

# Memory
real_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)
e2c_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)

real_unsafe_episodes = 0
total_real_episodes = 0
total_numsteps = 0

update_steps = 30


iterator_loop = itertools.count(1)

agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)
# agent.policy.actor = agent.policy.actor.float()
# sac_saved_agent = SAC.load("saved_sac/SAC")

# sac_saved_agent.policy.actor = sac_saved_agent.policy.actor.float()

# print(sac_saved_agent.policy.actor)

# sac_saved_agent.predict(env.observation_space.sample(), deterministic=True)



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

                updates += 1

        next_state, reward, done, trunc, info = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        cost = 0
        if env.unsafe(next_state, False):
            real_unsafe_episodes += 1
            episode_reward -= 100
            reward+=-100
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


    if args.neural_safety:
        env_model, cost_model = get_environment_model(
                states, actions, next_states, rewards, costs,
                torch.tensor(np.concatenate([env.observation_space.low, env.action_space.low])),
                torch.tensor(np.concatenate([env.observation_space.high, env.action_space.high])),
                model_pieces=20, seed=args.seed, policy=agent,
                use_neural_model=False, cost_model=None, e2c_predictor = None, latent_dim=args.red_dim)
    else:
        env_model = get_environment_model(
                states, actions, next_states, rewards,
                domains.DeepPoly(env.observation_space.low, env.observation_space.high),
                seed=args.seed, e2c_predictor = None, latent_dim=args.red_dim, horizon = args.horizon, epochs= 100)

    
    
    e2c_mean = env_model.mars.e2c_predictor.mean
    e2c_std = env_model.mars.e2c_predictor.std
    new_obs_space = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, domains.DeepPoly((env.observation_space.low - e2c_mean)/e2c_std, (env.observation_space.high - e2c_mean)/e2c_std)))
    env.observation_space = gym.spaces.Box(low=new_obs_space.lower.detach().numpy(), high=new_obs_space.upper.detach().numpy(), shape=(args.red_dim,))
    agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)
    
    safety_domain = domains.DeepPoly((env.original_safety.lower - e2c_mean)/e2c_std, (env.original_safety.upper - e2c_mean)/e2c_std)
    
    env.safety = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, safety_domain))
    
    
    # unsafe_domains_list = [verification.get_ae_bounds(env_model.mars.e2c_predictor, unsafe_dom) for unsafe_dom in env.unsafe_domains]
    # unsafe_domains_list = [domains.DeepPoly(*unsafe_dom) for unsafe_dom in unsafe_domains_list]
    
    unsafe_domains_list = domains.recover_safe_region(new_obs_space, [env.safety])
        
    
    
    polys = [np.array(env.safety.to_hyperplanes())]

    env.safe_polys = polys
    env.state_processor = env_model.mars.e2c_predictor.transform
    env.polys = [np.array(domain.to_hyperplanes()) for domain in unsafe_domains_list]
    
    print(env.safety)
    print(env.observation_space)
    
    if args.neural_safety:
        safe_agent = CSCShield(agent, cost_model,
                                threshold=args.neural_threshold)
    else:
        shield = ProjectionPolicy(
            env_model.get_symbolic_model(), env.observation_space,
            env.action_space, args.horizon, env.polys, env.safe_polys)
        safe_agent = Shield(shield, agent)

    # Push collected training data to agent


    for (state, action, reward, next_state, mask, cost) in zip(states, actions, rewards, next_states, dones, costs):
        agent.add(env_model.mars.e2c_predictor.transform(state), action, reward, env_model.mars.e2c_predictor.transform(next_state), mask, cost)

else:
    safe_agent=None
    
    
trajectories = []
agent_actions = []
shield_actions = []
# frames  = [env.render()]
trajectories_hor = {}
true_trajectories_hor = {}
original_trajectories_hor = {}
for hor in [1, 5, 15, 30, 50]:
    safe_agent.shield.horizon = hor
    trajectories = []
    true_trajectories = []
    original_trajectories = []

    for count in range(5):
        while True:
            state, info = env.reset()
            if not env.unsafe(info['state_original'], False):
                break
        original_state = state
        pred_state = state
        true_trajectory = [state]
        trajectory = [env_model.mars.e2c_predictor.inverse_transform(state.reshape(1, -1))[0]]
        original_trajectory = [info['state_original']]
        done = False
        trunc = False
        episode_reward = 0
        episode_steps = 0

        while not done and not trunc:
            # Decide action
            action = safe_agent(state) if safe_agent is not None else agent(state)

            next_state, reward, done, trunc, info = env.step(action)
            next_state_pred, _ = env_model(pred_state, action, use_neural_model=False)
            episode_reward += reward
            episode_steps += 1
             

            if env.unsafe(info['state_original'], False):
                print("UNSAFE")
                episode_reward += -100
                print(np.round(original_state, 2), "\n", action, "\n", np.round(info['state_original'], 2))
                true_trajectory.append(next_state)
                trajectory.append(env_model.mars.e2c_predictor.inverse_transform(next_state_pred.reshape(1, -1))[0])
                original_trajectory.append(info['state_original'])
                
                break

            state = next_state
            true_trajectory.append(next_state)
            trajectory.append(env_model.mars.e2c_predictor.inverse_transform(next_state_pred.reshape(1, -1))[0])
            original_trajectory.append(info['state_original'])
            original_state = info['state_original']
            pred_state = next_state_pred
            
            
            if (episode_steps + 1) % hor == 0:
                pred_state = state
            # frames.append(env.render())
        print(f"HOR {hor}, {env.unsafe(info['state_original'], False)} {done} {trunc}")
        # print(trajectory)
        # print(true_trajectory)
        trajectories.append(np.vstack(trajectory))
        true_trajectories.append(np.vstack(true_trajectory))
        original_trajectories.append(np.vstack(original_trajectory))
    print("DONE WITH SCALLE, ", hor)
    trajectories_hor[hor] = trajectories
    true_trajectories_hor[hor] = true_trajectories
    original_trajectories_hor[hor] = original_trajectories

from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score

for hor in trajectories_hor:
    print(hor, "ev_score:", explained_variance_score(np.vstack(original_trajectories_hor[hor]), np.vstack(trajectories_hor[hor])))
    print(hor, "mae:", mean_absolute_error(np.vstack(original_trajectories_hor[hor]), np.vstack(trajectories_hor[hor])))
    # print(hor, "rmse:", root_mean_squared_error(np.concatenate(true_trajectories_hor[hor]), np.concatenate(trajectories_hor[hor])))


    print("TRUE", np.round(np.vstack(original_trajectories_hor[hor])[:5], 2))
    print("PRED", np.round(np.vstack(trajectories_hor[hor])[:5], 2) )
    

# print([len(trajectories_hor[hor]) for hor in trajectories_hor])
            
# # states, actions, rewards, next_states, dones, costs = agent.memory.sample(len(agent.memory))

# import matplotlib.pyplot as plt

# color = {1:"red", 5:"blue", 10:"green", 15:"orange", 20:"black"}
# # plt.plot(trajectories[0][:, 0], trajectories[0][:, 1], label="SCALLE", color="blue")
# for hor in original_trajectories_hor:
    
#     plt.plot(original_trajectories_hor[hor][0][:, 0], original_trajectories_hor[hor][0][:, 1], label = hor, color=color[hor])
#     for trajectory in original_trajectories_hor[hor][1:3]:
#         plt.plot(trajectory[:, 0], trajectory[:, 1], color=color[hor])
#     # plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")
    
# # plt.plot(trajectory_sac[:, 0], trajectory_sac[:, 1], label="SAC")
# plt.plot([-1, -1], [-0.1, 2], color="green")
# plt.plot([-1, 1], [-0.1, -0.1], color="green")
# plt.plot([-1, 1], [2, 2], color="green")
# plt.plot([1, 1], [-0.1, 2], color="green")
# # plt.xlim(-2.5, 2.5)
# # plt.ylim(-2.5, 2.5)
# plt.legend()
# plt.savefig("QualAnalysis.png")

# plt.figure()


# # plt.plot(trajectories[0][:, 0], trajectories[0][:, 1], label="SCALLE", color="blue")
# for hor in original_trajectories_hor:
    
#     plt.plot(original_trajectories_hor[hor][0][:, 2], original_trajectories_hor[hor][0][:, 3], label = hor, color=color[hor])
#     for trajectory in original_trajectories_hor[hor][1:3]:
#         plt.plot(trajectory[:, 2], trajectory[:, 3], color=color[hor])
#     # plt.plot(trajectory[:, 0], trajectory[:, 1], color="blue")
    
# # plt.plot(trajectory_sac[:, 2], trajectory_sac[:, 3], label="SAC")
# plt.plot([-1.5, -1.5], [-1.5, 1.5], color="green")
# plt.plot([-1.5, 1.5], [-1.5, -1.5], color="green")
# plt.plot([-1.5, 1.5], [1, 1], color="green")
# plt.plot([1.5, 1.5], [-1.5, 1.5], color="green")
# # plt.xlim(-10, 10)
# # plt.ylim(-10, 10)
# plt.legend()
# plt.savefig("QualAnalysis2.png")


