import numpy as np
np.int = np.int64
np.float = np.float64

import argparse
import datetime
import itertools
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from pytorch_soft_actor_critic.replay_memory import ReplayMemory

from benchmarks import envs
from src.env_model import get_environment_model
from src.policy import Shield, SACPolicy, ProjectionPolicy, CSCShield
from abstract_interpretation import domains, verification
from encoder.autoencoder import Autoencoder, fit_encoder
import gymnasium as gym


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
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10000000, metavar='N',
                    help='maximum number of steps (default: 10000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
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
parser.add_argument('--autoencoder', type=int, default=0,
                    help='Timestep to switch to autoencoder')
parser.add_argument('--red_dim', type=int, default=4,
                    help='Reduced dimension size')
args = parser.parse_args()

print("Arguments:")
print(args)

env = envs.get_env_from_name(args.env_name)
env.seed(args.seed)


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)
safe_agent = None

# Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.policy, "autotune" if args.automatic_entropy_tuning else ""))


file = open('logs/{}_SAC_{}_{}_{}.txt'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.policy, "autotune" if args.automatic_entropy_tuning else ""), "w+")

# Memory
real_data = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

total_unsafe_episodes = 0
total_episodes = 0

cost_model = None

iterator_loop = itertools.count(1)
while True:
    i_episode = next(iterator_loop)
    total_episodes = i_episode
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    if (i_episode // 10) % 8 == 0:
        print(i_episode, ": Real data")
        tmp_buffer = []
        real_buffer = []
        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                if safe_agent is not None:
                    action = safe_agent(state)
                else:
                    action = agent(state)

            if len(agent.memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_l, alph = \
                        agent.train()

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_l, updates)
                    writer.add_scalar('entropy_temprature/alpha', alph,
                                      updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            cost = 0
            if env.unsafe(next_state):
                total_unsafe_episodes += 1
                episode_reward -= 1000
                print("UNSAFE (outside testing)", next_state)
                done = True
                cost = 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py
            mask = 1 if episode_steps == env._max_episode_steps \
                else float(not done)

            tmp_buffer.append((state, action, reward, next_state, mask, cost))

            if env.unsafe(next_state):
                episode_reward -= 10000



            # Don't add states to the training data if they hit the edge of
            # the state space, this seems to cause problems for the regression.
            if not (np.any(next_state < env.observation_space.low) or
                    np.any(next_state > env.observation_space.high)):
                    real_buffer.append((state, action, reward, next_state, mask,
                                    cost))


            state = next_state


        for (state, action, reward, next_state, mask, _) in tmp_buffer:
            if cost > 0:
                agent.add(state, action, reward, next_state, mask, 1)
            else:
                agent.add(state, action, reward, next_state, mask, 0)

        for (state, action, rewards, next_state, mask, _) in real_buffer:
            if cost > 0:
                real_data.push(state, action, reward, next_state, mask, 1)
            else:
                real_data.push(state, action, reward, next_state, mask, 0)
        if safe_agent is not None:
            try:
                s, a, t = safe_agent.report()
                print("Shield steps:", s, "  Neural steps:", a)
                print("Average time:", t / (s + a))
                safe_agent.reset_count()
            except Exception:
                pass

    else:
        print(i_episode, ": Simulated data")
        while not done:
            if episode_steps % 100 == 0:
                print(i_episode, episode_steps, total_numsteps)
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent(state)  # Sample action from policy

            if len(agent.memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_l, alph = \
                        agent.train()

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_l, updates)
                    writer.add_scalar('entropy_temprature/alpha', alph,
                                      updates)
                    updates += 1

            next_state, reward = env_model(state, action,
                                           use_neural_model=False)
            done = not np.all(np.abs(next_state) < 1e5) and \
                not np.any(np.isnan(next_state))
            done = done or env.predict_done(next_state)
            done = done or episode_steps == env._max_episode_steps or \
                not np.all(np.abs(next_state) < 1e5)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            cost = 0
            if env.unsafe(next_state):
                total_unsafe_episodes += 1
                episode_reward -= 1000
                done = True
                cost = 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py
            mask = 1 if episode_steps == env._max_episode_steps \
                else float(not done)

            agent.add(state, action, reward, next_state, mask, cost)

            state = next_state

    if (i_episode - 9) % 100 == 0:
        try:
            states, actions, rewards, next_states, dones, costs = \
                real_data.sample(min(len(real_data), 50000), get_cost=True)
        except:
            continue
        if args.neural_safety:
            env_model, cost_model = get_environment_model(
                    states, actions, next_states, rewards, costs,
                    torch.tensor(env.observation_space.low),
                    torch.tensor(env.observation_space.high),
                    model_pieces=20, seed=args.seed, policy=agent,
                    use_neural_model=False, cost_model=cost_model)
        else:
            env_model, cost_model = get_environment_model(
                    states, actions, next_states, rewards, costs,
                    torch.tensor(env.observation_space.low),
                    torch.tensor(env.observation_space.high),
                    model_pieces=20, seed=args.seed, policy=None,
                    use_neural_model=False, cost_model=None)

        if args.neural_safety:
            safe_agent = CSCShield(agent, cost_model,
                                   threshold=args.neural_threshold)
        else:
            shield = ProjectionPolicy(
                env_model.get_symbolic_model(), env.observation_space,
                env.action_space, args.horizon, env.polys, env.safe_polys)
            safe_agent = Shield(shield, agent)

    if total_numsteps > args.autoencoder and env.state_processor is None:
        # Write code to include autoencoder, and switch agent and environments. 
        states, _, _, _, _, _ = \
            real_data.sample(min(len(real_data), args.autoencoder), get_cost=True)

        encoder = Autoencoder(env.observation_space.shape[0], args.red_dim)
        fit_encoder(states, encoder)
        
        env.state_processor = encoder.transform
        env.safety = verification.get_constraints(encoder.encoder, env.safety)
        env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(args.red_dim,))
        agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)
        safe_agent = None
        real_data = ReplayMemory(args.replay_size, args.seed)
        iterator_loop = itertools.count(1)
        total_numsteps = 0
        cost_model = None
        print("Total unsafe without AE:", total_unsafe_episodes, "/", total_episodes, file=file)
        total_unsafe_episodes = 0
        total_episodes = 0
        continue
        
    
    if total_numsteps > args.num_steps/2:
        break
    
        

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
          .format(i_episode, total_numsteps,
                  episode_steps, round(episode_reward, 2)))

    if (i_episode - 99) % 1 == 0 and args.eval is True and \
            safe_agent is not None:
        print("starting testing...")
        avg_reward = 0.
        episodes = 1
        unsafe_episodes = 0
        avg_length = 0.
        shield_count = 0
        neural_count = 0
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            trajectory = [state]
            while not done:
                action = safe_agent(state)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1

                if episode_steps >= env._max_episode_steps:
                    done = True
                if env.unsafe(next_state):
                    print("UNSAFE")
                    print(state, "\n",  action, "\n", next_state)
                    unsafe_episodes += 1
                    done = True
                if done:
                    try:
                        s, a, t = safe_agent.report()
                        print("Finished test episode:", s, "shield and", a,
                              "neural")
                        shield_count+=s
                        neural_count+=a
                       
                        print("Average time:", t / (s + a))
                        safe_agent.reset_count()
                    except Exception:
                        pass
                    break
                state = next_state
                trajectory.append(state)
            avg_reward += episode_reward
            avg_length += episode_steps
        avg_reward /= episodes
        avg_length /= episodes
        shield_count /= episodes
        neural_count /= episodes
        writer.add_scalar('agent/shield', shield_count, updates)
        writer.add_scalar('agent/neural', neural_count, updates)

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Unsafe: {}, Avg. Length: {}, Avg. Reward: {}"
              .format(episodes, unsafe_episodes, round(avg_length, 2),
                      round(avg_reward, 2)))
        print("----------------------------------------")
        if (i_episode - 99) % 100 == 0:
            print("Trajectory:")
            print(trajectory)

print("Total unsafe:", total_unsafe_episodes, "/", total_episodes)
print("Total unsafe:", total_unsafe_episodes, "/", total_episodes, file=file)
