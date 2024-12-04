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
from sklearn.metrics import classification_report
import itertools



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
parser.add_argument('--lr', type=float, default=0.00001, metavar='G',
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
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.policy, "autotune" if args.automatic_entropy_tuning else ""))

print(hyperparams)
if not os.path.exists("logs"):
    os.makedirs("logs")

file = open('logs/{}_SAC_{}_{}_{}.txt'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    args.policy, "autotune" if args.automatic_entropy_tuning else ""), "w+")

# Memory
real_data = ReplayMemory(args.replay_size, env.observation_space, args.seed)
e2c_data = ReplayMemory(args.replay_size, env.observation_space, args.seed)

real_unsafe_episodes = 0
total_real_episodes = 0
total_numsteps = 0

update_steps = 10


iterator_loop = itertools.count(1)

agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)


# PRETRAINING 

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
            print("UNSAFE (outside testing)", np.round(next_state, 2))
            done = True
            cost = 1

        real_buffer.append((state, action, reward, next_state, done,
            cost))


        state = next_state


    for (state, action, rewards, next_state, mask, cost) in real_buffer:
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
        real_data.sample(args.start_steps, get_cost=True, remove_samples = True)

    for (state, action, rewards, next_state, mask, cost) in zip(states, actions, rewards, next_states, dones, costs):
        e2c_data.push(state, action, rewards, next_state, mask, cost)

    # print("E2C DATA", len(e2c_data))

    states, actions, rewards, next_states, dones, costs = \
        e2c_data.sample(args.start_steps, get_cost=True)
        
    env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(args.red_dim,))
    agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)


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
                torch.tensor(np.concatenate([env.observation_space.low, env.action_space.low])),
                torch.tensor(np.concatenate([env.observation_space.high, env.action_space.high])),
                model_pieces=20, seed=args.seed, policy=None,
                use_neural_model=False, cost_model=None, e2c_predictor = None, latent_dim=args.red_dim, horizon = args.horizon)

    
    # new_unsafe_domains = []
    # for unsafe_domain in env.unsafe_domains:
    #     new_unsafe_domains.append(verification.get_constraints(env_model.mars.e2c_predictor.encoder.net, unsafe_domain))
    
    # print(new_unsafe_domains)
    
    
    # env.safety = verification.get_constraints(env_model.mars.e2c_predictor.encoder.net, env.safety)
    # recovered_safety = verification.get_constraints(env_model.mars.e2c_predictor.decoder.net, env.safety)
    # env.unsafe_zonotope = verification.get_constraints(encoder.encoder, env.unsafe_zonotope)
    
    
    new_obs_space = verification.get_constraints(env_model.mars.e2c_predictor.encoder.net, domains.DeepPoly(-np.ones(60, ), np.ones(60, )))
    
    env.observation_space = gym.spaces.Box(low=new_obs_space.lower.detach().numpy(), high=new_obs_space.upper.detach().numpy(), shape=(args.red_dim,))
    # print(new_unsafe_domains)
    # print(domains.recover_safe_region(domains.DeepPoly(env.observation_space.low, env.observation_space.high), new_unsafe_domains))
    env.safety = verification.get_constraints(env_model.mars.e2c_predictor.encoder.net, env.original_safety)
    
    
    new_unsafe_domains = domains.recover_safe_region(domains.DeepPoly(env.observation_space.low, env.observation_space.high), [env.safety])
    
    agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)

    polys = env.safety.to_hyperplanes()

    env.safe_polys = [np.array(polys)]
    env.state_processor = env_model.mars.e2c_predictor.transform
    env.polys = [np.array(domain.to_hyperplanes()) for domain in new_unsafe_domains]


    if args.neural_safety:
        safe_agent = CSCShield(agent, cost_model,
                                threshold=args.neural_threshold)
    else:
        shield = ProjectionPolicy(
            env_model.get_symbolic_model(), env.observation_space,
            env.action_space, args.horizon, env.polys, env.safe_polys)
        safe_agent = Shield(shield, agent)

    # Push collected training data to agent


    for (state, action, rewards, next_state, mask, cost) in zip(states, actions, rewards, next_states, dones, costs):
        agent.add(env_model.mars.e2c_predictor.transform(state), action, rewards, env_model.mars.e2c_predictor.transform(next_state), mask, cost)

else:
    safe_agent=None
# Start main training

# VERIFICATION
truth = []
predicted = []
for state in next_states:
    truth.append(env.unsafe(state, False))
    predicted.append(env.unsafe(env_model.mars.e2c_predictor.transform(state), simulated = True))
print(classification_report(truth, predicted))

truth = []
predicted = []
for i in range(1000):
    point1 = np.random.uniform(high = env.safety.lower.numpy(), low = -np.ones(env.safety.lower.shape))
    point2 = np.random.uniform(low = env.safety.upper.numpy(), high = np.ones(env.safety.lower.shape))
    predicted.append(env.unsafe(point1, simulated = True))
    predicted.append(env.unsafe(point2, simulated = True))
    truth.append(env.unsafe(env_model.mars.e2c_predictor.inverse_transform(point1)))
    truth.append(env.unsafe(env_model.mars.e2c_predictor.inverse_transform(point2)))

print(classification_report(truth, predicted))

    


print(sum(truth))

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
    state, info = env.reset()

    if (i_episode // 10) % 8 == 0 or args.no_safety:
        print(i_episode, ": Real data")
        tmp_buffer = []
        real_buffer = []
        
        last_state = info['state_original']
        while not done and not trunc:
            if safe_agent is not None:
                action = safe_agent(state)
            else:
                action = agent(state)

            # if len(agent.memory) > args.batch_size:
            if total_numsteps % update_steps == 0 and len(agent.memory) > args.batch_size:
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
            if env.unsafe(info['state_original'], False):
                real_unsafe_episodes += 1
                episode_reward -= 10
                print("UNSAFE (outside testing)", next_state)
                done = True
                cost = 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py
            mask = 1 if episode_steps == env._max_episode_steps \
                else float(not done)

            tmp_buffer.append((state, action, reward, next_state, mask, cost))




            # Don't add states to the training data if they hit the edge of
            # the state space, this seems to cause problems for the regression.
            
            real_buffer.append((last_state, action, reward, info['state_original'], mask,
                            cost))


            state = next_state
            last_state = info['state_original']


        for (state, action, reward, next_state, mask, cost) in tmp_buffer:
            if cost > 0:
                agent.add(state, action, reward, next_state, mask, 1)
            else:
                agent.add(state, action, reward, next_state, mask, 0)

        for (state, action, rewards, next_state, mask, cost) in real_buffer:
            if cost > 0:
                real_data.push(state, action, reward, next_state, mask, 1)
            else:
                real_data.push(state, action, reward, next_state, mask, 0)
        if safe_agent is not None:
            try:
                s, a, b, t = safe_agent.report()
                print("Shield steps:", s, "  Neural steps:", a, "  Backup steps:", b)
                print("Average time:", t / (s + a + b))
                safe_agent.reset_count()
            except Exception:
                pass
        
        total_real_episodes += 1 

    elif not args.no_safety:
        print(i_episode, ": Simulated data")
        while not done:
            if episode_steps % 100 == 0:
                print(i_episode, episode_steps, total_numsteps)
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent(state)  # Sample action from policy
            # if len(agent.memory) > args.batch_size:
            if total_numsteps % update_steps == 0  and len(agent.memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_l, alph = \
                        agent.train()

                    writer.add_scalar(f'loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar(f'loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar(f'loss/policy', policy_loss, updates)
                    writer.add_scalar(f'loss/entropy_loss', ent_l, updates)
                    writer.add_scalar(f'entropy_temprature/alpha', alph,
                                      updates)
                    updates += 1

            next_state, reward = env_model(state, action,
                                           use_neural_model=False)
            done = not np.all(np.abs(next_state) < 1e5) and \
                not np.any(np.isnan(next_state))
            # done = done or env.predict_done(next_state)
            done = done or episode_steps == env._max_episode_steps or \
                not np.all(np.abs(next_state) < 1e5)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            cost = 0
            if env.unsafe(next_state, True):
                print(next_state)
                unsafe_sim_episodes += 1
                episode_reward -= 10
                done = True
                cost = 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py
            mask = 1 if episode_steps == env._max_episode_steps \
                else float(not done)

            agent.add(state.reshape(-1, ), action.reshape(-1, ), reward, next_state.reshape(-1, ), mask, cost)

            state = next_state
        
        total_sim_episodes += 1 

    if (i_episode - 9) % 100 == 0 and not args.no_safety:
        try:
            
            
            states, actions, rewards, next_states, dones, costs = \
                real_data.sample(min(len(real_data), 70000), get_cost=True, removes_samples = True)
                
            states_e2c, actions_e2c, rewards_e2c, next_states_e2c, dones_e2c, costs_e2c = \
                e2c_data.sample(min(len(real_data), 30000), get_cost=True, removes_samples = True)
                
            
            for (state, action, rewards, next_state, mask, cost) in zip(states, actions, rewards, next_states, dones, costs):
                e2c_data.push(state, action, rewards, next_state, mask, cost)
                
            states = np.vstack([states, states_e2c])
            actions = np.vstack([actions, actions_e2c])
            rewards = np.vstack([rewards, rewards_e2c])
            next_states = np.vstack([next_states, next_states_e2c])
            dones = np.vstack([dones, dones_e2c])
            costs = np.vstack([costs, costs_e2c])
            
        except:
            continue
        
        env_model.mars.e2c_predictor.lr = 0.00001
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
                    torch.tensor(np.concatenate([env.observation_space.low, env.action_space.low])),
                    torch.tensor(np.concatenate([env.observation_space.high, env.action_space.high])),
                    model_pieces=20, seed=args.seed, policy=None,
                    use_neural_model=False, cost_model=None, e2c_predictor = env_model.mars.e2c_predictor, latent_dim=args.red_dim, horizon = args.horizon)
    
        env.safety = verification.get_constraints(env_model.mars.e2c_predictor.encoder.net, env.original_safety)
        # env.unsafe_zonotope = verification.get_constraints(encoder.encoder, env.unsafe_zonotope)
        
        env.state_processor = env_model.mars.e2c_predictor.transform

        # hyperplanes = env.safety.to_hyperplanes()
        # polys = []
        # for A, b in hyperplanes:
        #     polys.append(np.append(A, -b))

        # env.safe_polys = [np.array(polys)]
        
        # env.unsafe_constraints()
        new_obs_space = verification.get_constraints(env_model.mars.e2c_predictor.encoder.net, domains.DeepPoly(-np.ones(60, ), np.ones(60, )))
    
        env.observation_space = gym.spaces.Box(low=new_obs_space.lower.detach().numpy(), high=new_obs_space.upper.detach().numpy(), shape=(args.red_dim,))
        # print(new_unsafe_domains)
        # print(domains.recover_safe_region(domains.DeepPoly(env.observation_space.low, env.observation_space.high), new_unsafe_domains))
        env.safety = verification.get_constraints(env_model.mars.e2c_predictor.encoder.net, env.original_safety)
        
        new_unsafe_domains = domains.recover_safe_region(domains.DeepPoly(env.observation_space.low, env.observation_space.high), [env.safety])
        
        agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)

        polys = env.safety.to_hyperplanes()

        env.safe_polys = [np.array(polys)]
        env.state_processor = env_model.mars.e2c_predictor.transform
        env.polys = [np.array(domain.to_hyperplanes()) for domain in new_unsafe_domains]

        
        
        if args.neural_safety:
            safe_agent = CSCShield(agent, cost_model,
                                    threshold=args.neural_threshold)
        else:
            shield = ProjectionPolicy(
                env_model.get_symbolic_model(), env.observation_space,
                env.action_space, args.horizon, env.polys, env.safe_polys)
            safe_agent = Shield(shield, agent)
            



    
    if total_numsteps > args.num_steps:
        break
    
        

    writer.add_scalar(f'reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
          .format(i_episode, total_numsteps,
                  episode_steps, round(episode_reward, 2)))
    if safe_agent is not None:
        safe_agent.reset_count()

    if (i_episode - 99) % 1 == 0 and args.eval is True:
        print("starting testing...")
        avg_reward = 0.
        episodes = 1
        unsafe_episodes = 0
        avg_length = 0.
        shield_count = 0
        backup_count = 0
        neural_count = 0
        for _ in range(episodes):
            state, info = env.reset()
            episode_reward = 0
            done = False
            trunc = False
            episode_steps = 0
            trajectory = [state]
            while not done and not trunc:
                if safe_agent is not None:
                    action = safe_agent(state)
                else:
                    action = agent(state)
                next_state, reward, done, trunc, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                if episode_steps >= env._max_episode_steps:
                    done = True
                if env.unsafe(info['state_original'], False):
                    print("UNSAFE")
                    episode_reward += -10
                    print(state, "\n",  action, "\n", next_state)
                    unsafe_episodes += 1
                    done = True
                if done and safe_agent is not None:
                    try:
                        s, a, b, t = safe_agent.report()
                        print("Finished test episode:", s, "shield and", b, "backup and", a,
                              "neural")
                        shield_count+=s
                        backup_count+=b
                        neural_count+=a
                       
                        print("Average time:", t / (s + a + b))
                        safe_agent.reset_count()
                    except Exception as e:
                        print(e)
                        pass
                state = next_state
                trajectory.append(state)
            avg_reward += episode_reward
            avg_length += episode_steps
        avg_reward /= episodes
        avg_length /= episodes
        shield_count /= episodes
        neural_count /= episodes
        backup_count /= episodes
        unsafe_test_episodes+=unsafe_episodes
        total_test_episodes+=episodes
        writer.add_scalar(f'agent/shield', shield_count, i_episode)
        writer.add_scalar(f'agent/neural', neural_count, i_episode)
        writer.add_scalar(f'agent/backup', backup_count, i_episode)
        writer.add_scalar(f'agent/unsafe_real_episodes', real_unsafe_episodes, total_real_episodes)
        writer.add_scalar(f'agent/unsafe_real_episodes_ratio', real_unsafe_episodes/total_real_episodes, total_real_episodes)
        writer.add_scalar(f'agent/unsafe_sim_episodes', unsafe_sim_episodes, total_sim_episodes)
        writer.add_scalar(f'agent/unsafe_sim_episodes_ratio', (unsafe_sim_episodes+0.0000001)/(total_sim_episodes+0.0000001), total_sim_episodes)
        writer.add_scalar(f'agent/unsafe_test_episodes', unsafe_test_episodes, total_test_episodes)
        writer.add_scalar(f'agent/unsafe_test_episodes_ratio', (unsafe_test_episodes+0.0000001)/(total_test_episodes + 0.0000001), total_test_episodes)

        writer.add_scalar(f'reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Unsafe: {}, Avg. Length: {}, Avg. Reward: {}"
              .format(episodes, unsafe_episodes, round(avg_length, 2),
                      round(avg_reward, 2)))
        print("----------------------------------------")
        if (i_episode - 99) % 100 == 0:
            print("Trajectory:")
            print(trajectory)    
        # total_episodes += 1 
        
        
        
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