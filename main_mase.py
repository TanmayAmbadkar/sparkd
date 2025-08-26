import argparse
import datetime
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

# Import your PPO implementation and the new MASE module
from src.policy import PPOPolicy
from mase.mase import MASE     # The new MASE implementation
from benchmarks import envs

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='MASE PPO Training')
parser.add_argument('--env_name', default="Pendulum-v1")
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--num_steps', type=int, default=200000)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--lr', type=int, default=0.0003)
parser.add_argument('--mini_batch_size', type=int, default=64)
parser.add_argument('--update_every', type=int, default=2048, help="Number of steps to collect before updating PPO.")
parser.add_argument('--batch_size', type=int, default=256, help="Minibatch size for PPO updates.")
parser.add_argument('--cuda', action="store_true")
# MASE specific arguments
parser.add_argument('--cumulative_safety_limit', type=float, default=1.0, help="Total safety cost allowed per episode.")
parser.add_argument('--beta', type=float, default=2.5, help="Confidence level for the safety model.")
parser.add_argument('--penalty_c', type=float, default=100.0, help="Penalty multiplier for emergency stops.")
args = parser.parse_args()

def main():
    # --- Setup ---
    env = envs.get_env_from_name(args.env_name)
    
    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Tensorboard setup
    run_name = f"MASE_PPO_{args.env_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(f'runs/{run_name}')

    # --- Agent Initialization ---
    agent = PPOPolicy(env, args.update_every, args.seed, args.update_every, args)
    mase_shield = MASE(env.observation_space.shape[0], env.action_space, vars(args))

    # --- Logging and Counters ---
    total_numsteps = 0
    unsafe_episodes = 0
    total_episodes = 0
    emergency_stops = 0

    # --- Main Training Loop ---
    while total_numsteps < args.num_steps:
        state, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0
        done = False
        
        # Initialize safety budget for the episode (for cumulative cost problems)
        safety_budget = args.cumulative_safety_limit

        while not done:
            # 1. The RL agent proposes an action
            proposed_action = agent(state)

            # 2. The environment takes a step
            next_state, reward, terminated, truncated, info = env.step(proposed_action)
            done = terminated or truncated
            
            # Use a binary cost for this example (1 if unsafe, 0 otherwise)
            # You can replace this with info.get('cost', 0.0) for Safety Gym
            is_unsafe = env.unsafe(next_state, False) if hasattr(env, 'unsafe') else False
            safety_cost = 1.0 if is_unsafe else 0.0

            # 3. MASE: Calculate the safety threshold for the next state
            next_safety_threshold = safety_budget - safety_cost

            # 4. MASE: Perform the lookahead check for a dead end
            is_dead_end = mase_shield.check_for_dead_end(next_state, next_safety_threshold)

            final_reward = reward
            final_done = done

            if is_dead_end:
                # 5a. EMERGENCY STOP
                final_reward = mase_shield.get_emergency_penalty(next_state)
                final_done = True # Terminate the episode
                emergency_stops += 1
                print(f"Episode {total_episodes}: EMERGENCY STOP at step {episode_steps}. Penalty: {final_reward:.2f}")
            
            if is_unsafe and not is_dead_end:
                # This is a real safety violation that MASE did not prevent
                # It happens when the safety model is inaccurate
                unsafe_episodes += 1
                # Apply a penalty to discourage this
                final_reward -= 100.0
                final_done = True # Terminate the episode

            # 6. Add data to the RL agent's memory
            agent.add(state, proposed_action, final_reward, next_state, final_done, safety_cost)

            # 7. Add the observed safety data to the safety model's buffer
            mase_shield.add_safety_data(state, proposed_action, safety_cost)

            # 8. Update state and counters
            state = next_state
            safety_budget = next_safety_threshold
            episode_reward += reward # Log the true reward
            episode_cost += safety_cost
            episode_steps += 1
            total_numsteps += 1

            # 9. Periodically update the PPO policy and the MASE safety model
            if len(agent.memory) >= args.batch_size:
                agent.train()
                mase_shield.update_safety_model()

            if final_done:
                break
        
        # --- End of Episode Logging ---
        total_episodes += 1
        writer.add_scalar('reward/train', episode_reward, total_numsteps)
        writer.add_scalar('cost/train', episode_cost, total_numsteps)
        writer.add_scalar('safety/unsafe_episodes', unsafe_episodes, total_numsteps)
        writer.add_scalar('safety/emergency_stops', emergency_stops, total_numsteps)
        
        print(f"Episode: {total_episodes}, Total Steps: {total_numsteps}, Reward: {episode_reward:.2f}, Cost: {episode_cost}")

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
