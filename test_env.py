#!/usr/bin/env python
"""
Script to collect 50,000 transitions, train an environment model,
and then verify its accuracy in two ways:
  1. One-step prediction error in latent space.
  2. Full trajectory prediction error.
  
Assumes that the modules from E2C, abstract interpretation and the environment
benchmarks are available.
"""

import numpy as np
import torch
import os
import datetime
import itertools
import gymnasium as gym
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report
# Import your project modules (make sure these are in your PYTHONPATH)
from benchmarks import envs
from e2c.env_model import get_environment_model
from abstract_interpretation import domains, verification
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
import matplotlib.pyplot as plt


def collect_observations(env, num_steps, seed=123456):
    """
    Collect transitions (s, a, r, s', done) from the environment using a random policy.
    Returns:
      states, actions, rewards, next_states, dones as NumPy arrays.
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    env.seed(seed)
    obs, info = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, done, trunc, info = env.step(action)
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_obs)
        dones.append(done)
        if done:
            obs, info = env.reset()
        else:
            obs = next_obs
    return (np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones))


def train_env_model(states, actions, next_states, rewards, dones, env, seed, red_dim, horizon, epochs=70):
    """
    Train the environment model using the collected transitions.
    Here, we call get_environment_model (which is assumed to do the training).
    """
    # Here we pass dones as the reward-correction if needed.
    model = get_environment_model(
        input_states=states,
        actions=actions,
        output_states=next_states,
        rewards=rewards,
        domain=domains.DeepPoly(env.observation_space.low, env.observation_space.high),
        seed=seed,
        e2c_predictor=None,
        latent_dim=red_dim,
        horizon=horizon,
        epochs=epochs
    )
    return model


def evaluate_one_step(env, env_model, num_samples=1000):
    """
    For a batch of samples, pick a random action and compare:
      (a) The latent state obtained by encoding the next observation from the real environment.
      (b) The latent state predicted by the environment model.
    Returns the mean and standard deviation of the (L2) error in latent space.
    """
    errors = []
    true_samples = []
    pred_samples = []
    obs, info = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_obs, reward, done, trunc, info = env.step(action)
        next_obs_pred, reward_pred = env_model(obs, action, use_neural_model=False)

        true_samples.append(next_obs)
        pred_samples.append(next_obs_pred)

        # Get real latent by encoding next_obs using the E2C predictor's transform.
        # latent_real = env_model.mars.e2c_predictor.transform(next_obs)
        # Get predicted latent from the environment model.
        # print("Real latent: {}, Predicted latent: {}".format(next_obs, next_obs_pred))
        error = np.linalg.norm(next_obs - next_obs_pred)
        errors.append(error)
        obs = next_obs

        if done or trunc:
            obs, info = env.reset()

    true_samples = np.array(true_samples).reshape(-1, env_model.mars.e2c_predictor.z_dim)
    pred_samples = np.array(pred_samples).reshape(-1, env_model.mars.e2c_predictor.z_dim)
    return np.mean(errors), np.std(errors), explained_variance_score(true_samples, pred_samples)


def evaluate_trajectory(env, env_model, trajectory_length=20):
    """
    Starting from a random state, sample a sequence of actions (using a random policy)
    and roll out both the real environment and the environment model (in latent space).
    Returns:
      The mean squared error (MSE) between the latent trajectories over the whole rollout.
    """
    obs, info = env.reset()

    # For the real environment trajectory, we will encode each next observation.
    real_latents = [obs]
    # For the environment model, we roll out in latent space.
    pred_latents = [obs]

    current_obs = obs
    current_pred_obs = obs
    count = 0
    i = 0
    while count < trajectory_length:
        action = env.action_space.sample()  # same random action for both rollouts
        # Real environment rollout
        next_obs, reward, done, trunc, info = env.step(action)
        real_latents.append(next_obs)

        # Environment model rollout:
        pred_latent, rew = env_model(current_pred_obs, action, use_neural_model=False)
        pred_latents.append(pred_latent[0])
        # For a fair comparison, update the current_obs as the real observation
        # (Alternatively, you could update with the decoded prediction if available)
        current_obs = next_obs
        current_pred_obs = pred_latent[0]
        i+=1
        if done or trunc or i == trajectory_length - 1:
            obs, info = env.reset()
            current_obs = obs
            current_pred_obs = obs
            count+=1
            i = 0

    real_latents = np.array(real_latents).reshape(-1, env_model.mars.e2c_predictor.z_dim)
    pred_latents = np.array(pred_latents).reshape(-1, env_model.mars.e2c_predictor.z_dim)
    mse = np.mean((real_latents - pred_latents) ** 2)
    return mse, explained_variance_score(real_latents, pred_latents)

def modify_env(env, env_model, red_dim):
    e2c_mean = env_model.mars.e2c_predictor.mean
    e2c_std = env_model.mars.e2c_predictor.std
    new_obs_space = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, domains.DeepPoly((env.observation_space.low - e2c_mean)/e2c_std, (env.observation_space.high - e2c_mean)/e2c_std)))
    env.observation_space = gym.spaces.Box(low=new_obs_space.lower.detach().numpy(), high=new_obs_space.upper.detach().numpy(), shape=(red_dim,))
    
    safety_domain = domains.DeepPoly((env.original_safety.lower - e2c_mean)/e2c_std, (env.original_safety.upper - e2c_mean)/e2c_std)
    
    env.safety = domains.DeepPoly(*verification.get_ae_bounds(env_model.mars.e2c_predictor, safety_domain))
    
    
    unsafe_domains_list = domains.recover_safe_region(new_obs_space, [env.safety])
        
    
    
    polys = [np.array(env.safety.to_hyperplanes())]

    env.safe_polys = polys
    env.state_processor = env_model.mars.e2c_predictor.transform
    env.polys = [np.array(domain.to_hyperplanes()) for domain in unsafe_domains_list]
    

def compare_safety_classification(env, num_points=1000):
    """
    Randomly generate points in the original state space (using the original
    observation_space bounds) and classify each point as safe (label 1) or unsafe (label 0)
    using both the original safety function and the new safety (after modification).
    
    Returns a classification report comparing the original safety classification (as truth)
    and the new safety classification (as predictions).
    """
    true_labels = []       # from original safety (env.original_safe_polys)
    predicted_labels = []  # from new safety (env.safe_polys) via simulated=True

    # Use the bounds from the original observation space.
    low = env.original_observation_space.low
    high = env.original_observation_space.high

    unsafe_spaces = domains.recover_safe_region(domains.DeepPoly((env.original_observation_space.low), (env.original_observation_space.high)), [env.original_safety])

    random_func = lambda: unsafe_spaces[np.random.randint(len(unsafe_spaces))].sample(1)

    print("Generating {} random points...".format(random_func()))

    # Generate random points uniformly from the original observation space.
    for _ in range(num_points):
        pt = np.random.uniform(low, high)
        pt = random_func().detach().numpy()
        # In our convention, we define:
        # safe -> label 1, unsafe -> label 0.
        # env.unsafe(pt, simulated=False) uses the original safety polys.
        orig_unsafe = env.unsafe(pt, simulated=False)
        new_unsafe  = env.unsafe(env.state_processor(pt.reshape(1, -1)), simulated=True)

        true_label = 1 if orig_unsafe else 0
        predicted_label = 1 if new_unsafe else 0

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    report = classification_report(true_labels, predicted_labels, target_names=["Unsafe", "Safe"])
    print("Safety classification report:")
    print(report)
    return report


def main():
    # Hyperparameters and settings
    seed = 123456
    
    num_observations = 20000
    red_dim = 8   # Reduced latent dimension
    horizon = 20   # Horizon for safety/transition model
    env_name = "lunar_lander"  # Example environment name

    # Setup environment
    env = envs.get_env_from_name(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Collecting {} observations...".format(num_observations))
    states, actions, rewards, next_states, dones = collect_observations(env, num_observations, seed)
    print("Collected {} transitions.".format(len(states)))

    # Train environment model
    print("Training environment model...")
    env_model = train_env_model(states, actions, next_states, rewards, dones, env, seed, red_dim, horizon, epochs=10)
    modify_env(env, env_model, red_dim)


    print("Environment model trained.")

    # Evaluate one-step prediction error in latent space
    print("Evaluating one-step prediction accuracy...")
    mean_err, std_err, ev_score = evaluate_one_step(env, env_model, num_samples=1000)
    print("One-step latent prediction error: Mean = {:.4f}, Std = {:.4f}, EV Score = {:.4f}".format(mean_err, std_err, ev_score))

    # Evaluate full trajectory prediction error
    print("Evaluating full trajectory prediction accuracy...")
    traj_mse, ev_score= evaluate_trajectory(env, env_model, trajectory_length=20)
    print("Trajectory latent MSE {:.4f}, EV Score {:.4f}".format(traj_mse, ev_score))

    # Optionally, compare safety classification
    print("Comparing safety classification...")
    safety_report = compare_safety_classification(env, num_points=10000)

    # # Optionally, plot the trajectories in latent space for visualization
    # plt.figure()
    # plt.plot(real_latents[:, 0], real_latents[:, 1], 'g-', label="Real")
    # plt.plot(pred_latents[:, 0], pred_latents[:, 1], 'r--', label="Predicted")
    # plt.legend()
    # plt.xlabel("Latent dim 1")
    # plt.ylabel("Latent dim 2")
    # plt.title("Latent Trajectories: Real vs Environment Model")
    # plt.show()


if __name__ == "__main__":
    main()
