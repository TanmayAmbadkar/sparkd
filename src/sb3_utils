import torch as th
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import numpy as np
from gymnasium import spaces

def learn(
        policy,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
    """
    Train the policy using the specified number of timesteps.

    Parameters:
    policy: The policy to be trained.
    total_timesteps (int): The total number of timesteps to train the policy.
    callback: Callback function for training events.
    log_interval (int): Interval at which to log training information.
    tb_log_name (str): Name for TensorBoard log.
    reset_num_timesteps (bool): Whether to reset the number of timesteps.
    progress_bar (bool): Whether to display a progress bar.

    Returns:
    policy: The trained policy.
    observations (list): The list of observations collected during training.
    """
    iteration = 0

    total_timesteps, callback = policy._setup_learn(
        total_timesteps,
        callback,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    callback.on_training_start(locals(), globals())

    assert policy.env is not None
    
    observations = []

    while policy.num_timesteps < total_timesteps:
        continue_training, obs = collect_rollouts(policy, policy.env, callback, policy.rollout_buffer, n_rollout_steps=policy.n_steps)
        observations.extend(obs)
        
        if not continue_training:
            break

        iteration += 1
        policy._update_current_progress_remaining(policy.num_timesteps, total_timesteps)

        policy.train()
        print(f"\r{policy.num_timesteps}/{total_timesteps}", end="")

    callback.on_training_end()

    return policy, observations

def collect_rollouts(
        policy,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.

    Parameters:
    policy: The policy used to collect rollouts.
    env: The training environment.
    callback: Callback that will be called at each step (and at the beginning and end of the rollout).
    rollout_buffer: Buffer to fill with rollouts.
    n_rollout_steps (int): Number of experiences to collect per environment.

    Returns:
    bool: True if function returned with at least `n_rollout_steps` collected, False if callback terminated rollout prematurely.
    observations (list): The list of observations collected during rollouts.
    """
    assert policy._last_obs is not None, "No previous observation was provided"
    # Switch to eval mode (this affects batch norm / dropout)
    policy.policy.set_training_mode(False)

    n_steps = 0
    rollout_buffer.reset()
    # Sample new weights for the state dependent exploration
    if policy.use_sde:
        policy.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()
    observations = []

    while n_steps < n_rollout_steps:
        if policy.use_sde and policy.sde_sample_freq > 0 and n_steps % policy.sde_sample_freq == 0:
            # Sample a new noise matrix
            policy.policy.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(policy._last_obs, policy.device)
            actions, values, log_probs = policy.policy(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions

        if isinstance(policy.action_space, spaces.Box):
            if policy.policy.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions = policy.policy.unscale_action(clipped_actions)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions = np.clip(actions, policy.action_space.low, policy.action_space.high)

        observations.append(policy._last_obs)
        new_obs, rewards, dones, infos = env.step(clipped_actions)

        policy.num_timesteps += env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        if not callback.on_step():
            return False

        policy._update_info_buffer(infos, dones)
        n_steps += 1

        if isinstance(policy.action_space, spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstrapping with value function
        # see GitHub issue #633
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = policy.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = policy.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                rewards[idx] += policy.gamma * terminal_value

        rollout_buffer.add(
            policy._last_obs,  # type: ignore[arg-type]
            actions,
            rewards,
            policy._last_episode_starts,  # type: ignore[arg-type]
            values,
            log_probs,
        )
        policy._last_obs = new_obs  # type: ignore[assignment]
        policy._last_episode_starts = dones

    with th.no_grad():
        # Compute value for the last timestep
        values = policy.policy.predict_values(obs_as_tensor(new_obs, policy.device))  # type: ignore[arg-type]

    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.update_locals(locals())

    callback.on_rollout_end()

    return True, observations
