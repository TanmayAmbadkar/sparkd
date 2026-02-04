import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any

from src.algorithms.ppo.ppo import PPO
from src.algorithms.sac.replay_memory import ReplayMemory
from src.policies.abstract_agent import Agent

class PPOPolicy(Agent):

    def __init__(self,
                 gym_env: gym.Env,
                 replay_size: int,
                 seed: int,
                 batch_size: int,
                 args):
        obs_shape = gym_env.observation_space.shape
        input_dims = obs_shape if len(obs_shape) == 3 else obs_shape[0]
        self.agent = PPO(input_dims,
                         gym_env.action_space, args)
        self.memory = ReplayMemory(replay_size, gym_env.observation_space, gym_env.action_space.shape[0], seed)
        self.updates = 0
        self.minibatch_size = args.mini_batch_size
        self.batch_size = batch_size

    def __call__(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        return self.agent.select_action(state)[0]

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, cost: float) -> None:
        self.memory.push(state, action, reward, next_state, done, cost)

    def train(self) -> Dict[str, float]:
        ret = self.agent.update_parameters(self.memory, batch_size=self.minibatch_size, epochs = 40)
        self.updates += 1
        return ret

    def report(self):
        return 0, 0

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)
        
    def save_checkpoint(self, path):
        self.agent.save_checkpoint(path)
