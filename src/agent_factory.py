from omegaconf import DictConfig, OmegaConf
from src.policies.ppo_agent import PPOPolicy
from src.policies.sac_agent import SACPolicy
from src.policies.abstract_agent import Agent
import gymnasium as gym

def create_agent(cfg: DictConfig, env: gym.Env) -> Agent:
    """
    Factory function to create an agent based on configuration.
    """
    agent_type = cfg.agent.name
    
    # Merge root config and agent config to ensure both 'cuda' (root) and 'gamma' (agent) are available as top-level keys
    # Prioritize 'cfg' (root/overrides) over 'cfg.agent' (defaults)
    merged_args = OmegaConf.merge(cfg.agent, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    
    if agent_type == "sac":
        # SAC expects replay_size, seed, batch_size, and 'agent' config
        # The cfg.agent contains sac specific params
        return SACPolicy(
            gym_env=env,
            replay_size=cfg.agent.replay_size,
            seed=cfg.seed,
            batch_size=cfg.agent.batch_size,
            sac_args=merged_args
        )
    elif agent_type == "ppo":
        return PPOPolicy(
            gym_env=env,
            replay_size=cfg.agent.replay_size,
            seed=cfg.seed,
            batch_size=cfg.agent.batch_size,
            args=merged_args
        )
    elif agent_type == "all_c":
        from src.policies.all_c_agent import ALLCAgent
        return ALLCAgent(
            gym_env=env,
            args=merged_args,
            sac_args=merged_args
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
