import os
import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.environment.scenarios import pursuit_evasion_scenario

def env_creator(config):
    agent_configs, env_config = pursuit_evasion_scenario(
        n_interceptors=config.get("n_interceptors", 1),
        n_targets=config.get("n_targets", 1),
        seed=config.get("seed", 42)
    )
    # Merge overrides from config
    if "timestep_sec" in config:
        env_config["timestep_sec"] = config["timestep_sec"]
    if "episode_length" in config:
        env_config["episode_length"] = config["episode_length"]
        
    env = OrbitalEnv(agent_configs, env_config)
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    ray.init()
    
    register_env("orbital_env", lambda config: env_creator(config))
    
    # Simple test to verify environment creation
    test_env = env_creator({"n_interceptors": 1, "n_targets": 1})
    obs, infos = test_env.reset()
    print(f"Initial observations keys: {obs.keys()}")
    
    # Configure RLlib Trainer (PPO)
    config = (
        PPOConfig()
        .environment("orbital_env", env_config={"n_interceptors": 1, "n_targets": 1})
        .framework("torch")
        .env_runners(num_env_runners=1)
        .training(
            train_batch_size=200,
            num_epochs=5,
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )
    
    # Run a short training
    storage_path = os.path.expanduser("~/results/marl-swarm-evasion/ray_results")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 5},
        checkpoint_at_end=True,
        storage_path=storage_path,
    )
    
    ray.shutdown()
