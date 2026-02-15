import os
import argparse
import logging
import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.environment.scenarios import pursuit_evasion_scenario
from src.main.python.utils.helpers import get_logger
from src.main.python.utils.callbacks import OrbitalPhysicsCallbacks

# Initialize logger
logger = get_logger("train_app", level=logging.INFO)

def env_creator(config):
    """
    Creates and wraps the orbital environment for RLlib.
    """
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
    if "freeze_targets" in config:
        env_config["freeze_targets"] = config["freeze_targets"]
        
    env = OrbitalEnv(agent_configs, env_config)
    return ParallelPettingZooEnv(env)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train MARL agents in an orbital pursuit-evasion scenario.")
    
    # Scenario arguments
    scenario_group = parser.add_argument_group("Scenario Configuration")
    scenario_group.add_argument("--n-interceptors", type=int, default=1, help="Number of interceptor agents.")
    scenario_group.add_argument("--n-targets", type=int, default=1, help="Number of target agents.")
    scenario_group.add_argument("--timestep", type=float, default=60.0, help="Simulation timestep in seconds.")
    scenario_group.add_argument("--episode-length", type=int, default=100, help="Number of steps per episode.")
    scenario_group.add_argument(
        "--freeze-targets",
        action="store_true",
        help="Force target agents to apply zero delta-v at each step.",
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--iterations", type=int, default=20, help="Number of training iterations.")
    train_group.add_argument("--batch-size", type=int, default=4000, help="Training batch size.")
    train_group.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    train_group.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # Execution arguments
    exec_group = parser.add_argument_group("Execution Configuration")
    exec_group.add_argument("--num-workers", type=int, default=1, help="Number of environment rollout workers.")
    exec_group.add_argument("--num-gpus", type=float, default=0, help="Number of GPUs to use (can be fractional).")
    exec_group.add_argument("--checkpoint-freq", type=int, default=1, help="Frequency of checkpointing (in iterations).")
    exec_group.add_argument("--resume", action="store_true", help="Resume training from last checkpoint.")
    exec_group.add_argument("--name", type=str, required=False, default=None, help="Name of the experiment.")
    exec_group.add_argument("--local-dir", type=str, default="~/results/marl-swarm-evasion/ray_results", help="Local directory for results.")
    
    return parser.parse_args()

def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Maps agent IDs to policies.
    """
    if agent_id.startswith("interceptor"):
        return "interceptor_policy"
    elif agent_id.startswith("target"):
        return "target_policy"
    return "shared_policy"

def main():
    args = parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register the environment
    register_env("orbital_env", lambda config: env_creator(config))
    
    # Instantiate a temporary environment to retrieve observation and action spaces
    # This ensures that policies are correctly configured for the current scenario.
    temp_env_config = {
        "n_interceptors": args.n_interceptors,
        "n_targets": args.n_targets,
        "timestep_sec": args.timestep,
        "episode_length": args.episode_length,
        "seed": args.seed,
        "freeze_targets": args.freeze_targets,
    }
    temp_env = env_creator(temp_env_config)
    
    # ParallelPettingZooEnv exposes observation_space and action_space as dicts 
    # mapping agent IDs to their individual spaces.
    # We extract the individual spaces to define our policies.
    def get_agent_space(space_dict, role_prefix):
        for agent_id, space in space_dict.items():
            if agent_id.startswith(role_prefix):
                return space
        return next(iter(space_dict.values()))

    int_obs_space = get_agent_space(temp_env.observation_space, "interceptor")
    int_act_space = get_agent_space(temp_env.action_space, "interceptor")
    tar_obs_space = get_agent_space(temp_env.observation_space, "target")
    tar_act_space = get_agent_space(temp_env.action_space, "target")
    
    logger.info(f"Initialized training with {args.n_interceptors} interceptors and {args.n_targets} targets.")
    logger.info(f"Interceptor Obs Space: {int_obs_space}")
    logger.info(f"Target Obs Space: {tar_obs_space}")
    logger.info(f"Targets maneuvering disabled: {args.freeze_targets}")

    policies_to_train = ["interceptor_policy"] if args.freeze_targets else None
    
    # Configure RLlib PPO Algorithm
    config = (
        PPOConfig()
        .environment(
            "orbital_env", 
            env_config=temp_env_config
        )
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .resources(num_gpus=args.num_gpus)
        .training(
            train_batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma,
            num_epochs=10,
            model={"fcnet_hiddens": [256, 256]}
        )
        .multi_agent(
            policies={
                "interceptor_policy": (None, int_obs_space, int_act_space, {}),
                "target_policy": (None, tar_obs_space, tar_act_space, {}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .debugging(seed=args.seed)
        .callbacks(OrbitalPhysicsCallbacks)
    )
    
    # Prepare storage path
    storage_path = os.path.abspath(os.path.expanduser(args.local_dir))
    storage_path = os.path.normpath(storage_path)
    exp_name = args.name

    if args.resume and not exp_name:
        # If resume is requested but no name is provided, 
        # we assume local_dir is the experiment directory.
        exp_name = os.path.basename(storage_path)
        storage_path = os.path.dirname(storage_path)

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    
    logger.info(f"Results will be saved to: {os.path.join(storage_path, exp_name) if exp_name else storage_path}")

    # Start training
    tune.run(
        "PPO",
        name=exp_name,
        config=config.to_dict(),
        stop={"training_iteration": args.iterations},
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        storage_path=storage_path,
        resume=args.resume,
    )
    
    logger.info("Training completed successfully.")
    ray.shutdown()

if __name__ == "__main__":
    main()
