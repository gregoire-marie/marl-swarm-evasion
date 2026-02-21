import os
import argparse
import logging
from typing import Any, Dict

import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.environment.scenarios import pursuit_evasion_scenario
from src.main.python.utils.helpers import get_logger, policy_mapping_fn
from src.main.python.utils.callbacks import OrbitalPhysicsCallbacks

# Initialize logger
logger = get_logger("train_app", level=logging.INFO)
SUPPORTED_MANEUVER_FRAMES = ("ECI", "TNW")


def parse_maneuver_frame(value: str) -> str:
    """
    Parses the maneuver frame string, unsensitive to the case.
    """
    frame = str(value).strip().upper()
    if frame not in SUPPORTED_MANEUVER_FRAMES:
        raise argparse.ArgumentTypeError(
            f"Unsupported maneuver frame '{value}'. Supported frames: {list(SUPPORTED_MANEUVER_FRAMES)}."
        )
    return frame


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
        "--maneuver-frame",
        type=parse_maneuver_frame,
        default="ECI",
        help="Action frame for maneuvers: ECI or TNW.",
    )
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
    train_group.add_argument("--num-epochs", type=int, default=10, help="Number of SGD epochs per training batch.")
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

def _validate_args(args: argparse.Namespace) -> None:
    if args.n_interceptors <= 0:
        raise ValueError("--n-interceptors must be >= 1.")
    if args.n_targets <= 0:
        raise ValueError("--n-targets must be >= 1.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be >= 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.num_gpus < 0:
        raise ValueError("--num-gpus must be >= 0.")
    if args.checkpoint_freq < 0:
        raise ValueError("--checkpoint-freq must be >= 0.")


def _build_experiment_name(args: argparse.Namespace) -> str:
    if args.name:
        return args.name
    return f"ppo_{args.n_interceptors}i_{args.n_targets}t_{args.maneuver_frame.lower()}"


def setup_training(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Setup all that is needed for training.
    """
    _validate_args(args)

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    os.makedirs(local_dir, exist_ok=True)
    experiment_name = _build_experiment_name(args)

    agent_configs = pursuit_evasion_scenario(
        n_interceptors=args.n_interceptors,
        n_targets=args.n_targets,
        seed=args.seed,
    )
    orbital_env_config = {
        "timestep_sec": args.timestep,
        "episode_length": args.episode_length,
        "start_time": "2025-01-01 00:00:00",
        "maneuver_frame": args.maneuver_frame,
        "freeze_targets": args.freeze_targets,
    }

    env_name = OrbitalEnv.metadata.get("name", "orbital_env_v0")

    def env_creator(config: Dict[str, Any]) -> ParallelPettingZooEnv:
        agent_cfgs = pursuit_evasion_scenario(
            n_interceptors=int(config["n_interceptors"]),
            n_targets=int(config["n_targets"]),
            seed=int(config["seed"]),
        )
        return ParallelPettingZooEnv(
            OrbitalEnv(
                agent_configs=agent_cfgs,
                env_config=config["orbital_env_config"],
            )
        )

    register_env(env_name, env_creator)

    probe_env = OrbitalEnv(agent_configs=agent_configs, env_config=orbital_env_config)
    interceptor_id = next((aid for aid, cfg in agent_configs.items() if cfg["role"] == "interceptor"), None)
    target_id = next((aid for aid, cfg in agent_configs.items() if cfg["role"] == "target"), None)

    policies: Dict[str, PolicySpec] = {}
    interceptor_obs_space = None
    target_obs_space = None

    if interceptor_id is not None:
        interceptor_obs_space = probe_env.observation_space(interceptor_id)
        interceptor_act_space = probe_env.action_space(interceptor_id)
        policies["interceptor_policy"] = PolicySpec(
            observation_space=interceptor_obs_space,
            action_space=interceptor_act_space,
        )

    if target_id is not None:
        target_obs_space = probe_env.observation_space(target_id)
        target_act_space = probe_env.action_space(target_id)
        policies["target_policy"] = PolicySpec(
            observation_space=target_obs_space,
            action_space=target_act_space,
        )

    if not policies:
        raise ValueError("No policies were created. Check scenario agent configuration.")

    policies_to_train = [policy_id for policy_id in policies.keys() if policy_id != "target_policy" or not args.freeze_targets]
    if not policies_to_train:
        policies_to_train = list(policies.keys())

    def rllib_policy_mapping_fn(agent_id: str, *unused_args: Any, **unused_kwargs: Any) -> str:
        return policy_mapping_fn(agent_id)

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=env_name,
            env_config={
                "n_interceptors": args.n_interceptors,
                "n_targets": args.n_targets,
                "seed": args.seed,
                "orbital_env_config": orbital_env_config,
            },
            disable_env_checking=True,
        )
        .env_runners(num_env_runners=args.num_workers, rollout_fragment_length="auto")
        .training(
            train_batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma,
            num_epochs=args.num_epochs,
        )
        .resources(num_gpus=args.num_gpus)
        .debugging(seed=args.seed, log_level="INFO")
        .callbacks(OrbitalPhysicsCallbacks)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=rllib_policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
    )

    return {
        "args": args,
        "local_dir": local_dir,
        "experiment_name": experiment_name,
        "env_name": env_name,
        "config": config,
        "interceptor_obs_space": interceptor_obs_space,
        "target_obs_space": target_obs_space,
        "results_dir": os.path.join(local_dir, experiment_name),
    }

def launch_training(training_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform the training.
    """
    args: argparse.Namespace = training_ctx["args"]
    config: PPOConfig = training_ctx["config"]

    resume_mode = "AUTO" if args.resume else False

    analysis = tune.run(
        "PPO",
        name=training_ctx["experiment_name"],
        stop={"training_iteration": args.iterations},
        config=config.to_dict(),
        storage_path=training_ctx["local_dir"],
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        resume=resume_mode,
        verbose=1,
    )
    training_ctx["analysis"] = analysis

    if analysis.trials:
        trial = analysis.trials[0]
        last_checkpoint = analysis.get_last_checkpoint(trial)
        if hasattr(last_checkpoint, "path"):
            training_ctx["last_checkpoint_path"] = last_checkpoint.path
        elif last_checkpoint is not None:
            training_ctx["last_checkpoint_path"] = str(last_checkpoint)

    return training_ctx

def teardown_training(training_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tear down everything that need to be cleaned after the training.
    """
    if ray.is_initialized():
        ray.shutdown()
    return training_ctx


def main():
    args = parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Set up the training
    training_ctx = {}
    try:
        training_ctx = setup_training(args)
    
        logger.info(f"Initialized training with {args.n_interceptors} interceptors and {args.n_targets} targets.")
        logger.info(f"Interceptor Obs Space: {training_ctx['interceptor_obs_space']}")
        logger.info(f"Target Obs Space: {training_ctx['target_obs_space']}")
        logger.info(f"Maneuver frame: {args.maneuver_frame}")
        logger.info(f"Targets maneuvering disabled: {args.freeze_targets}")
    
        logger.info(f"Results will be saved to: {training_ctx['results_dir']}")

        # Start training
        training_ctx = launch_training(training_ctx)

        logger.info("Training completed successfully.")
        if "last_checkpoint_path" in training_ctx:
            logger.info(f"Last checkpoint: {training_ctx['last_checkpoint_path']}")

    finally:
        training_ctx = teardown_training(training_ctx)

if __name__ == "__main__":
    main()
