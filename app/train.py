import os
import argparse
import logging
from typing import Any, Dict

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from src.main.python.utils.helpers import get_logger
from src.main.python.utils.callbacks import OrbitalPhysicsCallbacks
from src.main.python.utils.rllib_setup import (
    DEFAULT_START_TIME,
    build_policies_to_train,
    build_policy_setup,
    build_rllib_env_config,
    parse_maneuver_frame,
    register_orbital_env,
    rllib_policy_mapping_fn,
    run_spec_from_args,
    save_run_parameters,
)

# Initialize logger
logger = get_logger("train_app", level=logging.INFO)


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
    scenario_group.add_argument("--start-time", type=str, default=DEFAULT_START_TIME, help="Simulation start time (UTC).")
    scenario_group.add_argument(
        "--max-delta-v-mps",
        type=float,
        default=20.0,
        help="The maximum single maneuver delta-v in m/s.",
    )
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

def _validate_training_args(args: argparse.Namespace) -> None:
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


def _build_experiment_name(args: argparse.Namespace, maneuver_frame: str) -> str:
    if args.name:
        return args.name
    return f"ppo_{args.n_interceptors}i_{args.n_targets}t_{maneuver_frame.lower()}"


def setup_training(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Setup all that is needed for training.
    """
    _validate_training_args(args)
    spec = run_spec_from_args(args)

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    os.makedirs(local_dir, exist_ok=True)
    experiment_name = _build_experiment_name(args, spec.maneuver_frame)
    results_dir = os.path.join(local_dir, experiment_name)
    save_run_parameters(spec, results_dir)
    env_name = register_orbital_env()
    policy_setup = build_policy_setup(spec)
    policies_to_train = build_policies_to_train(
        policy_setup["policies"].keys(),
        freeze_targets=spec.freeze_targets,
    )

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=env_name,
            env_config=build_rllib_env_config(spec),
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
            policies=policy_setup["policies"],
            policy_mapping_fn=rllib_policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
    )

    return {
        "args": args,
        "spec": spec,
        "local_dir": local_dir,
        "experiment_name": experiment_name,
        "config": config,
        "interceptor_obs_space": policy_setup["interceptor_obs_space"],
        "target_obs_space": policy_setup["target_obs_space"],
        "results_dir": results_dir,
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
        if "last_checkpoint_path" in training_ctx:
            save_run_parameters(training_ctx["spec"], training_ctx["last_checkpoint_path"])

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
        spec = training_ctx["spec"]
    
        logger.info(f"Initialized training with {spec.n_interceptors} interceptors and {spec.n_targets} targets.")
        logger.info(f"Interceptor Obs Space: {training_ctx['interceptor_obs_space']}")
        logger.info(f"Target Obs Space: {training_ctx['target_obs_space']}")
        logger.info(f"Maneuver frame: {spec.maneuver_frame}")
        logger.info(f"Targets maneuvering disabled: {spec.freeze_targets}")
    
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
