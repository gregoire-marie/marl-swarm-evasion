import argparse
import json
import os
from typing import Any, Dict, Mapping, Optional, Type

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from src.main.python.experiment.curriculum import CurriculumConfig
from src.main.python.utils.callbacks import OrbitalPhysicsCallbacks
from src.main.python.utils.constants import DEFAULT_START_TIME
from src.main.python.utils.rllib_setup import (
    OrbitalRunSpec,
    build_curriculum_policies_to_train,
    build_policies_to_train,
    build_policy_setup,
    build_rllib_env_config,
    parse_maneuver_frame,
    register_orbital_env,
    rllib_policy_mapping_fn,
    run_spec_from_args,
    save_run_parameters,
)


def add_scenario_args(parser: argparse.ArgumentParser) -> None:
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


def add_training_args(parser: argparse.ArgumentParser, *, seed_default: Optional[int] = 42) -> None:
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--iterations", type=int, default=20, help="Number of training iterations.")
    train_group.add_argument("--batch-size", type=int, default=4000, help="Training batch size.")
    train_group.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    train_group.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    train_group.add_argument("--num-epochs", type=int, default=10, help="Number of SGD epochs per training batch.")
    train_group.add_argument("--seed", type=int, default=seed_default, help="Random seed.")


def add_execution_args(parser: argparse.ArgumentParser) -> None:
    exec_group = parser.add_argument_group("Execution Configuration")
    exec_group.add_argument("--num-workers", type=int, default=1, help="Number of environment rollout workers.")
    exec_group.add_argument("--num-gpus", type=float, default=0, help="Number of GPUs to use (can be fractional).")
    exec_group.add_argument("--checkpoint-freq", type=int, default=1, help="Frequency of checkpointing (in iterations).")
    exec_group.add_argument("--resume", action="store_true", help="Resume training from last checkpoint.")
    exec_group.add_argument("--name", type=str, required=False, default=None, help="Name of the experiment.")
    exec_group.add_argument(
        "--local-dir",
        type=str,
        default="~/results/marl-swarm-evasion/ray_results",
        help="Local directory for results.",
    )


def validate_training_args(args: argparse.Namespace) -> None:
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
    if getattr(args, "ray_num_cpus", None) is not None and args.ray_num_cpus <= 0:
        raise ValueError("--ray-num-cpus must be >= 1.")
    if getattr(args, "torch_num_threads", None) is not None and args.torch_num_threads <= 0:
        raise ValueError("--torch-num-threads must be >= 1.")


def build_experiment_name(
    args: argparse.Namespace,
    spec: OrbitalRunSpec,
    *,
    curriculum_config: Optional[CurriculumConfig] = None,
) -> str:
    if args.name:
        return args.name
    if curriculum_config is not None:
        return f"ppo_curriculum_{curriculum_config.N_max}i_{curriculum_config.M_max}t"
    return f"ppo_{spec.n_interceptors}i_{spec.n_targets}t_{spec.maneuver_frame.lower()}"


def spec_from_curriculum(curriculum_config: CurriculumConfig, *, seed: Optional[int] = None) -> OrbitalRunSpec:
    max_delta_v_mps = max(stage.max_delta_v_mps for stage in curriculum_config.stages)
    max_episode_length = max(stage.episode_length for stage in curriculum_config.stages)
    return OrbitalRunSpec(
        n_interceptors=curriculum_config.N_max,
        n_targets=curriculum_config.M_max,
        timestep=curriculum_config.timestep,
        episode_length=max_episode_length,
        start_time=curriculum_config.start_time,
        max_delta_v_mps=max_delta_v_mps,
        maneuver_frame=curriculum_config.stages[0].maneuver_frame,
        freeze_targets=False,
        seed=curriculum_config.seed if seed is None else int(seed),
    )


def setup_training(
    args: argparse.Namespace,
    *,
    curriculum_config: Optional[CurriculumConfig] = None,
    curriculum_parameters: Optional[Mapping[str, Any]] = None,
    callbacks_cls: Type[OrbitalPhysicsCallbacks] = OrbitalPhysicsCallbacks,
) -> Dict[str, Any]:
    validate_training_args(args)
    spec = (
        spec_from_curriculum(curriculum_config, seed=getattr(args, "seed", None))
        if curriculum_config is not None
        else run_spec_from_args(args)
    )

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    os.makedirs(local_dir, exist_ok=True)
    experiment_name = build_experiment_name(args, spec, curriculum_config=curriculum_config)
    results_dir = os.path.join(local_dir, experiment_name)
    save_run_parameters(spec, results_dir)
    if curriculum_config is not None:
        _save_curriculum_config(curriculum_config, results_dir, curriculum_parameters=curriculum_parameters)

    env_name = register_orbital_env()
    policy_setup = build_policy_setup(spec, curriculum_config=curriculum_config)
    policies_to_train = (
        build_curriculum_policies_to_train(curriculum_config)
        if curriculum_config is not None
        else build_policies_to_train(policy_setup["policies"].keys(), freeze_targets=spec.freeze_targets)
    )

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=env_name,
            env_config=build_rllib_env_config(spec, curriculum_config=curriculum_config),
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
        .debugging(seed=spec.seed, log_level="INFO")
        .callbacks(callbacks_cls)
        .multi_agent(
            policies=policy_setup["policies"],
            policy_mapping_fn=rllib_policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
    )

    return {
        "args": args,
        "spec": spec,
        "curriculum_config": curriculum_config,
        "curriculum_parameters": curriculum_parameters,
        "local_dir": local_dir,
        "experiment_name": experiment_name,
        "config": config,
        "interceptor_obs_space": policy_setup["interceptor_obs_space"],
        "target_obs_space": policy_setup["target_obs_space"],
        "results_dir": results_dir,
    }


def launch_training(training_ctx: Dict[str, Any]) -> Dict[str, Any]:
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
            curriculum_config = training_ctx.get("curriculum_config")
            if curriculum_config is not None:
                _save_curriculum_config(
                    curriculum_config,
                    training_ctx["last_checkpoint_path"],
                    curriculum_parameters=training_ctx.get("curriculum_parameters"),
                )

    return training_ctx


def teardown_training(training_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if ray.is_initialized():
        ray.shutdown()
    return training_ctx


def _save_curriculum_config(
    curriculum_config: CurriculumConfig,
    output_dir: str,
    *,
    curriculum_parameters: Optional[Mapping[str, Any]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "curriculum_config.json")
    output_data = dict(curriculum_parameters) if curriculum_parameters is not None else curriculum_config.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)
        f.write("\n")
    return output_path
