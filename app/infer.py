import os
import argparse
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.rllib.algorithms.algorithm import Algorithm

from src.main.python.utils.constants import DEFAULT_OBJECTIVES, R_EARTH
from src.main.python.utils.helpers import get_logger, policy_mapping_fn, resolve_checkpoint_path
from src.main.python.utils.rllib_setup import (
    compute_deterministic_module_action,
    create_raw_env,
    load_run_parameters_from_checkpoint,
    parse_maneuver_frame,
    register_orbital_env,
    run_spec_from_args,
    run_spec_from_rllib_env_config,
)
from astropy import units as u

logger = get_logger("inference_app")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference app for orbital MARL.")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a run directory containing RLlib checkpoint_* folders, or directly to one of these specific checkpoints.",
    )
    parser.add_argument("--n-interceptors", type=int, default=None, help="Number of interceptor agents. Defaults to the checkpoint config.")
    parser.add_argument("--n-targets", type=int, default=None, help="Number of target agents. Defaults to the checkpoint config.")
    parser.add_argument("--timestep", type=float, default=None, help="Simulation timestep in seconds. Defaults to the checkpoint config.")
    parser.add_argument("--episode-length", type=int, default=None, help="Number of steps per episode. Defaults to the checkpoint config.")
    parser.add_argument("--start-time", type=str, default=None, help="Simulation start time (UTC). Defaults to the checkpoint config.")
    parser.add_argument("--max-delta-v-mps", type=float, default=None, help="The maximum single maneuver delta-v in m/s. Defaults to the checkpoint config.")
    parser.add_argument(
        "--freeze-targets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether target agents must apply zero delta-v at each step. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--maneuver-frame",
        type=parse_maneuver_frame,
        default=None,
        help="Action frame for maneuvers: ECI or TNW. Defaults to the checkpoint config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the scenario. Defaults to the checkpoint config.")
    parser.add_argument("--out-dir", type=str, required=False, help="Directory to save plots.")
    return parser.parse_args()

def setup_inference(args: argparse.Namespace, checkpoint_path: str) -> Dict[str, Any]:
    """
    Setup all that is needed for inference.
    """
    register_orbital_env()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algo = Algorithm.from_checkpoint(checkpoint_path)
    rllib_checkpoint_spec = run_spec_from_rllib_env_config(algo.config.env_config)
    metadata_spec, run_parameters_path = load_run_parameters_from_checkpoint(
        checkpoint_path,
        base_spec=rllib_checkpoint_spec,
    )
    checkpoint_spec = metadata_spec or rllib_checkpoint_spec
    spec = run_spec_from_args(args, base_spec=checkpoint_spec)
    env = create_raw_env(spec)

    return {
        "args": args,
        "spec": spec,
        "run_parameters_path": run_parameters_path,
        "checkpoint_path": checkpoint_path,
        "algo": algo,
        "env": env,
    }

def launch_inference(inference_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform the inference.
    """
    spec = inference_ctx["spec"]
    env = inference_ctx["env"]
    algo: Algorithm = inference_ctx["algo"]

    observations, _ = env.reset(seed=spec.seed)
    agent_ids = list(env.agents)
    policy_ids = {aid: policy_mapping_fn(aid) for aid in agent_ids}
    modules = {policy_id: algo.get_module(policy_id) for policy_id in set(policy_ids.values())}
    missing_modules = [policy_id for policy_id, module in modules.items() if module is None]
    if missing_modules:
        raise KeyError(f"Missing RLModule(s) in restored checkpoint: {missing_modules}")
    module_states = {
        aid: modules[policy_id].get_initial_state()
        for aid, policy_id in policy_ids.items()
        if modules[policy_id].is_stateful()
    }
    initial_distances = env.get_pairwise_distances_m()

    times_min: List[float] = [0.0]
    step_times_min: List[float] = []
    trajectories_m: Dict[str, List[np.ndarray]] = {aid: [env.get_position_m(aid)] for aid in agent_ids}
    rewards_hist: Dict[str, List[float]] = {aid: [] for aid in agent_ids}
    fuel_hist: Dict[str, List[float]] = {aid: [env.get_remaining_delta_v_mps(aid)] for aid in agent_ids}
    action_mag_hist: Dict[str, List[float]] = {aid: [] for aid in agent_ids}
    distances_hist: Dict[str, List[float]] = {pair_key: [] for pair_key in initial_distances}
    for pair_key, distance in initial_distances.items():
        distances_hist[pair_key].append(distance)

    terminated = False
    truncated = False
    final_flags: Dict[str, bool] = {}

    step_count = 0
    while step_count < spec.episode_length and not (terminated or truncated):
        actions: Dict[str, np.ndarray] = {}
        for aid in agent_ids:
            policy_id = policy_ids[aid]
            action, next_state = compute_deterministic_module_action(
                modules[policy_id],
                observations[aid],
                normalize_actions=bool(algo.config.normalize_actions),
                clip_actions=bool(algo.config.clip_actions),
                module_state=module_states.get(aid),
            )
            if next_state is not None:
                module_states[aid] = next_state
            action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_vec.shape[0] != 3:
                raise ValueError(f"Expected 3D action for {aid}, got shape {action_vec.shape}.")
            action_vec = action_vec[:3]
            actions[aid] = action_vec

            applied_action = action_vec.copy()
            if spec.freeze_targets and aid.startswith("target"):
                applied_action = np.zeros(3, dtype=np.float32)
            norm = float(np.linalg.norm(applied_action))
            if norm > spec.max_delta_v_mps and norm > 0.0:
                norm = spec.max_delta_v_mps
            action_mag_hist[aid].append(norm)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        t_min = (step_count * spec.timestep) / 60.0
        times_min.append(t_min)
        step_times_min.append(t_min)

        for aid in agent_ids:
            rewards_hist[aid].append(float(rewards.get(aid, 0.0)))
            trajectories_m[aid].append(env.get_position_m(aid))
            fuel_hist[aid].append(env.get_remaining_delta_v_mps(aid))

        for pair_key, distance in env.get_pairwise_distances_m().items():
            distances_hist[pair_key].append(distance)

        terminated = bool(any(terminations.values())) if terminations else False
        truncated = bool(any(truncations.values())) if truncations else False

        if infos:
            any_info = next(iter(infos.values()))
            final_flags = dict(any_info.get("flags", {}))

    total_rewards = {aid: float(np.sum(rewards_hist[aid])) for aid in agent_ids}
    final_remaining_fuel = {aid: fuel_hist[aid][-1] for aid in agent_ids}
    final_distances = {pair_key: hist[-1] for pair_key, hist in distances_hist.items() if hist}

    report = {
        "steps": step_count,
        "terminated": terminated,
        "truncated": truncated,
        "flags": final_flags,
        "total_rewards": total_rewards,
        "remaining_delta_v_mps": final_remaining_fuel,
        "final_pairwise_distances_m": final_distances,
    }

    return {
        "spec": spec,
        "agent_ids": agent_ids,
        "times_min": times_min,
        "step_times_min": step_times_min,
        "trajectories_m": trajectories_m,
        "rewards": rewards_hist,
        "fuel_mps": fuel_hist,
        "action_magnitudes_mps": action_mag_hist,
        "distances_m": distances_hist,
        "report": report,
    }

def plot_inference(plot_save_dir: str, kargs):
    """
    Plot the inference results.
    """
    spec = kargs["spec"]
    agent_ids: List[str] = kargs["agent_ids"]
    times_min: List[float] = kargs["times_min"]
    step_times_min: List[float] = kargs["step_times_min"]
    trajectories_m: Dict[str, List[np.ndarray]] = kargs["trajectories_m"]
    rewards_hist: Dict[str, List[float]] = kargs["rewards"]
    fuel_hist: Dict[str, List[float]] = kargs["fuel_mps"]
    action_mag_hist: Dict[str, List[float]] = kargs["action_magnitudes_mps"]
    distances_hist: Dict[str, List[float]] = kargs["distances_m"]

    # 1. 3D Orbital Trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for aid in agent_ids:
        traj = np.asarray(trajectories_m[aid], dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=aid)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker="o", s=18)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], marker="x", s=24)

    # Plot Earth for reference
    u_sphere, v_sphere = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    R_earth = float(R_EARTH / (1 * u.m))
    x_earth = R_earth * np.cos(u_sphere) * np.sin(v_sphere)
    y_earth = R_earth * np.sin(u_sphere) * np.sin(v_sphere)
    z_earth = R_earth * np.cos(v_sphere)
    ax.plot_surface(x_earth, y_earth, z_earth, color="blue", alpha=0.1)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Orbital Trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_save_dir, "trajectories_3d.png"))
    plt.close(fig)

    # 2. Rewards, Fuel and Action Magnitudes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for aid in agent_ids:
        ax1.plot(step_times_min, rewards_hist[aid], label=aid)
        ax2.plot(times_min, fuel_hist[aid], label=aid)
        ax3.plot(step_times_min, action_mag_hist[aid], label=aid)

    ax1.set_ylabel("Reward")
    ax1.set_title("Rewards over Time")
    ax1.legend()
    ax1.grid(True)

    ax2.set_ylabel("Fuel (m/s)")
    ax2.set_title("Remaining Fuel over Time")
    ax2.legend()
    ax2.grid(True)

    ax3.set_ylabel("Action Mag (m/s)")
    ax3.set_xlabel("Time (min)")
    ax3.set_title("Action Magnitudes over Time")
    ax3.axhline(y=spec.max_delta_v_mps, color="k", linestyle="--", label="Max Δv Limit")
    ax3.legend()
    ax3.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_save_dir, "metrics_over_time.png"))
    plt.close(fig)

    # 3. Distances
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    if distances_hist:
        for pair_key, pair_distances in distances_hist.items():
            pair_label = pair_key.replace("__", " <-> ")
            dist_values = np.maximum(np.asarray(pair_distances, dtype=float), 1e-6)
            ax.plot(times_min, dist_values, label=pair_label)
    else:
        ax.text(0.5, 0.5, "No pairwise distances (single-agent case).", ha="center", va="center")

    ax.axhline(
        y=DEFAULT_OBJECTIVES["collision_distance_m"],
        color="r",
        linestyle="--",
        label="Collision Threshold",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Relative Distances (Log Scale)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_save_dir, "distances.png"))
    plt.close(fig)

def teardown_inference(inference_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tear down everything that need to be cleaned after the inference.
    """
    algo = inference_ctx.get("algo")
    if algo is not None:
        algo.stop()
    if ray.is_initialized():
        ray.shutdown()
    return inference_ctx

def main():
    args = parse_args()

    # Get the checkpoint path
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    logger.info(f"Using checkpoint: {checkpoint_path}")

    inference_ctx = {}
    try:
        # Set up the inference
        inference_ctx = setup_inference(args=args, checkpoint_path=checkpoint_path)
        if inference_ctx.get("run_parameters_path"):
            logger.info(f"Loaded run parameters: {inference_ctx['run_parameters_path']}")
        logger.info(f"Using inference spec: {inference_ctx['spec']}")

        logger.info("Starting simulation...")

        inference_data = launch_inference(inference_ctx=inference_ctx)
        inference_report = inference_data["report"]
        logger.info(f"Simulation finished. Report:\n{inference_report}")

        # Get the save directory for plots
        if args.out_dir is None:
            plot_save_dir = os.path.join(checkpoint_path, "inference")
        else:
            plot_save_dir = os.path.abspath(os.path.expanduser(args.out_dir))

        # Create the save directory for plots if it does not exist
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
            logger.info(f"Created output directory: {plot_save_dir}")

        # --- Plotting ---

        plot_inference(plot_save_dir=plot_save_dir, kargs=inference_data)

        logger.info(f"Plots saved to {plot_save_dir}")

    finally:
        inference_ctx = teardown_inference(inference_ctx)

if __name__ == "__main__":
    main()
