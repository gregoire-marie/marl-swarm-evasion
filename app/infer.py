import os
import argparse
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.environment.scenarios import pursuit_evasion_scenario
from src.main.python.orbital_meca.orbits import compute_eci_distance
from src.main.python.utils.constants import DEFAULT_OBJECTIVES, R_EARTH
from src.main.python.utils.helpers import get_logger, policy_mapping_fn, resolve_checkpoint_path
from astropy import units as u

logger = get_logger("inference_app")
SUPPORTED_MANEUVER_FRAMES = ("ECI", "TNW")


def parse_maneuver_frame(value: str) -> str:
    frame = str(value).strip().upper()
    if frame not in SUPPORTED_MANEUVER_FRAMES:
        raise argparse.ArgumentTypeError(
            f"Unsupported maneuver frame '{value}'. Supported frames: {list(SUPPORTED_MANEUVER_FRAMES)}."
        )
    return frame

def parse_args():
    parser = argparse.ArgumentParser(description="Inference app for orbital MARL.")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a run directory containing RLlib checkpoint_* folders, or directly to one of these specific checkpoints.",
    )
    parser.add_argument("--n-interceptors", type=int, default=1, help="Number of interceptor agents.")
    parser.add_argument("--n-targets", type=int, default=1, help="Number of target agents.")
    parser.add_argument("--timestep", type=float, default=60.0, help="Simulation timestep in seconds.")
    parser.add_argument("--episode-length", type=int, default=100, help="Number of steps per episode.")
    parser.add_argument("--start-time", type=str, default="2025-01-01 00:00:00", help="Date of start of inference episodes.")
    parser.add_argument("--max-delta-v-kms", type=float, default=0.02, help="The maximum single maneuver delta-v in km/s.")
    parser.add_argument(
        "--freeze-targets",
        action="store_true",
        help="Force target agents to apply zero delta-v at each step.",
    )
    parser.add_argument(
        "--maneuver-frame",
        type=parse_maneuver_frame,
        default="ECI",
        help="Action frame for maneuvers: ECI or TNW.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the scenario.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Compatibility flag (unused in single-episode local inference).",
    )
    parser.add_argument("--out-dir", type=str, required=False, help="Directory to save plots.")
    return parser.parse_args()

def _validate_args(args: argparse.Namespace) -> None:
    if args.n_interceptors <= 0:
        raise ValueError("--n-interceptors must be >= 1.")
    if args.n_targets <= 0:
        raise ValueError("--n-targets must be >= 1.")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be >= 1.")
    if args.max_delta_v_kms <= 0:
        raise ValueError("--max-delta-v-kms must be > 0.")
    if args.timestep <= 0:
        raise ValueError("--timestep must be > 0.")


def _rllib_env_creator(config: Dict[str, Any]) -> ParallelPettingZooEnv:
    agent_configs = pursuit_evasion_scenario(
        n_interceptors=int(config["n_interceptors"]),
        n_targets=int(config["n_targets"]),
        seed=int(config["seed"]),
    )
    return ParallelPettingZooEnv(
        OrbitalEnv(
            agent_configs=agent_configs,
            env_config=config["orbital_env_config"],
        )
    )


def _pairwise_distances(env: OrbitalEnv, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    distances: Dict[str, float] = {}
    for aid, bid in pairs:
        key = f"{aid}__{bid}"
        distances[key] = float(
            compute_eci_distance(
                env._agent_states[aid].orbit_state,  # noqa: SLF001 - intentional telemetry access for plotting
                env._agent_states[bid].orbit_state,  # noqa: SLF001 - intentional telemetry access for plotting
            )
        )
    return distances


def _position_km(env: OrbitalEnv, agent_id: str) -> np.ndarray:
    r, _ = env._agent_states[agent_id].orbit_state.get_rv()  # noqa: SLF001 - intentional telemetry access for plotting
    return np.asarray(r.to_value(u.km), dtype=float)


def setup_inference(args: argparse.Namespace, checkpoint_path: str) -> Dict[str, Any]:
    """
    Setup all that is needed for inference.
    """
    _validate_args(args)

    env_name = OrbitalEnv.metadata.get("name", "orbital_env_v0")
    register_env(env_name, _rllib_env_creator)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algo = Algorithm.from_checkpoint(checkpoint_path)

    agent_configs = pursuit_evasion_scenario(
        n_interceptors=args.n_interceptors,
        n_targets=args.n_targets,
        seed=args.seed,
    )
    orbital_env_config = {
        "timestep_sec": args.timestep,
        "episode_length": args.episode_length,
        "start_time": args.start_time,
        "max_delta_v_kms": args.max_delta_v_kms,
        "maneuver_frame": args.maneuver_frame,
        "freeze_targets": args.freeze_targets,
    }
    env = OrbitalEnv(agent_configs=agent_configs, env_config=orbital_env_config)

    return {
        "args": args,
        "checkpoint_path": checkpoint_path,
        "algo": algo,
        "env": env,
    }

def launch_inference(inference_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform the inference.
    """
    args: argparse.Namespace = inference_ctx["args"]
    env: OrbitalEnv = inference_ctx["env"]
    algo: Algorithm = inference_ctx["algo"]

    observations, _ = env.reset(seed=args.seed)
    agent_ids = list(env.agents)
    pair_ids = list(combinations(agent_ids, 2))

    times_min: List[float] = [0.0]
    step_times_min: List[float] = []
    trajectories_km: Dict[str, List[np.ndarray]] = {aid: [_position_km(env, aid)] for aid in agent_ids}
    rewards_hist: Dict[str, List[float]] = {aid: [] for aid in agent_ids}
    fuel_hist: Dict[str, List[float]] = {
        aid: [float(env._agent_states[aid].get_remaining_delta_v().to_value(u.km / u.s))]  # noqa: SLF001
        for aid in agent_ids
    }
    action_mag_hist: Dict[str, List[float]] = {aid: [] for aid in agent_ids}
    distances_hist: Dict[str, List[float]] = {f"{aid}__{bid}": [] for aid, bid in pair_ids}
    for pair_key, distance in _pairwise_distances(env, pair_ids).items():
        distances_hist[pair_key].append(distance)

    terminated = False
    truncated = False
    final_flags: Dict[str, bool] = {}

    step_count = 0
    while step_count < args.episode_length and not (terminated or truncated):
        actions: Dict[str, np.ndarray] = {}
        for aid in agent_ids:
            policy_id = policy_mapping_fn(aid)
            action = algo.compute_single_action(observations[aid], policy_id=policy_id, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_vec.shape[0] != 3:
                raise ValueError(f"Expected 3D action for {aid}, got shape {action_vec.shape}.")
            action_vec = action_vec[:3]
            actions[aid] = action_vec

            applied_action = action_vec.copy()
            if args.freeze_targets and aid.startswith("target"):
                applied_action = np.zeros(3, dtype=np.float32)
            norm = float(np.linalg.norm(applied_action))
            if norm > args.max_delta_v_kms and norm > 0.0:
                norm = args.max_delta_v_kms
            action_mag_hist[aid].append(norm)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1

        t_min = (step_count * args.timestep) / 60.0
        times_min.append(t_min)
        step_times_min.append(t_min)

        for aid in agent_ids:
            rewards_hist[aid].append(float(rewards.get(aid, 0.0)))
            trajectories_km[aid].append(_position_km(env, aid))
            fuel_hist[aid].append(float(env._agent_states[aid].get_remaining_delta_v().to_value(u.km / u.s)))  # noqa: SLF001

        for pair_key, distance in _pairwise_distances(env, pair_ids).items():
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
        "remaining_delta_v_kms": final_remaining_fuel,
        "final_pairwise_distances_km": final_distances,
    }

    return {
        "args": args,
        "agent_ids": agent_ids,
        "pair_ids": pair_ids,
        "times_min": times_min,
        "step_times_min": step_times_min,
        "trajectories_km": trajectories_km,
        "rewards": rewards_hist,
        "fuel_kms": fuel_hist,
        "action_magnitudes_kms": action_mag_hist,
        "distances_km": distances_hist,
        "report": report,
    }

def plot_inference(plot_save_dir: str, kargs):
    """
    Plot the inference results.
    """
    args: argparse.Namespace = kargs["args"]
    agent_ids: List[str] = kargs["agent_ids"]
    times_min: List[float] = kargs["times_min"]
    step_times_min: List[float] = kargs["step_times_min"]
    trajectories_km: Dict[str, List[np.ndarray]] = kargs["trajectories_km"]
    rewards_hist: Dict[str, List[float]] = kargs["rewards"]
    fuel_hist: Dict[str, List[float]] = kargs["fuel_kms"]
    action_mag_hist: Dict[str, List[float]] = kargs["action_magnitudes_kms"]
    distances_hist: Dict[str, List[float]] = kargs["distances_km"]

    # 1. 3D Orbital Trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for aid in agent_ids:
        traj = np.asarray(trajectories_km[aid], dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=aid)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker="o", s=18)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], marker="x", s=24)

    # Plot Earth for reference
    u_sphere, v_sphere = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    R_earth = float(R_EARTH / (1 * u.km))
    x_earth = R_earth * np.cos(u_sphere) * np.sin(v_sphere)
    y_earth = R_earth * np.sin(u_sphere) * np.sin(v_sphere)
    z_earth = R_earth * np.cos(v_sphere)
    ax.plot_surface(x_earth, y_earth, z_earth, color="blue", alpha=0.1)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
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

    ax2.set_ylabel("Fuel (km/s)")
    ax2.set_title("Remaining Fuel over Time")
    ax2.legend()
    ax2.grid(True)

    ax3.set_ylabel("Action Mag (km/s)")
    ax3.set_xlabel("Time (min)")
    ax3.set_title("Action Magnitudes over Time")
    ax3.axhline(y=args.max_delta_v_kms, color="k", linestyle="--", label="Max Δv Limit")
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
        y=DEFAULT_OBJECTIVES["collision_distance_km"],
        color="r",
        linestyle="--",
        label="Collision Threshold",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Distance (km)")
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
