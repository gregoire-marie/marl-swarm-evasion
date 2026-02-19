import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unsquash_action, clip_action

from main.python.utils.constants import R_EARTH
from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.environment.scenarios import pursuit_evasion_scenario
from src.main.python.utils.helpers import get_logger, policy_mapping_fn
from astropy import units as u

logger = get_logger("inference_app")
SUPPORTED_MANEUVER_FRAMES = ("ECI", "TNW")
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint_(\d+)$")
torch, _ = try_import_torch()


def parse_maneuver_frame(value: str) -> str:
    frame = str(value).strip().upper()
    if frame not in SUPPORTED_MANEUVER_FRAMES:
        raise argparse.ArgumentTypeError(
            f"Unsupported maneuver frame '{value}'. Supported frames: {list(SUPPORTED_MANEUVER_FRAMES)}."
        )
    return frame

def env_creator(config):
    """
    Creates and wraps the orbital environment for RLlib.
    """
    agent_configs = pursuit_evasion_scenario(
        n_interceptors=config.get("n_interceptors", 1),
        n_targets=config.get("n_targets", 1),
        seed=config.get("seed", 42)
    )
    env_config = {
        "timestep_sec": config.get("timestep_sec", 60.0),
        "episode_length": config.get("episode_length", 100),
        "start_time": config.get("start_time", "2025-01-01 00:00:00"),
        "max_delta_v_kms": config.get("max_delta_v_kms", 0.02),
        "freeze_targets": bool(config.get("freeze_targets", False)),
        "maneuver_frame": str(config.get("maneuver_frame", "ECI")).upper(),
    }

    env = OrbitalEnv(agent_configs, env_config)
    return ParallelPettingZooEnv(env)

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
    parser.add_argument("--out-dir", type=str, required=False, help="Directory to save plots.")
    return parser.parse_args()

def resolve_checkpoint_path(path: str) -> str:
    """
    Resolve a checkpoint input to a concrete RLlib checkpoint directory.
    """
    candidate_path = os.path.normpath(os.path.abspath(os.path.expanduser(path)))

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate_path}")
    if not os.path.isdir(candidate_path):
        raise ValueError(f"Checkpoint path must be a directory: {candidate_path}")

    if CHECKPOINT_DIR_PATTERN.fullmatch(os.path.basename(candidate_path)):
        return candidate_path

    entries = set(os.listdir(candidate_path))

    # If this dir itself already contains checkpoint state files, accept it.
    if (
        "rllib_checkpoint.json" in entries
        or any(
            name.startswith("algorithm_state.")
            and name.split(".")[-1] in {"pkl", "msgpack", "msgpck"}
            for name in entries
        )
        or any(re.fullmatch(r"checkpoint-\d+", name) for name in entries)
    ):
        return candidate_path

    # Otherwise, resolve a trial directory to its latest checkpoint_* subdirectory.
    checkpoint_candidates = []
    for name in entries:
        match = CHECKPOINT_DIR_PATTERN.fullmatch(name)
        if not match:
            continue
        checkpoint_dir = os.path.join(candidate_path, name)
        if os.path.isdir(checkpoint_dir):
            checkpoint_candidates.append((int(match.group(1)), checkpoint_dir))

    if not checkpoint_candidates:
        raise ValueError(
            "No RLlib checkpoint found. Provide either a checkpoint directory "
            f"(checkpoint_XXXXXX) or a trial directory containing checkpoint_* folders: {candidate_path}"
        )

    checkpoint_candidates.sort(key=lambda x: x[0], reverse=True)
    resolved_path = checkpoint_candidates[0][1]
    logger.info(
        f"Resolved trial directory to latest checkpoint: {resolved_path} "
        f"(from input: {candidate_path})"
    )
    return resolved_path

def compute_deterministic_action(algo: Algorithm, policy_id: str, obs: np.ndarray):
    """
    Compute a deterministic action.
    """
    module = algo.get_module(policy_id)

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    first_param = next(module.parameters(), None)
    if first_param is not None:
        obs_tensor = obs_tensor.to(first_param.device)

    with torch.no_grad():
        output = module.forward_inference({Columns.OBS: obs_tensor})

    if Columns.ACTIONS in output:
        action = output[Columns.ACTIONS]
    elif Columns.ACTION_DIST_INPUTS in output:
        action_dist_class = module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(output[Columns.ACTION_DIST_INPUTS])
        action = action_dist.to_deterministic().sample()
    else:
        raise ValueError(
            f"Policy '{policy_id}' output has neither '{Columns.ACTIONS}' nor "
            f"'{Columns.ACTION_DIST_INPUTS}'. Keys: {list(output.keys())}"
        )

    action = convert_to_numpy(action)[0]

    # Match RLlib module-to-env connector behavior.
    if algo.config.normalize_actions:
        action = unsquash_action(action, module.action_space)
    elif algo.config.clip_actions:
        action = clip_action(action, module.action_space)

    return action


def main():
    args = parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    register_env("orbital_env", lambda config: env_creator(config))

    # Load algorithm from checkpoint
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # Instantiate raw environment for inference and direct state access.
    agent_configs = pursuit_evasion_scenario(
        n_interceptors=args.n_interceptors,
        n_targets=args.n_targets,
        seed=args.seed
    )
    scenario_env_config = {
        "timestep_sec": args.timestep,
        "episode_length": args.episode_length,
        "start_time": "2025-01-01 00:00:00",
        "max_delta_v_kms": 0.02,
        "freeze_targets": args.freeze_targets,
        "maneuver_frame": args.maneuver_frame,
    }
    env = OrbitalEnv(agent_configs, scenario_env_config)
    logger.info(f"Using maneuver frame: {args.maneuver_frame}")

    observations, infos = env.reset(seed=args.seed)

    # Data collection
    history = {
        "time": [],
        "rewards": {agent_id: [] for agent_id in env.agents},
        "fuel": {agent_id: [] for agent_id in env.agents},
        "positions": {agent_id: [] for agent_id in env.agents},
        "actions": {agent_id: [] for agent_id in env.agents},
        "distances": [], # Distances between all pairs
        "flags": []
    }

    terminated = False
    truncated = False
    step = 0

    logger.info("Starting simulation...")

    while not terminated and not truncated:
        actions = {}
        for agent_id, obs in observations.items():
            policy_id = policy_mapping_fn(agent_id)
            action = compute_deterministic_action(algo, policy_id, obs)

            actions[agent_id] = action
            history["actions"][agent_id].append(action)

        observations, rewards, terminations, truncations, step_infos = env.step(actions)

        # Record data
        history["time"].append(step * args.timestep)
        for agent_id in env.agents:
            history["rewards"][agent_id].append(rewards[agent_id])
            agent = env._agent_states[agent_id]
            history["fuel"][agent_id].append(agent.get_remaining_delta_v().to_value("km/s"))
            r, v = agent.orbit_state.get_rv()
            history["positions"][agent_id].append(r.to_value("km"))

        # Record flags (pick from any agent info)
        any_agent_id = next(iter(env.agents))
        history["flags"].append(step_infos[any_agent_id]["flags"])

        # Distances between all pairs for this step
        step_distances = {}
        agents_list = sorted(env.agents)
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                aid1, aid2 = agents_list[i], agents_list[j]
                p1 = history["positions"][aid1][-1]
                p2 = history["positions"][aid2][-1]
                dist = np.linalg.norm(p1 - p2)
                step_distances[f"{aid1}_vs_{aid2}"] = dist
        history["distances"].append(step_distances)

        terminated = any(terminations.values())
        truncated = any(truncations.values())
        step += 1

    logger.info(f"Simulation finished after {step} steps.")

    # Post-processing for plotting
    time_axis = np.array(history["time"]) / 60.0 # to minutes

    if (not "out_dir" in args) or (args.out_dir is None):
        args.out_dir = f"{checkpoint_path}/inference_results"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")

    # --- Plotting ---

    # 1. 3D Orbital Trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for agent_id, pos_list in history["positions"].items():
        pos_arr = np.array(pos_list)
        label = agent_id
        color = 'red' if 'interceptor' in agent_id else 'blue'
        ax.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], label=label, color=color)
        # Mark start and end
        ax.scatter(pos_arr[0, 0], pos_arr[0, 1], pos_arr[0, 2], color=color, marker='o')
        ax.scatter(pos_arr[-1, 0], pos_arr[-1, 1], pos_arr[-1, 2], color=color, marker='x')

    # Plot Earth for reference
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    R_earth = float(R_EARTH / (1 * u.km))
    x_earth = R_earth * np.cos(u_sphere) * np.sin(v_sphere)
    y_earth = R_earth * np.sin(u_sphere) * np.sin(v_sphere)
    z_earth = R_earth * np.cos(v_sphere)
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.1)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbital Trajectories')
    ax.legend()
    plt.savefig(os.path.join(args.out_dir, "trajectories_3d.png"))

    # 2. Rewards, Fuel and Action Magnitudes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for agent_id in env.agents:
        color = 'red' if 'interceptor' in agent_id else 'blue'
        ax1.plot(time_axis, history["rewards"][agent_id], label=f"{agent_id} Reward", color=color)
        ax2.plot(time_axis, history["fuel"][agent_id], label=f"{agent_id} Fuel", color=color, linestyle='--')

        act_mags = [np.linalg.norm(a) for a in history["actions"][agent_id]]
        ax3.plot(time_axis, act_mags, label=f"{agent_id} Action Mag", color=color, linestyle=':')

    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards over Time')
    ax1.legend()
    ax1.grid(True)

    ax2.set_ylabel('Fuel (km/s)')
    ax2.set_title('Remaining Fuel over Time')
    ax2.legend()
    ax2.grid(True)

    ax3.set_ylabel('Action Mag (km/s)')
    ax3.set_xlabel('Time (min)')
    ax3.set_title('Action Magnitudes over Time')
    ax3.axhline(y=scenario_env_config.get("max_delta_v_kms", 0.01), color='k', linestyle='--', label='Max Δv Limit')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "metrics_over_time.png"))

    # 3. Distances
    plt.figure(figsize=(10, 6))
    dist_keys = history["distances"][0].keys()
    for key in dist_keys:
        dists = [d[key] for d in history["distances"]]
        plt.plot(time_axis, dists, label=key)

    plt.axhline(y=0.1, color='r', linestyle='--', label='Collision Threshold')
    plt.yscale('log')
    plt.xlabel('Time (min)')
    plt.ylabel('Distance (km)')
    plt.title('Relative Distances (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(args.out_dir, "distances.png"))

    logger.info(f"Plots saved to {args.out_dir}")

    # Final Flags
    last_flags = history["flags"][-1]
    logger.info(f"Final Episode Flags: {last_flags}")

    # plt.show()
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
