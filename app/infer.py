import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.environment.scenarios import pursuit_evasion_scenario
from src.main.python.utils.helpers import get_logger

logger = get_logger("inference_app")
SUPPORTED_MANEUVER_FRAMES = ("ECI", "TNW")


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
    if "maneuver_frame" in config:
        env_config["maneuver_frame"] = str(config["maneuver_frame"]).upper()
        
    env = OrbitalEnv(agent_configs, env_config)
    return ParallelPettingZooEnv(env)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference app for orbital MARL.")
    parser.add_argument("checkpoint", type=str, help="Path to the RLlib checkpoint.")
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

def main():
    args = parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_env("orbital_env", lambda config: env_creator(config))
    
    # Load algorithm from checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    print(args.checkpoint)
    algo = Algorithm.from_checkpoint(args.checkpoint)
    
    # Instantiate environment for inference
    env_config = {
        "n_interceptors": args.n_interceptors,
        "n_targets": args.n_targets,
        "timestep_sec": args.timestep,
        "episode_length": args.episode_length,
        "freeze_targets": args.freeze_targets,
        "maneuver_frame": args.maneuver_frame,
        "seed": args.seed
    }
    # We use the raw OrbitalEnv for easier data access, but we need to match RLlib's view if needed.
    # Actually, it's better to use the same creator but keep the original env.
    agent_configs, scenario_env_config = pursuit_evasion_scenario(
        n_interceptors=args.n_interceptors,
        n_targets=args.n_targets,
        seed=args.seed
    )
    scenario_env_config["timestep_sec"] = args.timestep
    scenario_env_config["episode_length"] = args.episode_length
    scenario_env_config["freeze_targets"] = args.freeze_targets
    scenario_env_config["maneuver_frame"] = args.maneuver_frame
    
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
            # Determine policy mapping
            if agent_id.startswith("interceptor"):
                policy_id = "interceptor_policy"
            elif agent_id.startswith("target"):
                policy_id = "target_policy"
            else:
                policy_id = "shared_policy"
                
            module = algo.get_module(policy_id)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            # Ensure tensor is on the same device as the module
            device = next(module.parameters()).device
            obs_tensor = obs_tensor.to(device)
            
            with torch.no_grad():
                output = module.forward_inference({"obs": obs_tensor})
            
            if "actions" in output:
                action = output["actions"].cpu().numpy()[0]
            elif "action_dist_inputs" in output:
                # For continuous PPO, action_dist_inputs are often [mean, log_std]
                dist_inputs = output["action_dist_inputs"].cpu().numpy()[0]
                # Assuming the model returns [mean, log_std], we take the mean for inference
                action = dist_inputs[:len(dist_inputs)//2]
            else:
                logger.warning(f"Could not find actions in module output for {agent_id}. Keys: {output.keys()}")
                action = np.zeros(3) # Fallback

            actions[agent_id] = action
            history["actions"][agent_id].append(action)

        observations, rewards, terminations, truncations, step_infos = env.step(actions)
        print(actions, observations)

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
        args.out_dir = f"{args.checkpoint}/inference_results"

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
    R_earth = 6378.137
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
