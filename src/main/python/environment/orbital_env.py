from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
import numpy as np
from astropy.time import Time, TimeDelta

from src.main.python.agents.satellite_agent import SatelliteAgent
from src.main.python.environment.reward_engine import compute_rewards


class OrbitalEnv(ParallelEnv):
    """
    A PettingZoo-compatible orbital dynamics environment for MARL satellite agents.

    This parallel environment simulates orbital propagation and delta-v maneuvers for
    satellite agents using Keplerian dynamics. Each agent can act independently with a
    3D continuous delta-v vector in ECI frame. Designed for MARL training with RLlib.

    Unit conventions
    ----------------
    - External RL interface (actions/observations): plain numpy float arrays
      • Actions are delta-v components in km/s (floats)
      • Observations are unitless float vectors built from Keplerian elements (converted to floats),
        remaining Δv (km/s), and pairwise distances (km)
    - Internals (physics): astropy.units.Quantity is used end-to-end for positions, velocities,
      angles, time, and Δv. Conversions to floats happen only at the API edges for RL.

    Attributes:
        agents (List[str]): Active agent IDs in the environment.
        possible_agents (List[str]): All agent IDs (initially same as agents).
        agent_configs (dict): Mapping of agent_id to config dict with orbit and role.
        env_config (dict): Configuration for timestep, episode duration, etc.
        timestep (TimeDelta): Time interval between simulation steps.
        episode_length (int): Number of steps per episode.
        max_delta_v (float): Maximum delta-v magnitude allowed per step [km/s].
        _current_time (Time): Current simulation time.
        _step_count (int): Current simulation step index.
        _agent_states (dict): Mapping of agent_id → SatelliteAgent instance.

    Metadata:
        name: "orbital_env_v0"
        render_modes: ["human"]
        is_parallelizable: True
    """
    metadata = {
        "name": "orbital_env_v0",
        "render_modes": ["human"],
        "is_parallelizable": True
    }

    def __init__(self, agent_configs, env_config):
        """
        Initialize the OrbitalEnv simulation environment.

        Args:
            agent_configs (dict): Per-agent configuration, where each entry contains:
                - "role" (str): "interceptor" or "target".
                - "init_orbit" (tuple): Classical elements (a, e, i, RAAN, argp, M) as astropy Quantities
                  with units [km, one, deg, deg, deg, deg].
                - "init_delta_v" (float or Quantity): Initial Δv budget. If float, interpreted as km/s.
            env_config (dict): Environment parameters including:
                - "timestep_sec" (float): Time step in seconds.
                - "episode_length" (int): Maximum number of steps per episode.
                - "start_time" (str): ISO date for simulation start (UTC).
                - "max_delta_v_kms" (float): Max delta-v allowed per action (in km/s).
        """
        self.agents = list(agent_configs.keys())
        self.possible_agents = self.agents.copy()
        self.agent_configs = agent_configs
        self.env_config = env_config

        self.timestep = TimeDelta(env_config.get("timestep_sec", 10), format="sec")
        self.episode_length = env_config.get("episode_length", 1000)
        self.max_delta_v = env_config.get("max_delta_v_kms", 0.1)  # km/s

        self._current_time = None
        self._step_count = 0
        self._agent_states = {}  # agent_id -> SatelliteAgent

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and time.

        Args:
            seed (int, optional): Random seed (unused for now).
            options (dict, optional): Additional options for reset (unused).

        Returns:
            dict: Dictionary mapping agent_id → observation (np.ndarray).
        """
        self._step_count = 0
        self._current_time = Time(self.env_config.get("start_time", "2025-01-01 00:00:00"), scale="utc")

        # Reset agent states
        self._agent_states = {
            agent_id: SatelliteAgent(agent_id, config, self._current_time)
            for agent_id, config in self.agent_configs.items()
        }

        observations = {
            agent_id: agent.get_observation(self._agent_states)
            for agent_id, agent in self._agent_states.items()
        }

        return observations

    def step(self, actions):
        """
        Advance the simulation one timestep using agents' delta-v actions.

        Args:
            actions (dict): Mapping from agent_id → 3D np.ndarray delta-v (in km/s, floats).

        Returns:
            Tuple:
                - observations (dict): agent_id → observation (np.ndarray).
                - rewards (dict): agent_id → float reward.
                - dones (dict): agent_id → bool indicating episode completion.
                - infos (dict): agent_id → extra info dict (flags, time, step).
        """
        self._step_count += 1
        self._current_time += self.timestep

        # Apply actions (clip to action space bounds; default to zero if missing)
        for agent_id in self.agents:
            dv_vector = actions.get(agent_id, None)
            if dv_vector is None:
                dv = np.zeros(3, dtype=np.float32)
            else:
                dv = np.asarray(dv_vector, dtype=np.float32)
            # Clip per configured bounds
            dv = np.clip(dv, -self.max_delta_v, self.max_delta_v)

            agent = self._agent_states[agent_id]
            agent.apply_action(dv, self._current_time)

        # Propagate all agents to the new current time
        for agent in self._agent_states.values():
            agent.propagate_to(self._current_time)

        # Compute rewards and global flags
        rewards_raw, flags = compute_rewards(self._agent_states, self._current_time)
        # Ensure plain Python floats in rewards dict
        rewards = {aid: float(rewards_raw.get(aid, 0.0)) for aid in self.agents}

        # Observations after state update
        observations = {
            agent_id: agent.get_observation(self._agent_states)
            for agent_id, agent in self._agent_states.items()
        }

        # Episode termination: end on episode length or any critical flag
        any_flag = any(flags.values())
        dones = {agent_id: (self._step_count >= self.episode_length) or any_flag for agent_id in self.agents}
        dones["__all__"] = all(dones.values())

        # Infos: expose flags and basic diagnostics per agent
        infos = {
            agent_id: {
                "flags": flags.copy(),
                "time": self._current_time.isot,
                "step": self._step_count,
            }
            for agent_id in self.agents
        }

        return observations, rewards, dones, infos

    def observation_space(self, agent_id):
        """
        Returns the observation space for a single agent.

        Each observation includes:
        - 6 own Keplerian elements
        - 1 fuel level
        - For each other agent:
            - 6 Keplerian elements
            - 1 relative distance

        Total dim: 7 + (N-1) × 7 = 7N
        """
        num_agents = len(self.agents)
        obs_dim = 7 * num_agents

        return Box(
            low=-1e5,
            high=1e5,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def action_space(self, agent_id):
        """
        Define the action space for a given agent.

        Actions are 3D delta-v vectors (in ECI), bounded by max_delta_v.

        Args:
            agent_id (str): Agent identifier.

        Returns:
            gymnasium.spaces.Box: Bounded 3D continuous action space [km/s].
        """
        # 3D delta-v vector in ECI, bounded by max delta-v
        return Box(low=-self.max_delta_v,
                   high=self.max_delta_v,
                   shape=(3,),
                   dtype=np.float32)

    def render(self):
        """
        Print a summary of the environment state and agent orbits.

        Useful for debugging or simple visual inspection during training.
        """
        print(f"Time: {self._current_time.iso}, Step: {self._step_count}")
        for agent_id, agent in self._agent_states.items():
            print(f"{agent_id}: {agent.orbit_state.summary()}")
