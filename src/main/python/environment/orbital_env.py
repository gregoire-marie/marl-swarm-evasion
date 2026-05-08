from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
import numpy as np
from astropy.time import Time, TimeDelta
from astropy import units as u

from src.main.python.agents.satellite_agent import SatelliteAgent
from src.main.python.environment.reward_engine import compute_rewards
from src.main.python.orbital_meca.orbits import compute_eci_distance
from src.main.python.utils.constants import DEFAULT_START_TIME
from src.main.python.utils.random import set_global_seed

SUPPORTED_MANEUVER_FRAMES = {"ECI", "TNW"}


class OrbitalEnv(ParallelEnv):
    """
    A PettingZoo-compatible orbital dynamics environment for MARL satellite agents.

    This parallel environment simulates orbital propagation and delta-v maneuvers for
    satellite agents using Keplerian dynamics. Each agent can act independently with a
    3D continuous delta-v vector in a configurable frame ("ECI" or "TNW").
    Designed for MARL training with RLlib.

    Unit conventions
    ----------------
    - External RL interface (actions/observations): plain numpy float arrays
      • Actions are delta-v components in m/s (floats)
      • Observations are unitless float vectors built from Keplerian elements (converted to floats),
        remaining Δv (m/s), and pairwise distances (m)
    - Internals (physics): astropy.units.Quantity is used end-to-end for positions, velocities,
      angles, time, and Δv. Conversions to floats happen only at the API edges for RL.

    Attributes:
        agents (List[str]): Active agent IDs in the environment.
        possible_agents (List[str]): All agent IDs (initially same as agents).
        agent_configs (dict): Mapping of agent_id to config dict with orbit and role.
        env_config (dict): Configuration for timestep, episode duration, etc.
        timestep (TimeDelta): Time interval between simulation steps.
        episode_length (int): Number of steps per episode.
        max_delta_v (float): Maximum delta-v magnitude allowed per step [m/s].
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
                  with units [m, one, deg, deg, deg, deg].
                - "init_delta_v" (float or Quantity): Initial Δv budget. If float, interpreted as m/s.
            env_config (dict): Environment parameters including:
                - "timestep_sec" (float): Time step in seconds.
                - "episode_length" (int): Maximum number of steps per episode.
                - "start_time" (str): ISO date for simulation start (UTC).
                - "max_delta_v_mps" (float): Max delta-v allowed per action (in m/s).
                - "maneuver_frame" (str): Maneuver frame for actions, "ECI" or "TNW".
        """
        self.agents = list(agent_configs.keys())
        self.possible_agents = self.agents.copy()
        self.agent_configs = agent_configs
        self.env_config = env_config

        self.timestep = TimeDelta(env_config.get("timestep_sec", 10), format="sec")
        self.episode_length = env_config.get("episode_length", 1000)
        self.max_delta_v = env_config.get("max_delta_v_mps", 100.0)  # m/s
        self.maneuver_frame = str(env_config.get("maneuver_frame", "ECI")).strip().upper()
        if self.maneuver_frame not in SUPPORTED_MANEUVER_FRAMES:
            raise ValueError(
                f"Unsupported maneuver frame '{self.maneuver_frame}'. "
                f"Supported frames: {sorted(SUPPORTED_MANEUVER_FRAMES)}."
            )
        self.freeze_targets = bool(env_config.get("freeze_targets", False))

        self._current_time = None
        self._step_count = 0
        self._agent_states = {}  # agent_id -> SatelliteAgent
        self._seed = None

    def _sanitize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Cast observations to match the declared observation-space dtype.
        """
        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and time.

        Args:
            seed (int, optional): Random seed. When provided, seeds Python, NumPy,
                and Torch (if available) for deterministic behavior.
            options (dict, optional): Additional options for reset (unused).

        Returns:
            Tuple[dict, dict]: 
                - observations (dict): Mapping agent_id → observation (np.ndarray).
                - infos (dict): Empty or containing initial diagnostic info.
        """
        # Apply (optional) seeding for determinism
        if seed is not None:
            self._seed = int(seed)
            set_global_seed(self._seed)

        self._step_count = 0
        self._current_time = Time(self.env_config.get("start_time", DEFAULT_START_TIME), scale="utc")

        # Reset agent states
        self._agent_states = {
            agent_id: SatelliteAgent(agent_id, config, self._current_time)
            for agent_id, config in self.agent_configs.items()
        }

        observations = {}
        for agent_id, agent in self._agent_states.items():
            raw_obs = agent.get_observation(self._agent_states)
            observations[agent_id] = self._sanitize_observation(raw_obs)

        infos = {agent_id: {} for agent_id in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Advance the simulation one timestep using agents' delta-v actions.

        Args:
            actions (dict): Mapping from agent_id → 3D np.ndarray delta-v (in m/s, floats).

        Returns:
            Tuple:
                - observations (dict): agent_id → observation (np.ndarray).
                - rewards (dict): agent_id → float reward.
                - terminations (dict): agent_id → bool indicating episode completion (failure/success).
                - truncations (dict): agent_id → bool indicating episode length limit.
                - infos (dict): agent_id → extra info dict (flags, time, step).
        """
        self._step_count += 1
        self._current_time += self.timestep

        # Apply actions
        for agent_id in self.agents:
            agent = self._agent_states[agent_id]
            if self.freeze_targets and agent.role == "target":
                dv = np.zeros(3, dtype=np.float32)
            else:
                dv_vector = actions.get(agent_id, None)
                if dv_vector is None:
                    dv = np.zeros(3, dtype=np.float32)
                else:
                    dv = np.asarray(dv_vector, dtype=np.float32)
                
                # Action clipping: enforce L2-norm constraint if it exceeds max_delta_v
                norm = np.linalg.norm(dv)
                if norm > self.max_delta_v:
                    dv = (dv / norm) * self.max_delta_v

            agent.apply_action(dv, self._current_time, maneuver_frame=self.maneuver_frame)

        # Propagate all agents to the new current time
        for agent in self._agent_states.values():
            agent.propagate_to(self._current_time)

        # Compute rewards and global flags
        rewards_raw, flags = compute_rewards(self._agent_states)
        # Ensure plain Python floats in rewards dict
        rewards = {aid: float(rewards_raw.get(aid, 0.0)) for aid in self.agents}

        # Observations after state update
        observations = {}
        for agent_id, agent in self._agent_states.items():
            raw_obs = agent.get_observation(self._agent_states)
            observations[agent_id] = self._sanitize_observation(raw_obs)

        # Episode termination and truncation
        # Terminate on any critical flag (collision, intercept, no fuel)
        any_flag = any(flags.values())
        terminations = {agent_id: any_flag for agent_id in self.agents}
        
        # Truncate on episode length
        truncated = self._step_count >= self.episode_length
        truncations = {agent_id: truncated for agent_id in self.agents}

        # Infos: expose flags and basic diagnostics per agent
        infos = {
            agent_id: {
                "flags": flags.copy(),
                "time": self._current_time.isot,
                "step": self._step_count,
            }
            for agent_id in self.agents
        }

        return observations, rewards, terminations, truncations, infos

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
            low=np.full((obs_dim,), -10.0, dtype=np.float32),
            high=np.full((obs_dim,), 10.0, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def action_space(self, agent_id):
        """
        Define the action space for a given agent.

        Actions are 3D delta-v vectors in the selected maneuver frame
        (`env_config["maneuver_frame"]`), bounded by max_delta_v.

        Args:
            agent_id (str): Agent identifier.

        Returns:
            gymnasium.spaces.Box: Bounded 3D continuous action space [m/s].
        """
        # 3D delta-v vector in selected maneuver frame, bounded by max delta-v
        max_dv = np.float32(self.max_delta_v)
        return Box(
            low=np.full((3,), -max_dv, dtype=np.float32),
            high=np.full((3,), max_dv, dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def get_position_m(self, agent_id: str) -> np.ndarray:
        """
        Return the current ECI position of one agent in meters.
        """
        r, _ = self._agent_states[agent_id].orbit_state.get_rv()
        return np.asarray(r.to_value(u.m), dtype=float)

    def get_remaining_delta_v_mps(self, agent_id: str) -> float:
        """
        Return the current remaining delta-v budget of one agent in m/s.
        """
        return float(self._agent_states[agent_id].get_remaining_delta_v().to_value(u.m / u.s))

    def get_orbital_elements(self, agent_id: str) -> dict:
        """
        Return the current Keplerian elements of one agent as plain floats.
        """
        a, e, inc, raan, argp, mean_anomaly = self._agent_states[agent_id].orbit_state.get_keplerian()
        return {
            "a_m": float(a.to_value(u.m)),
            "e": float(e.to_value(u.one) if hasattr(e, "to") else e),
            "i_deg": float(inc.to_value(u.deg)),
            "raan_deg": float(raan.to_value(u.deg)),
            "argp_deg": float(argp.to_value(u.deg)),
            "M_deg": float(mean_anomaly.to_value(u.deg)),
        }

    def get_pairwise_distances_m(self) -> dict:
        """
        Return pairwise distances between all current agents in meters.
        """
        distances = {}
        agent_ids = list(self.agents)
        for i, aid in enumerate(agent_ids):
            for bid in agent_ids[i + 1 :]:
                distances[f"{aid}__{bid}"] = float(
                    compute_eci_distance(
                        self._agent_states[aid].orbit_state,
                        self._agent_states[bid].orbit_state,
                    )
                )
        return distances

    def render(self):
        """
        Print a summary of the environment state and agent orbits.

        Useful for debugging or simple visual inspection during training.
        """
        print(f"Time: {self._current_time.iso}, Step: {self._step_count}")
        for agent_id, agent in self._agent_states.items():
            agent.summary()
