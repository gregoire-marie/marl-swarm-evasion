from astropy.time import Time
from astropy import units as u
import numpy as np

from src.main.python.agents.orbit_state import OrbitState
from src.main.python.orbital_meca.orbits import compute_eci_distance
from src.main.python.utils.helpers import keplerian_to_array, get_logger, delta_v_norm

log = get_logger("SatelliteAgent")

class SatelliteAgent:
    """
    Represents a satellite agent in the environment (interceptor or target).

    This class handles the agent's orbital state, propagates it, applies delta-v maneuvers,
    and provides observations for reinforcement learning tasks.

    Attributes:
        id (str): Unique identifier for the agent.
        role (str): Agent role, either 'interceptor' or 'target'.
        initial_elements (tuple): Initial Keplerian elements (a, e, i, RAAN, argp, M).
        epoch (Time): Initial epoch of the simulation.
        orbit_state (OrbitState): Current orbital state.
        used_delta_v (Quantity): Cumulative delta-v applied (km/s).
    """

    def __init__(self, agent_id: str, config: dict, epoch: Time):
        """
        Initializes a satellite agent.

        Args:
            agent_id (str): Unique ID for the agent.
            config (dict): Configuration dictionary containing:
                - "role": String either "interceptor" or "target".
                - "init_orbit": Tuple of 6 Keplerian orbital elements as astropy Quantities.
                - "init_delta_v": Astropy quantity (in km/s) setting the initial orbital maneuver budget.
            epoch (Time): Start time of the simulation.
        """
        self.id = agent_id
        self.role = config["role"]
        self.initial_elements = config["init_orbit"]
        # Normalize init_delta_v to Quantity[km/s] even if provided as plain float
        init_dv_val = config.get("init_delta_v", 0.0)
        if hasattr(init_dv_val, "to"):
            self.init_delta_v = init_dv_val.to(u.km / u.s)
        else:
            self.init_delta_v = float(init_dv_val) * u.km / u.s  # km/s
        self.epoch = epoch

        self.orbit_state = OrbitState(self.initial_elements, epoch)
        self.used_delta_v = 0.0 * u.km / u.s

    def apply_action(self, dv_vector: np.ndarray, time: Time):
        """
        Applies an instantaneous delta-v maneuver at a given time.

        Args:
            dv_vector (np.ndarray): Delta-v vector in ECI frame, shape (3,), values in km/s (floats).
            time (Time): Time at which the maneuver is performed.
        """
        dv = dv_vector * u.km / u.s  # Quantity[km/s]
        if np.linalg.norm(dv_vector) > 0:
            log.debug(f"[{self.id}] Applying Δv = {dv_vector} km/s at t={time.iso}")

        self.orbit_state.apply_delta_v(dv, time)
        # Accumulate used Δv as a Quantity[km/s] to preserve unit consistency
        self.used_delta_v += delta_v_norm(dv)

    def propagate_to(self, time: Time):
        """
        Propagates the orbit to a new epoch.

        Args:
            time (Time): Future time to which the orbit is propagated.
        """
        self.orbit_state.propagate_to(time)

    def get_observation(self, other_agents: dict) -> np.ndarray:
        """
        Constructs the observation vector for the agent.

        The observation includes:
        - Own Keplerian elements (6D)
        - Own fuel remaining (1D)
        - For each other agent (regardless of role):
            - Their Keplerian elements (6D)
            - Their relative distance (1D)

        Total observation size: 7 + (N-1) × 7 = 7N

        Args:
            other_agents (dict): Mapping of agent_id → SatelliteAgent

        Returns:
            np.ndarray: Flat observation vector (float32)
        """

        obs = []

        # --- Own state ---
        own_kep = keplerian_to_array(self.orbit_state.orbit)  # (6,)
        obs.extend(own_kep)
        # Remaining delta-v gauge as float (km/s)
        remaining_dv_value = self.init_delta_v.to_value(u.km / u.s) - self.used_delta_v.to_value(u.km / u.s)
        remaining_dv_value = max(0.0, float(remaining_dv_value))
        obs.append(remaining_dv_value)

        # --- Other agents (all roles) ---
        for other_id, other in sorted(other_agents.items()):
            if other_id == self.id:
                continue
            other_kep = keplerian_to_array(other.orbit_state.orbit)
            distance = compute_eci_distance(self.orbit_state, other.orbit_state)  # km
            obs.extend(other_kep)
            obs.append(distance)

        return np.array(obs, dtype=np.float32)

    def get_observation_v1(self, other_agents: dict) -> np.ndarray:
        """
        Constructs the observation vector for the agent.

        The observation includes the agent's own Keplerian elements and those of all other agents.
        Closest approach info and uncertainty are not yet included.

        Args:
            other_agents (dict): Mapping from agent_id to SatelliteAgent.

        Returns:
            np.ndarray: Flattened observation vector (float32).
        """
        own_kep = keplerian_to_array(self.orbit_state.orbit)
        obs = [own_kep]

        for other_id, other in other_agents.items():
            if other_id == self.id:
                continue
            other_kep = keplerian_to_array(other.orbit_state.orbit)
            obs.append(other_kep)

            # TODO: Add closest approach data and covariance

        return np.concatenate(obs, dtype=np.float32)

    def get_remaining_delta_v(self) -> u.Quantity:
        """
        Returns the amount of remaining delta-v in the agent.

        Returns:
            Quantity: Remaining delta-v (km/s).
        """
        return self.init_delta_v - self.get_used_delta_v()

    def get_used_delta_v(self) -> u.Quantity:
        """
        Returns the total delta-v applied by the agent so far.

        Returns:
            Quantity: Total delta-v (km/s).
        """
        return self.used_delta_v

    def summary(self):
        """
        Logs a summary of the current orbit in Keplerian form.
        """
        o = self.orbit_state.orbit
        log.info(
            f"[{self.id}] a={o.a:.1f}, e={o.ecc:.4f}, i={o.inc.to(u.deg):.2f}, M={o.M.to(u.deg):.1f}"
        )
