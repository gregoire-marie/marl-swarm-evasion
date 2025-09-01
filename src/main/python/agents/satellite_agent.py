from astropy.time import Time
from astropy import units as u
import numpy as np

from src.main.python.agents.orbit_state import OrbitState
from src.main.python.orbital_meca.orbits import compute_eci_distance
from src.main.python.utils.helpers import keplerian_to_array, get_logger

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
        delta_v_total (Quantity): Cumulative delta-v applied (km/s).
    """

    def __init__(self, agent_id: str, config: dict, epoch: Time):
        """
        Initializes a satellite agent.

        Args:
            agent_id (str): Unique ID for the agent.
            config (dict): Configuration dictionary containing:
                - "role": either "interceptor" or "target".
                - "init_orbit": Tuple of 6 classical orbital elements as astropy Quantities.
                - "delta_v": Float setting the initial orbital maneuver budget (in seconds).
            epoch (Time): Start time of the simulation.
        """
        self.id = agent_id
        self.role = config["role"]
        self.initial_elements = config["init_orbit"]
        self.max_delta_v_budget = config.get("delta_v", 10.0)  # km/s
        self.epoch = epoch

        self.orbit_state = OrbitState(self.initial_elements, epoch)
        self.delta_v_total = 0.0 * u.km / u.s

    def apply_action(self, dv_vector: np.ndarray, time: Time):
        """
        Applies an instantaneous delta-v maneuver at a given time.

        Args:
            dv_vector (np.ndarray): Delta-v vector in ECI frame, shape (3,), units in km/s.
            time (Time): Time at which the maneuver is performed.
        """
        dv = dv_vector * u.km / u.s
        if np.linalg.norm(dv_vector) > 0:
            log.debug(f"[{self.id}] Applying Δv = {dv_vector} km/s at t={time.iso}")

        self.orbit_state.apply_delta_v(dv, time)
        self.delta_v_total += np.linalg.norm(dv)

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
        remaining_dv = max(0.0, self.max_delta_v_budget - self.delta_v_total.to_value(u.km / u.s))
        obs.append(remaining_dv)

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

    def get_total_delta_v(self) -> u.Quantity:
        """
        Returns the total delta-v applied by the agent so far.

        Returns:
            Quantity: Total delta-v (km/s).
        """
        return self.delta_v_total

    def summary(self):
        """
        Logs a summary of the current orbit in Keplerian form.
        """
        o = self.orbit_state.orbit
        log.info(
            f"[{self.id}] a={o.a:.1f}, e={o.ecc:.4f}, i={o.inc.to(u.deg):.2f}, M={o.M.to(u.deg):.1f}"
        )
