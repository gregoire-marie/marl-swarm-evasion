from astropy.time import Time
from astropy import units as u
import numpy as np

from src.main.python.agents.orbit_state import OrbitState
from src.main.python.orbital_meca.orbits import compute_eci_distance
from src.main.python.utils.helpers import keplerian_to_array, get_logger, delta_v_norm
from src.main.python.utils.normalization import A_REF, A_SCALE, DIST_SCALE, FUEL_SCALE, normalize_angle

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
        used_delta_v (Quantity): Cumulative delta-v applied (m/s).
    """

    def __init__(self, agent_id: str, config: dict, epoch: Time):
        """
        Initializes a satellite agent.

        Args:
            agent_id (str): Unique ID for the agent.
            config (dict): Configuration dictionary containing:
                - "role": String either "interceptor" or "target".
                - "init_orbit": Tuple of 6 Keplerian orbital elements as astropy Quantities.
                - "init_delta_v": Astropy quantity (in m/s) setting the initial orbital maneuver budget.
            epoch (Time): Start time of the simulation.
        """
        self.id = agent_id
        self.role = config["role"]
        self.initial_elements = config["init_orbit"]
        # Normalize init_delta_v to Quantity[m/s] even if provided as plain float.
        init_dv_val = config.get("init_delta_v", 0.0)
        if hasattr(init_dv_val, "to"):
            self.init_delta_v = init_dv_val.to(u.m / u.s)
        else:
            self.init_delta_v = float(init_dv_val) * u.m / u.s
        self.epoch = epoch

        self.orbit_state = OrbitState(self.initial_elements, epoch)
        self.used_delta_v = 0.0 * u.m / u.s

    def apply_action(self, dv_vector: np.ndarray, time: Time, maneuver_frame: str = "ECI"):
        """
        Applies an instantaneous delta-v maneuver at a given time.

        Args:
            dv_vector (np.ndarray or Quantity): Delta-v vector in the selected
                maneuver frame, shape (3,), values in m/s.
            time (Time): Time at which the maneuver is performed.
            maneuver_frame (str): Maneuver frame, either "ECI" or "TNW".
        """
        if hasattr(dv_vector, "to"):
            dv = dv_vector.to(u.m / u.s)
            dv_values = np.asarray(dv.to_value(u.m / u.s), dtype=float)
        else:
            dv_values = np.asarray(dv_vector, dtype=float)
            dv = dv_values * u.m / u.s

        if dv_values.shape != (3,):
            raise ValueError(f"Delta-v action must have shape (3,), got {dv_values.shape}.")

        if np.linalg.norm(dv_values) > 0:
            log.debug(f"[{self.id}] Applying Δv = {dv_values} m/s in {maneuver_frame} at t={time.iso}")

        self.orbit_state.apply_delta_v(dv, time, maneuver_frame=maneuver_frame)
        # Accumulate used Δv as a Quantity[m/s] to preserve unit consistency.
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
        Constructs the observation vector for the agent with normalization.

        The observation includes:
        - Own Keplerian elements (6D, normalized)
        - Own fuel remaining (1D, normalized)
        - For each other agent (regardless of role):
            - Their Keplerian elements (6D, normalized)
            - Their relative distance (1D, normalized)

        Total observation size: 7 + (N-1) × 7 = 7N

        Args:
            other_agents (dict): Mapping of agent_id → SatelliteAgent

        Returns:
            np.ndarray: Flat observation vector (float32)
        """

        obs = []

        # --- Own state ---
        own_kep_raw = keplerian_to_array(self.orbit_state.orbit)  # [a, e, i, raan, argp, M]
        
        # Normalize Keplerian elements
        a_norm = (own_kep_raw[0] - A_REF) / A_SCALE
        e_norm = own_kep_raw[1]  # Eccentricity is already 0-1
        i_norm = normalize_angle(own_kep_raw[2])
        raan_norm = normalize_angle(own_kep_raw[3])
        argp_norm = normalize_angle(own_kep_raw[4])
        m_norm = normalize_angle(own_kep_raw[5])
        
        obs.extend([a_norm, e_norm, i_norm, raan_norm, argp_norm, m_norm])
        
        # Remaining delta-v gauge
        remaining_dv_value = self.init_delta_v.to_value(u.m / u.s) - self.used_delta_v.to_value(u.m / u.s)
        remaining_dv_value = max(0.0, float(remaining_dv_value))
        # Normalize fuel (using its own init_delta_v if possible, else FUEL_SCALE)
        fuel_norm_scale = self.init_delta_v.to_value(u.m / u.s) if self.init_delta_v.to_value(u.m / u.s) > 0 else FUEL_SCALE
        obs.append(remaining_dv_value / fuel_norm_scale)

        # --- Other agents (all roles) ---
        for other_id, other in sorted(other_agents.items()):
            if other_id == self.id:
                continue
            
            other_kep_raw = keplerian_to_array(other.orbit_state.orbit)
            
            # Normalize other's Keplerian elements
            oa_norm = (other_kep_raw[0] - A_REF) / A_SCALE
            oe_norm = other_kep_raw[1]
            oi_norm = normalize_angle(other_kep_raw[2])
            oraan_norm = normalize_angle(other_kep_raw[3])
            oargp_norm = normalize_angle(other_kep_raw[4])
            om_norm = normalize_angle(other_kep_raw[5])
            
            obs.extend([oa_norm, oe_norm, oi_norm, oraan_norm, oargp_norm, om_norm])
            
            distance = compute_eci_distance(self.orbit_state, other.orbit_state)  # m
            obs.append(distance / DIST_SCALE)

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
            Quantity: Remaining delta-v (m/s).
        """
        return self.init_delta_v - self.get_used_delta_v()

    def get_used_delta_v(self) -> u.Quantity:
        """
        Returns the total delta-v applied by the agent so far.

        Returns:
            Quantity: Total delta-v (m/s).
        """
        return self.used_delta_v

    def summary(self):
        """
        Logs a summary of the current orbit in Keplerian form.
        """
        a, e, i, raan, argp, M = self.orbit_state.get_keplerian()
        log.info(
            f"[{self.id}] a={a.to(u.m):.1f}, e={e:.4f}, i={i.to(u.deg):.2f}, M={M.to(u.deg):.1f}"
        )
