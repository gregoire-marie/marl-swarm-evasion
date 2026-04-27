import numpy as np
from itertools import combinations
from astropy import units as u
from astropy.time import Time
from typing import Dict, Optional, Tuple

from src.main.python.agents.satellite_agent import SatelliteAgent
from src.main.python.orbital_meca.orbits import compute_eci_distance, compute_altitude_m
from src.main.python.utils.constants import DEFAULT_OBJECTIVES, DEFAULT_REWARD_WEIGHTS
from typing import Callable


def objective_d_shaping_generator(objective: float, w: float):
    """Smoothly increase reward as value exceeds obj, saturates at large values."""
    def fn(x):
        r = (objective / w) * (np.log(x + 1e-6) / np.log(objective + 1e-6) - 1)
        return float(np.tanh(r))  # Tanh prevents exploding gradient
    return fn

def zero_d_shaping_generator(w: float):
    """Aggressively reward smaller values (inverse-square), capped to prevent gradient explosion."""
    def fn(x):
        return min(float(w / (x + 1e-6)), 20.0)
    return fn


def objective_d_shaping_generator_v2(d_safe: float, alpha: float) -> Callable[[float], float]:
    """Quadratic soft penalty for proximity.

    Returns 0 when distance >= d_safe.
    Returns negative smooth penalty when distance < d_safe.

    Args:
        d_safe (float): Safety distance threshold.
        alpha (float): Penalty scale.

    Returns:
        Callable[[float], float]: Distance → reward.
    """
    def fn(distance: float) -> float:
        if distance >= d_safe:
            return 0.0
        x = 1.0 - (distance / d_safe)
        return -alpha * (x * x)
    return fn


def zero_d_shaping_generator_v2(beta: float, eps: float = 1e-3, d_max: float = None) -> Callable[[float], float]:
    """Aggressive reciprocal interception reward.

    Args:
        beta (float): Reward scale.
        eps (float): Numerical stability.
        d_max (float, optional): Zero reward beyond this distance.

    Returns:
        Callable[[float], float]: Distance → reward.
    """
    def fn(distance: float) -> float:
        if d_max is not None and distance >= d_max:
            return 0.0
        return float(beta / (distance + eps))
    return fn


def linear_reward_generator(w: float):
    """Linearly penalize or reward based on magnitude."""
    def fn(x):
        return float(w * x)
    return fn

def compute_rewards(
    agent_states: Dict[str, SatelliteAgent],
    current_time: Time,
    objectives: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Compute per-agent rewards and simulation status flags based on constellation state.

    Returns a dict of rewards, and a dict of termination conditions.

    Args:
        agent_states (Dict[str, SatelliteAgent]): Map from agent_id to SatelliteAgent.
        current_time (Time): Current time of the simulation.
        objectives (Optional[Dict[str, float]]): Thresholds expressed as plain floats with explicit units:
            - collision_distance_m: meters
            - avoid_distance_m: meters
            - same_role_spacing_m: meters
            - minimal_delta_v_mps: m/s
        weights (Optional[Dict[str, float]]): Weights for each reward component (dimensionless floats).

    Returns:
        Tuple[Dict[str, float], Dict[str, bool]]:
            - rewards[agent_id] = float
            - flags = {
                "intercept_success": bool,
                "interceptors_coll": bool,
                "targets_coll": bool,
                "no_fuel": bool,
            }

    Unit conventions
    -----------------
    - Distances are computed via compute_eci_distance(...) and are plain floats in meters.
    - Delta-v usage is obtained from SatelliteAgent as an astropy Quantity and converted to floats in m/s
      for shaping functions.
    - Shaping functions accept and return plain floats; no astropy Quantities should be passed into them.
    """
    if objectives is None:
        objectives = DEFAULT_OBJECTIVES
    if weights is None:
        weights = DEFAULT_REWARD_WEIGHTS

    rewards: Dict[str, float] = {agent_id: 0.0 for agent_id in agent_states}
    flags = {
        "intercept_success": False,
        "interceptors_coll": False,
        "targets_coll": False,
        "no_fuel": False,
        "reentry": False,
    }

    interceptors = {k: a for k, a in agent_states.items() if a.role == "interceptor"}
    targets = {k: a for k, a in agent_states.items() if a.role == "target"}

    # === Define reward shaping functions ===
    intercept_reward_fn = zero_d_shaping_generator(
        w=weights["intercept_shaping"]
    )  # Interceptors distance with targets: hard minimization

    target_evasion_reward_fn = objective_d_shaping_generator(objective=objectives["avoid_distance_m"], w=weights[
        "evasion_shaping"])  # Targets distance with interceptors: soft maximization

    interceptor_spacing_reward_fn = objective_d_shaping_generator(objective=objectives["same_role_spacing_m"], w=weights[
        "interceptor_dispersion"])  # Interceptor distance with interceptors: soft maximization

    target_spacing_reward_fn = objective_d_shaping_generator(objective=objectives["same_role_spacing_m"], w=weights[
        "target_dispersion"])  # Targets distance with targets : soft maximization

    fuel_penalty_fn = linear_reward_generator(
        w=weights["fuel_penalty"]
    )  # Fuel usage minimization: linear minimization

    # === Interceptor ↔ Target (evasion & interception) ===
    # Aggressively minimize distance (interceptor), softly maximize distance (target)
    for int_id, interceptor in interceptors.items():
        for tgt_id, target in targets.items():
            dist_m = compute_eci_distance(interceptor.orbit_state, target.orbit_state)

            if dist_m < objectives["collision_distance_m"]:
                flags["intercept_success"] = True

            rewards[int_id] += intercept_reward_fn(dist_m)
            rewards[tgt_id] += target_evasion_reward_fn(dist_m)

    # === Interceptor ↔ Interceptor (dispersion) ===
    # Softly maximize distance
    for id1, id2 in combinations(interceptors.keys(), 2):
        a1, a2 = interceptors[id1], interceptors[id2]
        dist_m = compute_eci_distance(a1.orbit_state, a2.orbit_state)

        if dist_m < objectives["collision_distance_m"]:
            flags["interceptors_coll"] = True

        reward = interceptor_spacing_reward_fn(dist_m)
        rewards[id1] += reward
        rewards[id2] += reward

    # === Target ↔ Target (dispersion) ===
    # Softly maximize distance
    for id1, id2 in combinations(targets.keys(), 2):
        a1, a2 = targets[id1], targets[id2]
        dist_m = compute_eci_distance(a1.orbit_state, a2.orbit_state)

        if dist_m < objectives["collision_distance_m"]:
            flags["targets_coll"] = True

        reward = target_spacing_reward_fn(dist_m)
        rewards[id1] += reward
        rewards[id2] += reward

    # === Fuel usage penalty (all agents) ===
    for agent_id, agent in agent_states.items():
        dv_used = agent.get_used_delta_v().to_value(u.m / u.s)
        remaining_dv = agent.get_remaining_delta_v().to_value(u.m / u.s)

        if remaining_dv < objectives["minimal_delta_v_mps"]:
            flags["no_fuel"] = True

        rewards[agent_id] += fuel_penalty_fn(dv_used)

    # === Reentry termination criterium (all agents) ===
    for agent_id, agent in agent_states.items():
        curr_alt = compute_altitude_m(agent.orbit_state)

        if curr_alt < objectives["reentry_altitude_m"]:
            flags["reentry"] = True
    return rewards, flags
