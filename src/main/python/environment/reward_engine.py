import numpy as np
from itertools import combinations
from astropy import units as u
from typing import Callable, Dict, Optional, Tuple

from src.main.python.agents.satellite_agent import SatelliteAgent
from src.main.python.orbital_meca.orbits import compute_eci_distance, compute_altitude_m
from src.main.python.utils.constants import (
    DEFAULT_OBJECTIVES,
    DEFAULT_REWARD_WEIGHTS_V1,
    DEFAULT_REWARD_WEIGHTS_V2,
    DEFAULT_REWARD_WEIGHTS
)

# Set to "v1" for the legacy shaping functions, or "v2" for the current engine.
ENGINE_VERSION = "v1"
SUPPORTED_ENGINE_VERSIONS = ("v1", "v2")


def objective_d_shaping_generator_v1(objective: float, w: float):
    """Smoothly increase reward as value exceeds obj, saturates at large values."""
    def fn(x):
        r = (objective / w) * (np.log(x + 1e-6) / np.log(objective + 1e-6) - 1)
        return float(np.tanh(r))  # Tanh prevents exploding gradient
    return fn

def zero_d_shaping_generator_v1(w: float):
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


def zero_d_shaping_generator_v2(alpha: float, beta: float, r_max: float, eps: float = 1e-3, d_max: float = None) -> Callable[[float], float]:
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
        return min(r_max, float(beta / (alpha * (distance + eps))))
    return fn


def linear_reward_generator(w: float):
    """Linearly penalize or reward based on magnitude."""
    def fn(x):
        return float(w * x)
    return fn


def _build_distance_reward_functions(
    engine_version: str
) -> Tuple[
    Callable[[float], float],
    Callable[[float], float],
    Callable[[float], float],
    Callable[[float], float],
    Callable[[float], float],
]:
    """Returns the reward functions as a tuple.

    Returns None if engine version is not supported.

    Returns:
        (intercept_reward_fn,
        target_evasion_reward_fn,
        interceptor_spacing_reward_fn,
        target_spacing_reward_fn,
        fuel_penalty_fn)

    """
    if engine_version == "v1":
        fn_tuple = (
            zero_d_shaping_generator_v1(w=DEFAULT_REWARD_WEIGHTS_V1["intercept_shaping"]),
            objective_d_shaping_generator_v1(
                objective=DEFAULT_OBJECTIVES["avoid_distance_m"],
                w=DEFAULT_REWARD_WEIGHTS_V1["evasion_shaping"],
            ),
            objective_d_shaping_generator_v1(
                objective=DEFAULT_OBJECTIVES["same_role_spacing_m"],
                w=DEFAULT_REWARD_WEIGHTS_V1["interceptor_dispersion"],
            ),
            objective_d_shaping_generator_v1(
                objective=DEFAULT_OBJECTIVES["same_role_spacing_m"],
                w=DEFAULT_REWARD_WEIGHTS_V1["target_dispersion"],
            ),
        )

    elif engine_version == "v2":
        fn_tuple = (
            zero_d_shaping_generator_v2(
                alpha=DEFAULT_REWARD_WEIGHTS_V2["zero_d_alpha"],
                beta=DEFAULT_REWARD_WEIGHTS_V2["zero_d_beta"],
                eps=DEFAULT_REWARD_WEIGHTS_V2["zero_d_eps"],
                r_max=DEFAULT_REWARD_WEIGHTS["intercept_reward"],
                d_max=DEFAULT_REWARD_WEIGHTS_V2["zero_d_max"],
            ),
            objective_d_shaping_generator_v2(
                d_safe=DEFAULT_OBJECTIVES["avoid_distance_m"],
                alpha=DEFAULT_REWARD_WEIGHTS_V2["evasion_shaping"] * DEFAULT_REWARD_WEIGHTS_V2["objective_d_alpha"],
            ),
            objective_d_shaping_generator_v2(
                d_safe=DEFAULT_OBJECTIVES["same_role_spacing_m"],
                alpha=DEFAULT_REWARD_WEIGHTS_V2["interceptor_dispersion"] * DEFAULT_REWARD_WEIGHTS_V2["objective_d_alpha"],
            ),
            objective_d_shaping_generator_v2(
                d_safe=DEFAULT_OBJECTIVES["same_role_spacing_m"],
                alpha=DEFAULT_REWARD_WEIGHTS_V2["target_dispersion"] * DEFAULT_REWARD_WEIGHTS_V2["objective_d_alpha"],
            ),
        )
    else:
        raise ValueError(
            f"Unsupported reward engine version '{engine_version}'. "
            f"Supported versions: {SUPPORTED_ENGINE_VERSIONS}."
        )

    fuel_fn = linear_reward_generator(w=DEFAULT_REWARD_WEIGHTS["fuel_penalty"])  # Fuel usage minimization: linear minimization)
    return fn_tuple + (fuel_fn,)


def compute_rewards(
    agent_states: Dict[str, SatelliteAgent],
    engine_version: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Compute per-agent rewards and simulation status flags based on constellation state.

    Returns a dict of rewards, and a dict of termination conditions.

    Args:
        agent_states (Dict[str, SatelliteAgent]): Map from agent_id to SatelliteAgent.
        engine_version (Optional[str]): Reward engine version to use. Defaults to ENGINE_VERSION.

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
    if engine_version is None:
        engine_version = ENGINE_VERSION

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
    (
        intercept_reward_fn,
        target_evasion_reward_fn,
        interceptor_spacing_reward_fn,
        target_spacing_reward_fn,
        fuel_penalty_fn
    ) = _build_distance_reward_functions(engine_version)

    # === Interceptor ↔ Target (evasion & interception) ===
    # Aggressively minimize distance (interceptor), softly maximize distance (target)
    for int_id, interceptor in interceptors.items():
        for tgt_id, target in targets.items():
            dist_m = compute_eci_distance(interceptor.orbit_state, target.orbit_state)

            if dist_m < DEFAULT_OBJECTIVES["collision_distance_m"]:
                flags["intercept_success"] = True
                rewards[int_id] += DEFAULT_REWARD_WEIGHTS["intercept_reward"]
                rewards[tgt_id] += DEFAULT_REWARD_WEIGHTS["intercept_penalty"]

            rewards[int_id] += intercept_reward_fn(dist_m)
            rewards[tgt_id] += target_evasion_reward_fn(dist_m)

    # === Interceptor ↔ Interceptor (dispersion) ===
    # Softly maximize distance
    for id1, id2 in combinations(interceptors.keys(), 2):
        a1, a2 = interceptors[id1], interceptors[id2]
        dist_m = compute_eci_distance(a1.orbit_state, a2.orbit_state)

        if dist_m < DEFAULT_OBJECTIVES["collision_distance_m"]:
            flags["interceptors_coll"] = True
            rewards[id1] += DEFAULT_REWARD_WEIGHTS["collision_penalty"]
            rewards[id2] += DEFAULT_REWARD_WEIGHTS["collision_penalty"]

        reward = interceptor_spacing_reward_fn(dist_m)
        rewards[id1] += reward
        rewards[id2] += reward

    # === Target ↔ Target (dispersion) ===
    # Softly maximize distance
    for id1, id2 in combinations(targets.keys(), 2):
        a1, a2 = targets[id1], targets[id2]
        dist_m = compute_eci_distance(a1.orbit_state, a2.orbit_state)

        if dist_m < DEFAULT_OBJECTIVES["collision_distance_m"]:
            flags["targets_coll"] = True
            rewards[id1] += DEFAULT_REWARD_WEIGHTS["collision_penalty"]
            rewards[id2] += DEFAULT_REWARD_WEIGHTS["collision_penalty"]

        reward = target_spacing_reward_fn(dist_m)
        rewards[id1] += reward
        rewards[id2] += reward

    # === Fuel usage penalty (all agents) ===
    for agent_id, agent in agent_states.items():
        last_action_dv = agent.get_last_action_delta_v().to_value(u.m / u.s)
        remaining_dv = agent.get_remaining_delta_v().to_value(u.m / u.s)

        if remaining_dv < DEFAULT_OBJECTIVES["minimal_delta_v_mps"]:
            flags["no_fuel"] = True
            rewards[agent_id] += DEFAULT_REWARD_WEIGHTS["no_fuel_penalty"]

        rewards[agent_id] += fuel_penalty_fn(last_action_dv)

    # === Reentry termination criterium (all agents) ===
    for agent_id, agent in agent_states.items():
        curr_alt = compute_altitude_m(agent.orbit_state)

        if curr_alt < DEFAULT_OBJECTIVES["reentry_altitude_m"]:
            flags["reentry"] = True
            rewards[agent_id] += DEFAULT_REWARD_WEIGHTS["reentry_penalty"]

    return rewards, flags
