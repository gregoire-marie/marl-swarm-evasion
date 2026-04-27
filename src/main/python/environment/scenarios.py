import numpy as np
from astropy import units as u

from src.main.python.utils.constants import R_EARTH

def get_random_leo_elements(rng, alt_min=400_000.0, alt_max=600_000.0):
    """Generate random LEO Keplerian elements."""
    alt = rng.uniform(alt_min, alt_max)
    a = R_EARTH + alt * u.m
    e = rng.uniform(0.0, 0.001) * u.one
    inc = rng.uniform(0.0, 98.0) * u.deg
    raan = rng.uniform(0.0, 360.0) * u.deg
    argp = rng.uniform(0.0, 360.0) * u.deg
    m = rng.uniform(0.0, 360.0) * u.deg
    return (a, e, inc, raan, argp, m)

def pursuit_evasion_scenario(n_interceptors=1, n_targets=1, seed=None):
    """
    Generate per-agent configs for a pursuit-evasion scenario.
    """
    rng = np.random.default_rng(seed)
    
    agent_configs = {}
    
    # Base orbit for the group
    base_elements = get_random_leo_elements(rng)
    
    # Interceptors
    for i in range(n_interceptors):
        agent_id = f"interceptor_{i}"
        # Small perturbations from base orbit
        elements = list(base_elements)
        elements[3] = elements[3] + rng.uniform(-1.0, 1.0) * u.deg  # RAAN perturbation
        elements[5] = elements[5] + rng.uniform(-1.0, 1.0) * u.deg  # Mean anomaly perturbation
        
        agent_configs[agent_id] = {
            "role": "interceptor",
            "init_orbit": tuple(elements),
            "init_delta_v": 10000.0
        }
        
    # Targets
    for i in range(n_targets):
        agent_id = f"target_{i}"
        # Small perturbations from base orbit
        elements = list(base_elements)
        elements[3] = elements[3] + rng.uniform(-1.0, 1.0) * u.deg
        elements[5] = elements[5] + rng.uniform(-1.0, 1.0) * u.deg
        
        agent_configs[agent_id] = {
            "role": "target",
            "init_orbit": tuple(elements),
            "init_delta_v": 5000.0  # Targets usually have less fuel or are more constrained
        }
        
    return agent_configs

def constellation_scenario(n_agents=4, seed=None):
    """
    Generate per-agent configs for a mixed-role constellation scenario.
    """
    rng = np.random.default_rng(seed)
    agent_configs = {}
    
    base_elements = get_random_leo_elements(rng)
    
    for i in range(n_agents):
        agent_id = f"sat_{i}"
        elements = list(base_elements)
        elements[5] = (i * 360.0 / n_agents) * u.deg  # Evenly spaced in Mean Anomaly
        
        agent_configs[agent_id] = {
            "role": "interceptor" if i % 2 == 0 else "target",
            "init_orbit": tuple(elements),
            "init_delta_v": 10000.0
        }
        
    return agent_configs
