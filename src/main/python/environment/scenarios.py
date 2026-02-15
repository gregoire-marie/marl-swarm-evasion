import numpy as np
from astropy import units as u
from astropy.time import Time

def get_random_leo_elements(rng, alt_min=400.0, alt_max=600.0):
    """Generate random LEO Keplerian elements."""
    alt = rng.uniform(alt_min, alt_max)
    a = (6378.137 + alt) * u.km
    e = rng.uniform(0.0, 0.001) * u.one
    inc = rng.uniform(0.0, 98.0) * u.deg
    raan = rng.uniform(0.0, 360.0) * u.deg
    argp = rng.uniform(0.0, 360.0) * u.deg
    m = rng.uniform(0.0, 360.0) * u.deg
    return (a, e, inc, raan, argp, m)

def pursuit_evasion_scenario(n_interceptors=1, n_targets=1, seed=None):
    """
    Generate a pursuit-evasion scenario with agents in similar orbits.
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
            "init_delta_v": 10.0
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
            "init_delta_v": 5.0  # Targets usually have less fuel or are more constrained
        }
        
    env_config = {
        "timestep_sec": 60.0,
        "episode_length": 100,
        "start_time": "2025-01-01 00:00:00",
        "max_delta_v_kms": 0.02,
    }
    
    return agent_configs, env_config

def constellation_scenario(n_agents=4, seed=None):
    """
    Generate a constellation of agents spread around the same orbital plane.
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
            "init_delta_v": 10.0
        }
        
    env_config = {
        "timestep_sec": 60.0,
        "episode_length": 100,
        "start_time": "2025-01-01 00:00:00",
        "max_delta_v_kms": 0.01,
    }
    
    return agent_configs, env_config
