from astropy import units as u

# -- Agents commands --
DEFAULT_OBJECTIVES = {
    "collision_distance_m": 2000.0,      # Threshold where a target is considered intercepted (m)
    "avoid_distance_m": 10000.0,         # Minimal spacing for targets to keep with interceptors (m)
    "same_role_spacing_m": 10000.0,      # Minimal spacing to keep between agents with the same role (m)
    "minimal_delta_v_mps": 0.01,         # Minimal delta-v to consider fuel depleted in an agent (m/s)
    "reentry_altitude_m": 120000.0,      # Altitude at which a satellite is considered reentered in the atmosphere (m)
}

# -- Rewards --
DEFAULT_REWARD_WEIGHTS = {
    # Interceptor objectives
    "intercept_shaping": 10.0,
    "interceptor_dispersion": 3.0,

    # Target objectives
    "evasion_shaping": 3.0,
    "target_dispersion": 3.0,

    # All agents
    "fuel_penalty": -1.0e-3,                # Linear penalty for Δv used (m/s input)
    "reentry_penalty": -100.0,              # Applied when altitude falls below the reentry threshold
}

DEFAULT_REWARD_WEIGHTS_V2 = {
    # Soft shaping (separation / evasion / dispersion)
    "objective_d_safe": 10000.0,        # default safety distance (m)
    "objective_d_alpha": 1.0,          # default soft penalty scale

    # Hard shaping (interception)
    "zero_d_beta": 1.0,           # strength of reciprocal reward
    "zero_d_eps": 1e-3,           # numerical stability
    "zero_d_max": None,         # if set, reward=0 beyond this distance (m)
}

# -- Physical constants --

MU_EARTH = 3.986004418e14 * u.m**3 / u.s**2   # Standard gravitational parameter of Earth
R_EARTH = 6378137.0 * u.m                     # Mean radius of Earth
