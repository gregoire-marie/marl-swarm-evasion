from astropy import units as u

# -- Simulation defaults --
DEFAULT_START_TIME = "2025-01-01 00:00:00"

# -- Agents commands --
DEFAULT_OBJECTIVES = {
    "collision_distance_m": 2000.0,     # Interceptors that bring the distance to a target under this threshold: win (m)
    "avoid_distance_m": 10000.0,        # Targets that keep the distance to all interceptors above this threshold: win (m)
    "same_role_spacing_m": 10000.0,     # All agents ought to keep the distance to same-role agents above this threshold (m)
    "minimal_delta_v_mps": 0.01,        # All agents ought to keep their remaining delta-v above this threshold (m/s)
    "reentry_altitude_m": 120000.0,     # All agents ought to keep their altitude above this threshold (m)
}

# -- Rewards --

## These rewards are shared regardless of the reward engine version
DEFAULT_REWARD_WEIGHTS = {
    ### All agents
    ## Reward functions parameters
    "fuel_penalty": -1.0e-3,        # Slope of the linear penalty proportional to used-Δv (in m/s)

    ## One time rewards
    "reentry_penalty": -10.0,       # One-time penalty when altitude falls below the reentry threshold
    "no_fuel_penalty": -10.0,       # One-time penalty when fuel falls below the minimal delta-v threshold
    "intercept_reward": 10.0,       # One-time reward when an interceptor intercepts a target
    "intercept_penalty": -10.0,     # One-time penalty when a target is intercepted
    "collision_penalty": -10.0,     # One time penalty when same-role agents collide
}

## These rewards function parameters depend on the reward engine version
DEFAULT_REWARD_WEIGHTS_V1 = {
    # Interceptor objectives
    "intercept_shaping": 10.0,          # Shapes the reward for distance to target minimization objective
    "interceptor_dispersion": 3.0,      # Shapes the reward for distance to other interceptors maximization objective

    # Target objectives
    "evasion_shaping": 3.0,             # Shapes the reward for distance to interceptors maximization objective
    "target_dispersion": 3.0,           # Shapes the reward for distance to other targets maximization objective
}

DEFAULT_REWARD_WEIGHTS_V2 = {
    # Soft shaping (separation / evasion / dispersion)
    "objective_d_safe": 10000.0,        # default safety distance (m)
    "objective_d_alpha": 1.0,           # default soft penalty scale

    # Hard shaping (interception)
    "intercept_shaping": 10.0,
    "interceptor_dispersion": 3.0,

    # Soft shaping (evasion)
    "evasion_shaping": 3.0,
    "target_dispersion": 3.0,

    "zero_d_alpha": 0.001,                 # Strength of reciprocal reward
    "zero_d_beta": 5.0,                 # Strength of reciprocal reward
    "zero_d_eps": 1e-3,                 # Numerical stability
    "zero_d_max": None,                 # if set, reward=0 beyond this distance (m)
}

# -- Physical constants --

MU_EARTH = 3.986004418e14 * u.m**3 / u.s**2   # Standard gravitational parameter of Earth
R_EARTH = 6378137.0 * u.m                     # Mean radius of Earth
