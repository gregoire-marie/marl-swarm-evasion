from astropy import units as u

# -- Agents commands --
DEFAULT_OBJECTIVES = {
    "collision_distance_km": 2.0,        # Threshold where a target is considered intercepted (km)
    "avoid_distance_km": 10.0,           # Minimal spacing for targets to keep with interceptors (km)
    "same_role_spacing_km": 10.0,        # Minimal spacing to keep between agents with the same role (km)
    "minimal_delta_v_kms": 0.01e-3,      # Minimal delta-v to consider fuel depleted in an agent (km/s)
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
    "fuel_penalty": -1.0,                   # Linear penalty for Δv used
}

DEFAULT_REWARD_WEIGHTS_V2 = {
    # Soft shaping (separation / evasion / dispersion)
    "objective_d_safe": 10.0,        # default safety distance (km)
    "objective_d_alpha": 1.0,          # default soft penalty scale

    # Hard shaping (interception)
    "zero_d_beta": 1.0,           # strength of reciprocal reward
    "zero_d_eps": 1e-3,           # numerical stability
    "zero_d_max": None,         # if set, reward=0 beyond this
}

# -- Physical constants --

MU_EARTH = 398600.4418 * u.km**3 / u.s**2   # Standard gravitational parameter of Earth
R_EARTH = 6378.137 * u.km                   # Mean radius of Earth

# -- Default satellite config --

DEFAULT_MASS = 50 * u.kg                # Typical microsatellite mass
MAX_DELTA_V = 0.1 * u.km / u.s          # Max dv per timestep (in ECI)
MAX_TOTAL_DELTA_V = 10.0 * u.km / u.s   # Upper limit for mission delta-v

# -- Simulation settings --

DEFAULT_TIMESTEP = 10 * u.s
DEFAULT_EPISODE_LENGTH = 1000           # Max steps per episode
NUM_CLOSE_ENCOUNTERS = 3

# -- Encounter thresholds --

CA_MIN_DIST_INTERCEPTOR = 5 * u.km      # Min distance between interceptors
CA_MIN_DIST_TARGET = 10 * u.km          # Min distance between target & interceptor
CA_UNCERTAINTY_THRESHOLD = 1.0          # Unitless or TBD definition (covariance trace, etc.)

# -- Observation config --

NUM_KPL_ELEMENTS = 6                    # a, e, i, RAAN, argp, mean anomaly
NUM_ENCOUNTERS_PER_AGENT = 3            # Must match reward and obs design
COV_MATRIX_SIZE = 6                     # 6x6 covariance (flattened = 36)
