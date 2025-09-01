from astropy import units as u

# -- Physical constants --

MU_EARTH = 398600.4418 * u.km**3 / u.s**2  # Standard gravitational parameter of Earth
R_EARTH = 6378.137 * u.km                 # Mean radius of Earth

# -- Default satellite config --

DEFAULT_MASS = 50 * u.kg                  # Typical microsatellite mass
MAX_DELTA_V = 0.1 * u.km / u.s            # Max dv per timestep (in ECI)
MAX_TOTAL_DELTA_V = 10.0 * u.km / u.s     # Upper limit for mission delta-v

# -- Simulation settings --

DEFAULT_TIMESTEP = 10 * u.s
DEFAULT_EPISODE_LENGTH = 1000             # Max steps per episode
NUM_CLOSE_ENCOUNTERS = 3

# -- Encounter thresholds --

CA_MIN_DIST_INTERCEPTOR = 5 * u.km        # Min distance between interceptors
CA_MIN_DIST_TARGET = 10 * u.km            # Min distance between target & interceptor
CA_UNCERTAINTY_THRESHOLD = 1.0            # Unitless or TBD definition (covariance trace, etc.)

# -- Observation config --

NUM_KPL_ELEMENTS = 6                      # a, e, i, RAAN, argp, mean anomaly
NUM_ENCOUNTERS_PER_AGENT = 3              # Must match reward and obs design
COV_MATRIX_SIZE = 6                       # 6x6 covariance (flattened = 36)
