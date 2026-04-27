import numpy as np

# Normalization constants for observations
# We aim to scale most values to roughly [-1, 1] or [0, 1]

# Semi-major axis: LEO is ~6.5e6 to ~8.5e6 m.
# Using a reference of 7.0e6 m and scaling by 1.0e6 m.
A_REF = 7_000_000.0
A_SCALE = 1_000_000.0

# Distances: can vary from 0 to millions of meters.
# Using 1.0e6 m as a reference scale.
DIST_SCALE = 1_000_000.0

# Fuel: remaining delta-v.
# Using 10,000 m/s as a typical max value if not provided.
FUEL_SCALE = 10000.0

def normalize_angle(angle_rad):
    """Wrap angle to [-pi, pi] and normalize to [-1, 1]."""
    # Wrap to [-pi, pi]
    wrapped = (angle_rad + np.pi) % (2 * np.pi) - np.pi
    return wrapped / np.pi

def denormalize_angle(norm_angle):
    """Denormalize from [-1, 1] to radians."""
    return norm_angle * np.pi
