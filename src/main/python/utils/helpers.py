import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from poliastro.twobody.angles import nu_to_E, E_to_M

# ========== Logger Setup ==========

def get_logger(name="orbital", level=logging.INFO):
    """
    Returns a configured logger instance.

    Parameters
    ----------
    name : str
        Name of the logger (e.g. module or agent id).
    level : logging level
        Default is logging.INFO.

    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
        logger.propagate = False
        logger.setLevel(level)
    return logger

# Optional: example global logger
log = get_logger("main_logger")

# ========== Other Helpers ==========

def unwrap_angle(angle_rad):
    """
    Unwrap angle to avoid discontinuities near 2π.

    Parameters
    ----------
    angle_rad : float or ndarray
        Angle in radians.

    Returns
    -------
    unwrapped : float or ndarray
        Angle in radians, unwrapped to continuous range.
    """
    return np.unwrap(np.atleast_1d(angle_rad)).squeeze()

def keplerian_to_array(orbit):
    """
    Converts Poliastro Orbit to unwrapped Keplerian elements array.

    Returns
    -------
    np.ndarray of [a, e, i, raan, argp, mean_anomaly] (floats)
    """
    a, e, inc, raan, argp, nu = orbit.classical()
    # Convert true anomaly back to mean anomaly (why: keep API consistent with constructor)
    e_q = (e if hasattr(e, "unit") else e * u.one)
    E = nu_to_E(nu, e_q)
    M = E_to_M(E, e_q).to(u.deg)
    return np.array([
        a.to_value(u.km),
        e.value,
        unwrap_angle(inc.to_value(u.rad)),
        unwrap_angle(raan.to_value(u.rad)),
        unwrap_angle(argp.to_value(u.rad)),
        unwrap_angle(M.to_value(u.rad)),
    ], dtype=np.float32)

def delta_v_norm(dv_vec):
    """
    Returns magnitude of delta-v vector.

    Parameters
    ----------
    dv_vec : astropy Quantity with shape (3,)

    Returns
    -------
    Quantity in km/s or m/s
    """
    return np.linalg.norm(dv_vec.to_value(u.km / u.s)) * u.km / u.s

def flatten_covariance(cov_matrix):
    """
    Flattens 6x6 covariance matrix to 36D vector (row-major).

    Parameters
    ----------
    cov_matrix : ndarray of shape (6, 6)

    Returns
    -------
    1D ndarray of length 36
    """
    return cov_matrix.flatten()

def vector_to_cartesian(vec):
    """
    Convert np.ndarray (3,) to astropy CartesianRepresentation.

    Parameters
    ----------
    vec : ndarray [x, y, z]

    Returns
    -------
    CartesianRepresentation with km units
    """
    return CartesianRepresentation(*vec) * u.km
