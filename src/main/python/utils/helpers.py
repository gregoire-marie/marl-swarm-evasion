import logging
import os
import re

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

# ========== File Helpers ==========

def resolve_checkpoint_path(path: str) -> str:
    """
    Resolves a user-supplied checkpoint reference into a concrete RLlib checkpoint directory.

    This helper accepts either:
    - a specific RLlib checkpoint directory (e.g., ``.../checkpoint_000123``), or
    - a trial/run directory that contains multiple ``checkpoint_*`` subdirectories.

    The input path is expanded (``~``), converted to an absolute path, and normalized. If the
    resolved directory already looks like an RLlib checkpoint (e.g., contains checkpoint
    metadata/state files), it is returned as-is. Otherwise, the directory is treated as a
    trial directory and the most recent checkpoint is selected by the numeric suffix of
    ``checkpoint_*`` (highest value wins).

    Args:
        path: Filesystem path to an RLlib checkpoint directory or to a trial directory
            containing ``checkpoint_*`` subdirectories.

    Returns:
        Absolute, normalized path to the resolved RLlib checkpoint directory.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ValueError: If the path is not a directory, or if no RLlib checkpoint can be
            identified under the provided directory.
    """
    CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint_(\d+)$")

    candidate_path = os.path.normpath(os.path.abspath(os.path.expanduser(path)))

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate_path}")
    if not os.path.isdir(candidate_path):
        raise ValueError(f"Checkpoint path must be a directory: {candidate_path}")

    if CHECKPOINT_DIR_PATTERN.fullmatch(os.path.basename(candidate_path)):
        return candidate_path

    entries = set(os.listdir(candidate_path))

    # If this dir itself already contains checkpoint state files, accept it.
    if (
        "rllib_checkpoint.json" in entries
        or any(
            name.startswith("algorithm_state.")
            and name.split(".")[-1] in {"pkl", "msgpack", "msgpck"}
            for name in entries
        )
        or any(re.fullmatch(r"checkpoint-\d+", name) for name in entries)
    ):
        return candidate_path

    # Otherwise, resolve a trial directory to its latest checkpoint_* subdirectory.
    checkpoint_candidates = []
    for name in entries:
        match = CHECKPOINT_DIR_PATTERN.fullmatch(name)
        if not match:
            continue
        checkpoint_dir = os.path.join(candidate_path, name)
        if os.path.isdir(checkpoint_dir):
            checkpoint_candidates.append((int(match.group(1)), checkpoint_dir))

    if not checkpoint_candidates:
        raise ValueError(
            "No RLlib checkpoint found. Provide either a checkpoint directory "
            f"(checkpoint_XXXXXX) or a trial directory containing checkpoint_* folders: {candidate_path}"
        )

    checkpoint_candidates.sort(key=lambda x: x[0], reverse=True)
    resolved_path = checkpoint_candidates[0][1]
    return resolved_path

# ========== RL Helpers ==========

def policy_mapping_fn(agent_id: str) -> str:
    """
    Maps agent IDs to policies.
    """
    if agent_id.startswith("interceptor"):
        return "interceptor_policy"
    elif agent_id.startswith("target"):
        return "target_policy"
    return "shared_policy"

# ========== Technical Helpers ==========

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
    Quantity in m/s
    """
    return np.linalg.norm(dv_vec.to_value(u.m / u.s)) * u.m / u.s

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
