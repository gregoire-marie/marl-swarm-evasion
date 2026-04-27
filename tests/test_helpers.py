import logging
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from uuid import uuid4

from src.main.python.utils.helpers import (
    get_logger,
    policy_mapping_fn,
    resolve_checkpoint_path,
    unwrap_angle,
    keplerian_to_array,
    delta_v_norm,
    flatten_covariance,
    vector_to_cartesian,
)
from src.main.python.agents.orbit_state import OrbitState


def sample_elements():
    return (
        (6_378_000.0 + 500_000.0) * u.m,
        0.00033 * u.one,
        51.6 * u.deg,
        0 * u.deg,
        0 * u.deg,
        0 * u.deg,
    )


def test_unwrap_angle_continuity():
    # Construct a sequence crossing 2π boundary
    two_pi = 2 * np.pi
    angles = np.array([two_pi - 0.1, two_pi + 0.1, 2 * two_pi + 0.05])
    unwrapped = unwrap_angle(angles)

    # The unwrap should remove 2π jumps, making transitions small
    diffs = np.diff(unwrapped)
    assert np.all(np.isfinite(unwrapped))
    # Expected local differences: ~+0.2 then ~-0.05 (after unwrapping last sample)
    assert np.isclose(diffs[0], 0.2, atol=1e-6)
    assert np.isclose(diffs[1], -0.05, atol=1e-6)


def test_keplerian_to_array_properties():
    epoch = Time("2025-01-01 00:00:00", scale="utc")
    state = OrbitState(sample_elements(), epoch)
    arr = keplerian_to_array(state.orbit)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6,)
    assert arr.dtype == np.float32

    # Semi-major axis in m should match input within tolerance
    a_m = sample_elements()[0].to_value(u.m)
    assert np.isclose(arr[0], a_m, rtol=1e-3)
    # All finite
    assert np.all(np.isfinite(arr))


def test_delta_v_norm_units_and_value():
    dv_vec = np.array([10.0, -20.0, 0.0]) * u.m / u.s
    mag = delta_v_norm(dv_vec)
    assert hasattr(mag, "unit")
    assert mag.unit == (u.m / u.s)
    expected = np.linalg.norm(dv_vec.to_value(u.m / u.s))
    assert np.isclose(mag.to_value(u.m / u.s), expected, rtol=1e-12)


def test_flatten_covariance():
    cov = np.arange(36, dtype=float).reshape(6, 6)
    flat = flatten_covariance(cov)
    assert isinstance(flat, np.ndarray)
    assert flat.shape == (36,)
    assert flat[0] == cov[0, 0]
    assert flat[-1] == cov[5, 5]


def test_vector_to_cartesian_units():
    vec = np.array([1.0, 2.0, 3.0])
    rep = vector_to_cartesian(vec)
    # CartesianRepresentation stores components as Quantity
    assert rep.x.unit == u.m
    assert rep.y.unit == u.m
    assert rep.z.unit == u.m
    assert np.isclose(rep.x.to_value(u.m), 1.0)
    assert np.isclose(rep.y.to_value(u.m), 2.0)
    assert np.isclose(rep.z.to_value(u.m), 3.0)


def test_get_logger_reuses_existing_handler():
    logger_name = f"test_logger_{uuid4().hex}"

    logger = get_logger(logger_name, level=logging.DEBUG)
    same_logger = get_logger(logger_name, level=logging.ERROR)

    assert same_logger is logger
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG
    assert logger.handlers[0].level == logging.DEBUG


def test_resolve_checkpoint_path_rejects_missing_and_file_paths(tmp_path):
    missing_path = tmp_path / "missing"
    file_path = tmp_path / "checkpoint.txt"
    file_path.write_text("not a directory")

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(str(missing_path))

    with pytest.raises(ValueError, match="must be a directory"):
        resolve_checkpoint_path(str(file_path))


def test_resolve_checkpoint_path_accepts_concrete_checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint_000123"
    checkpoint_dir.mkdir()

    resolved = resolve_checkpoint_path(str(checkpoint_dir))

    assert resolved == str(checkpoint_dir.resolve())


def test_resolve_checkpoint_path_accepts_checkpoint_state_directory(tmp_path):
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "rllib_checkpoint.json").write_text("{}")

    resolved = resolve_checkpoint_path(str(checkpoint_dir))

    assert resolved == str(checkpoint_dir.resolve())


def test_resolve_checkpoint_path_picks_latest_trial_checkpoint(tmp_path):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "checkpoint_000001").mkdir()
    (trial_dir / "checkpoint_000010").mkdir()
    (trial_dir / "checkpoint_000003").mkdir()

    resolved = resolve_checkpoint_path(str(trial_dir))

    assert resolved == str((trial_dir / "checkpoint_000010").resolve())


def test_resolve_checkpoint_path_requires_checkpoint_content(tmp_path):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "notes").mkdir()

    with pytest.raises(ValueError, match="No RLlib checkpoint found"):
        resolve_checkpoint_path(str(trial_dir))


@pytest.mark.parametrize(
    ("agent_id", "expected_policy"),
    [
        ("interceptor_0", "interceptor_policy"),
        ("target_0", "target_policy"),
        ("observer_0", "shared_policy"),
    ],
)
def test_policy_mapping_fn(agent_id, expected_policy):
    assert policy_mapping_fn(agent_id) == expected_policy
