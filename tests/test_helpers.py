import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from src.main.python.utils.helpers import (
    unwrap_angle,
    keplerian_to_array,
    delta_v_norm,
    flatten_covariance,
    vector_to_cartesian,
)
from src.main.python.agents.orbit_state import OrbitState


def sample_elements():
    return (
        (6378.0 + 500.0) * u.km,
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

    # Semi-major axis in km should match input within tolerance
    a_km = sample_elements()[0].to_value(u.km)
    assert np.isclose(arr[0], a_km, rtol=1e-3)
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
    assert rep.x.unit == u.km
    assert rep.y.unit == u.km
    assert rep.z.unit == u.km
    assert np.isclose(rep.x.to_value(u.km), 1.0)
    assert np.isclose(rep.y.to_value(u.km), 2.0)
    assert np.isclose(rep.z.to_value(u.km), 3.0)
