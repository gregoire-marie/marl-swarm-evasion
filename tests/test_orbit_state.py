from astropy import units as u
from astropy.time import Time, TimeDelta
import numpy as np

from src.main.python.agents.orbit_state import OrbitState
from src.main.python.utils.constants import MU_EARTH

def get_sample_elements():
    """Approximate LEO orbit similar to the ISS"""
    return (
        (6378.0 + 423.0) * u.km,      # a (semi-major axis)
        0.00033 * u.one,    # eccentricity
        51.6 * u.deg,       # inclination
        0 * u.deg,          # RAAN
        0 * u.deg,          # argument of perigee
        0 * u.deg           # mean anomaly
    )


ess_epoch = Time("2025-01-01 00:00:00", scale="utc")


def test_initialization_and_rv():
    state = OrbitState(get_sample_elements(), ess_epoch)
    r, v = state.get_rv()

    # Structural assertions
    assert r.shape == (3,)
    assert v.shape == (3,)
    assert r.unit == u.km
    assert v.unit == u.km / u.s

    # Check norm against vis-viva equation
    a, e, *_ = get_sample_elements()
    r_mag = np.linalg.norm(r.to_value())
    expected_r = a.to_value(u.km) * (1 - e.value)  # Perigee for ν = 0
    assert np.isclose(r_mag, expected_r, rtol=1e-3)

    v_mag = np.linalg.norm(v.to_value())
    expected_v = np.sqrt(MU_EARTH.to_value(u.km**3 / u.s**2) * (2 / expected_r - 1 / a.to_value(u.km)))
    assert np.isclose(v_mag, expected_v, rtol=1e-3)


def test_propagation_forward():
    # Arrange
    elements = get_sample_elements()
    state = OrbitState(elements, ess_epoch)

    r0, v0 = state.get_rv()
    t0 = state.epoch

    dt = 60.0  # seconds
    new_epoch = t0 + TimeDelta(dt, format="sec")

    # Act
    state.propagate_to(new_epoch)
    r1, v1 = state.get_rv()

    # Assert: epoch updated
    assert abs((state.epoch - new_epoch).sec) < 1e-6, "Epoch mismatch after propagation."

    # Compute measured position & velocity changes
    dr = np.linalg.norm((r1 - r0).to_value(u.km))         # [km]
    dv = np.linalg.norm((v1 - v0).to_value(u.km / u.s))   # [km/s]

    # Compute expected Δr using linear motion: dr ≈ |v₀| * Δt
    v0_mag = np.linalg.norm(v0.to_value(u.km / u.s))      # [km/s]
    expected_dr = v0_mag * dt                             # [km]

    assert np.isclose(dr, expected_dr, rtol=0.0005), (
        f"Position change Δr = {dr:.3f} km, expected ≈ {expected_dr:.3f} km"
    )

    # Compute expected Δv using central acceleration: dv ≈ |a| * Δt
    r0_mag = np.linalg.norm(r0.to_value(u.km))            # [km]
    a_mag = MU_EARTH.to_value() / r0_mag**2               # [km/s²]
    expected_dv = a_mag * dt                              # [km/s]

    assert np.isclose(dv, expected_dv, rtol=0.0005), (
        f"Velocity change Δv = {dv:.5f} km/s, expected ≈ {expected_dv:.5f} km/s"
    )


def test_delta_v_application():
    # Propagate the orbit manually first to get v_before at same point
    state = OrbitState(get_sample_elements(), ess_epoch)

    maneuver_time = ess_epoch + TimeDelta(10.0, format="sec")

    state.propagate_to(maneuver_time)
    r_before, v_before = state.get_rv()

    # Apply delta-v at that exact point
    dv = np.array([0.01, -0.005, 0.002]) * u.km / u.s
    state.apply_delta_v(dv, maneuver_time)

    r_after, v_after = state.get_rv()

    delta_v_measured = np.linalg.norm((v_after - v_before).to_value())
    expected_dv = np.linalg.norm(dv.to_value())

    assert np.isclose(delta_v_measured, expected_dv, rtol=1e-6)


def test_keplerian_output():
    original_elements = get_sample_elements()
    state = OrbitState(original_elements, ess_epoch)

    a, e, inc, raan, argp, M = state.get_keplerian()

    # Compare to input values with tight tolerance
    atol_a = 1e-2 * u.km
    atol_ang = 1e-3 * u.deg
    atol_e = 1e-6

    assert np.isclose(a.to_value(u.km), original_elements[0].to_value(u.km), atol=atol_a.to_value())
    assert np.isclose(e.value, original_elements[1].value, atol=atol_e)
    assert np.isclose(inc.to_value(u.deg), original_elements[2].to_value(u.deg), atol=atol_ang.to_value())
    assert np.isclose(raan.to_value(u.deg), original_elements[3].to_value(u.deg), atol=atol_ang.to_value())
    assert np.isclose(argp.to_value(u.deg), original_elements[4].to_value(u.deg), atol=atol_ang.to_value())
    assert np.isclose(M.to_value(u.deg), original_elements[5].to_value(u.deg), atol=atol_ang.to_value())
