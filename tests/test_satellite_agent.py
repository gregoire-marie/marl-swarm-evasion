from astropy.time import Time, TimeDelta
from astropy import units as u
import numpy as np

from src.main.python.agents.satellite_agent import SatelliteAgent


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


def test_initialization():
    epoch = Time("2025-01-01 00:00:00", scale="utc")

    config = {
        "role": "interceptor",
        "init_orbit": get_sample_elements(),
        "delta_v": 10.0
    }

    agent = SatelliteAgent("agent_0", config, epoch)

    assert agent.id == "agent_0"
    assert agent.role == "interceptor"
    assert agent.orbit_state.epoch == epoch

    dv_total = agent.get_total_delta_v()
    assert isinstance(dv_total, u.Quantity)
    assert dv_total.unit == u.km / u.s
    assert dv_total.to_value() == 0.0  # Should be 0 at init


def test_propagation_and_action():
    epoch = Time("2025-01-01 00:00:00", scale="utc")

    config = {
        "role": "interceptor",
        "init_orbit": get_sample_elements(),
        "delta_v": 10.0
    }

    agent = SatelliteAgent("agent_1", config, epoch)

    r0, v0 = agent.orbit_state.get_rv()

    # Propagate forward
    dt = TimeDelta(60.0, format="sec")
    future_time = epoch + dt
    agent.propagate_to(future_time)

    assert abs((agent.orbit_state.epoch - future_time).sec) < 1e-3

    r1, v1 = agent.orbit_state.get_rv()
    delta_r = np.linalg.norm((r1 - r0).to_value(u.km))
    assert delta_r > 0.1, f"Expected non-zero position change, got {delta_r:.6f} km"

    # Apply Δv
    dv_vec = np.array([0.01, 0.0, 0.0], dtype=np.float32)
    agent.apply_action(dv_vec, future_time)

    # Check new velocity magnitude
    r2, v2 = agent.orbit_state.get_rv()
    dv_applied = np.linalg.norm((v2 - v1).to_value(u.km / u.s))
    expected_dv = np.linalg.norm(dv_vec)

    assert np.isclose(dv_applied, expected_dv, rtol=1e-6), (
        f"Δv mismatch: applied {dv_applied:.6f}, expected {expected_dv:.6f}"
    )

    # Check cumulative Δv updated
    dv_total = agent.get_total_delta_v().to_value(u.km / u.s)
    assert np.isclose(dv_total, expected_dv, rtol=1e-6)


def test_observation_vector():
    epoch = Time("2025-01-01 00:00:00", scale="utc")

    config_0 = {
        "role": "target",
        "init_orbit": get_sample_elements(),
        "delta_v": 10.0
    }
    config_1 = {
        "role": "interceptor",
        "init_orbit": get_sample_elements(),
        "delta_v": 10.0
    }

    agent_0 = SatelliteAgent("agent_0", config_0, epoch)
    agent_1 = SatelliteAgent("agent_1", config_1, epoch)

    agents_dict = {
        "agent_0": agent_0,
        "agent_1": agent_1,
    }

    obs = agent_0.get_observation(agents_dict)

    # Expected: 6 elements per agent (Keplerian), plus 1 delta-V gauge, and 1 distance to each other agent
    expected_len = 7 * len(agents_dict)
    assert obs.shape == (expected_len,), f"Expected obs shape {(expected_len,)}, got {obs.shape}"
    assert obs.dtype == np.float32

    # Verify that values aren't all zero
    assert not np.all(obs == 0.0), "Observation vector should not be all zeros"

    # Check that agent_0’s own elements match first 6 values
    own_elements = get_sample_elements()
    a, e, i, raan, argp, M = own_elements
    expected_first_elem = a.to_value(u.km)
    actual_first_elem = obs[0]

    assert np.isclose(actual_first_elem, expected_first_elem, rtol=1e-3), (
        f"First Keplerian element mismatch: {actual_first_elem} vs {expected_first_elem}"
    )

    # TODO: Add closest approaches and covariance
