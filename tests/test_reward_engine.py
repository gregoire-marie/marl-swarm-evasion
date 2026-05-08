import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from src.main.python.agents.satellite_agent import SatelliteAgent
from src.main.python.environment.reward_engine import compute_rewards
from src.main.python.utils.constants import DEFAULT_OBJECTIVES, DEFAULT_REWARD_WEIGHTS

# === Fixtures ===

@pytest.fixture
def default_epoch():
    return Time("2025-01-01 00:00:00", scale="utc")

@pytest.fixture
def default_orbit_near():
    return (
        6_771_000.0 * u.m, 0.0001 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 0 * u.deg
    )

@pytest.fixture
def default_orbit_far():
    return (
        6_771_000.0 * u.m, 0.0001 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 210 * u.deg  # Further apart to avoid false positives
    )

def default_agent(agent_id, role, orbit, epoch, delta_v=1000.0 * u.m / u.s):
    config = {
        "role": role,
        "init_orbit": orbit,
        "init_delta_v": delta_v,
    }
    return SatelliteAgent(agent_id, config, epoch)


# === Tests ===

def test_interception_success(default_epoch, default_orbit_near):
    i1 = default_agent("i1", "interceptor", default_orbit_near, default_epoch)
    t1 = default_agent("t1", "target", default_orbit_near, default_epoch)

    agents = {"i1": i1, "t1": t1}
    rewards, flags = compute_rewards(agents)

    assert flags["intercept_success"]
    assert not flags["interceptors_coll"]
    assert not flags["targets_coll"]
    assert not flags["no_fuel"]

    assert isinstance(rewards["i1"], float)
    assert isinstance(rewards["t1"], float)
    assert rewards["i1"] > 0.0  # interception success = high reward
    assert rewards["t1"] < 0.0  # evasion failure = penalized


def test_interception_shaping(default_epoch, default_orbit_near, default_orbit_far):
    i1 = default_agent("i1", "interceptor", default_orbit_near, default_epoch)
    t1 = default_agent("t1", "target", default_orbit_far, default_epoch)

    agents = {"i1": i1, "t1": t1}
    rewards, flags = compute_rewards(agents)

    assert not flags["intercept_success"]
    assert rewards["i1"] > 0.0
    assert rewards["t1"] > 0.0


def test_same_role_dispersion_and_collision(default_epoch, default_orbit_near, default_orbit_far):
    i1_orbit = (
        6_771_000.0 * u.m, 0.0001 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 0 * u.deg
    )
    i2_orbit = (
        6_771_000.0 * u.m, 0.0001 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 210 * u.deg
    )
    t1_orbit = (
        (6_771_000.0 + 200_000.0) * u.m, 0.0001 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 0 * u.deg
    )
    t2_orbit = t1_orbit  # Same orbit to provoke collision

    i1 = default_agent("i1", "interceptor", i1_orbit, default_epoch)
    i2 = default_agent("i2", "interceptor", i2_orbit, default_epoch)

    t1 = default_agent("t1", "target", t1_orbit, default_epoch)
    t2 = default_agent("t2", "target", t2_orbit, default_epoch)

    agents = {"i1": i1, "i2": i2, "t1": t1, "t2": t2}
    rewards, flags = compute_rewards(agents)

    assert not flags["intercept_success"]
    assert not flags["interceptors_coll"]
    assert flags["targets_coll"]

def test_interceptor_collision(default_epoch, default_orbit_near):
    i1_orbit = default_orbit_near
    i2_orbit = default_orbit_near  # Same orbit to provoke collision

    i1 = default_agent("i1", "interceptor", i1_orbit, default_epoch)
    i2 = default_agent("i2", "interceptor", i2_orbit, default_epoch)

    agents = {"i1": i1, "i2": i2}
    rewards, flags = compute_rewards(agents)

    assert flags["interceptors_coll"]


def test_fuel_penalty_and_no_fuel(default_epoch, default_orbit_near):
    agent = default_agent("s1", "target", default_orbit_near, default_epoch)
    agent.used_delta_v = 1500.0 * u.m / u.s

    rewards, flags = compute_rewards({"s1": agent})

    assert flags["no_fuel"]
    expected = DEFAULT_REWARD_WEIGHTS["fuel_penalty"] * 1500.0
    assert np.isclose(rewards["s1"], expected, rtol=1e-2)


def test_reentry_penalty(default_epoch):
    reentry_orbit = (
        (6_378_137.0 + DEFAULT_OBJECTIVES["reentry_altitude_m"] - 1_000.0) * u.m,
        0.0 * u.one,
        51.6 * u.deg,
        0 * u.deg,
        0 * u.deg,
        0 * u.deg,
    )
    agent = default_agent("s1", "target", reentry_orbit, default_epoch)

    rewards, flags = compute_rewards({"s1": agent})

    assert flags["reentry"]
    assert np.isclose(rewards["s1"], DEFAULT_REWARD_WEIGHTS["reentry_penalty"])


def test_mixed_constellation_flags_and_rewards(default_epoch, default_orbit_near, default_orbit_far):
    i1 = default_agent("i1", "interceptor", default_orbit_near, default_epoch)
    i2 = default_agent("i2", "interceptor", default_orbit_far, default_epoch)
    t1 = default_agent("t1", "target", default_orbit_far, default_epoch)

    i1.used_delta_v = 200.0 * u.m / u.s
    i2.used_delta_v = 100.0 * u.m / u.s
    t1.used_delta_v = 500.0 * u.m / u.s

    agents = {"i1": i1, "i2": i2, "t1": t1}
    rewards, flags = compute_rewards(agents)

    assert flags["intercept_success"]
    assert not flags["interceptors_coll"]
    assert not flags["targets_coll"]
    assert not flags["no_fuel"]

    for k in rewards:
        assert not np.isclose(rewards[k], 0.0, atol=1e-3)


def test_reward_gradient_saturation():
    from src.main.python.environment.reward_engine import (
        objective_d_shaping_generator, zero_d_shaping_generator, linear_reward_generator
    )

    soft_fn = objective_d_shaping_generator(objective=100.0, w=1.0)
    hard_fn = zero_d_shaping_generator(w=10.0)
    linear_fn = linear_reward_generator(w=-1.0)

    # Soft shaping saturates
    assert np.isclose(soft_fn(1e-6), -1.0, atol=1e-3)
    assert soft_fn(1e6) <= 1.0

    # Hard shaping bounded
    assert hard_fn(10.0) < 1.0

    # Linear shaping clipped
    assert linear_fn(100.0) == -100.0


# === Run manually ===
if __name__ == "__main__":
    pytest.main([__file__])
