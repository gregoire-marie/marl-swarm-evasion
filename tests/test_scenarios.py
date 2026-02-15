import pytest
import numpy as np
from src.main.python.environment.scenarios import (
    get_random_leo_elements,
    pursuit_evasion_scenario,
    constellation_scenario
)

def test_get_random_leo_elements():
    rng = np.random.default_rng(42)
    elements = get_random_leo_elements(rng)
    assert len(elements) == 6
    # a, e, inc, raan, argp, m
    assert 400.0 <= (elements[0].to_value('km') - 6378.137) <= 600.0
    assert 0.0 <= elements[1].value <= 0.001
    assert 0.0 <= elements[2].to_value('deg') <= 98.0

def test_pursuit_evasion_scenario():
    agent_configs, env_config = pursuit_evasion_scenario(n_interceptors=2, n_targets=1, seed=42)
    assert len(agent_configs) == 3
    assert "interceptor_0" in agent_configs
    assert "interceptor_1" in agent_configs
    assert "target_0" in agent_configs
    assert agent_configs["interceptor_0"]["role"] == "interceptor"
    assert agent_configs["target_0"]["role"] == "target"
    assert env_config["timestep_sec"] == 60.0
    assert env_config["maneuver_frame"] == "ECI"

def test_constellation_scenario():
    agent_configs, env_config = constellation_scenario(n_agents=4, seed=123)
    assert len(agent_configs) == 4
    roles = [config["role"] for config in agent_configs.values()]
    assert roles.count("interceptor") == 2
    assert roles.count("target") == 2
    assert env_config["episode_length"] == 100
    assert env_config["maneuver_frame"] == "ECI"


def test_scenarios_accept_tnw_frame():
    _, pe_env = pursuit_evasion_scenario(
        n_interceptors=1,
        n_targets=1,
        seed=1,
        maneuver_frame="tnw",
    )
    _, const_env = constellation_scenario(n_agents=2, seed=2, maneuver_frame="TNW")

    assert pe_env["maneuver_frame"] == "TNW"
    assert const_env["maneuver_frame"] == "TNW"


def test_scenarios_reject_invalid_maneuver_frame():
    with pytest.raises(ValueError):
        pursuit_evasion_scenario(maneuver_frame="BAD_FRAME")

    with pytest.raises(ValueError):
        constellation_scenario(maneuver_frame="X")
