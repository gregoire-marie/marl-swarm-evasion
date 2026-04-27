import numpy as np
from astropy import units as u
from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.agents.orbit_state import OrbitState
from src.main.python.orbital_meca.orbits import compute_eci_distance


def make_dummy_config(n_agents=2, maneuver_frame="ECI"):
    """Returns test config with `n_agents` in LEO with spaced RAANs."""
    base_alt = 500.0  # km
    base_a = (6378.0 + base_alt) * u.km
    agent_configs = {}

    for i in range(n_agents):
        agent_id = f"agent_{i}"
        config = {
            "role": "interceptor" if i % 2 == 0 else "target",
            "init_orbit": (
                base_a,
                0.00033 * u.one,
                51.6 * u.deg,
                (i * 10.0) * u.deg,  # RAAN offset
                0 * u.deg,
                0 * u.deg
            ),
            "init_delta_v": 10.0
        }
        agent_configs[agent_id] = config

    env_config = {
        "timestep_sec": 10,
        "episode_length": 10,
        "start_time": "2025-01-01 00:00:00",
        "max_delta_v_kms": 0.1,
        "maneuver_frame": maneuver_frame,
    }

    return agent_configs, env_config


def test_orbital_env_reset_and_step():
    agent_configs, env_config = make_dummy_config(n_agents=3)
    env = OrbitalEnv(agent_configs, env_config)

    obs_0, infos_0 = env.reset()

    # --- Type and structure checks ---
    assert isinstance(obs_0, dict)
    assert set(obs_0.keys()) == set(env.agents)
    assert isinstance(infos_0, dict)
    assert set(infos_0.keys()) == set(env.agents)

    for agent_id, ob in obs_0.items():
        # Type
        assert isinstance(ob, np.ndarray)
        assert ob.ndim == 1
        assert ob.dtype == np.float32

        # Shape matches declared observation space
        obs_space = env.observation_space(agent_id)
        expected_obs_dim = obs_space.shape[0]
        assert ob.shape[0] == expected_obs_dim, f"{agent_id} obs shape mismatch"
        assert obs_space.low.dtype == np.float32
        assert obs_space.high.dtype == np.float32

        # In observation space bounds
        assert obs_space.contains(ob), f"{agent_id} obs not in observation space"

    # Store pre-step state
    pre_dvs = {aid: env._agent_states[aid].get_used_delta_v().to_value(u.km / u.s) for aid in env.agents}
    pre_obs = {aid: ob.copy() for aid, ob in obs_0.items()}

    # Build zero-action dictionary (no Δv)
    actions = {
        agent_id: np.zeros(3, dtype=np.float32)
        for agent_id in env.agents
    }

    obs_1, rewards, terms, truncs, infos = env.step(actions)

    # --- Output structure ---
    assert set(obs_1.keys()) == set(env.agents)
    assert set(rewards.keys()) == set(env.agents)
    assert set(terms.keys()) == set(env.agents)
    assert set(truncs.keys()) == set(env.agents)
    assert set(infos.keys()) == set(env.agents)

    # --- Post-step checks ---
    for agent_id in env.agents:
        # Types
        assert isinstance(obs_1[agent_id], np.ndarray)
        assert obs_1[agent_id].dtype == np.float32
        assert isinstance(rewards[agent_id], float)
        assert isinstance(terms[agent_id], bool)
        assert isinstance(truncs[agent_id], bool)
        assert isinstance(infos[agent_id], dict)

        # Observation should have changed due to propagation
        delta_obs = np.linalg.norm(obs_1[agent_id] - pre_obs[agent_id])
        assert delta_obs > 1e-3, f"Observation for {agent_id} did not change after propagation"

        # Δv should not have changed
        post_dv = env._agent_states[agent_id].get_used_delta_v().to_value(u.km / u.s)
        assert np.isclose(post_dv, pre_dvs[agent_id], atol=1e-6), f"Δv changed for {agent_id} without action"

        # Observation space compliance post-step
        assert env.observation_space(agent_id).contains(obs_1[agent_id]), f"{agent_id} post-step obs out of bounds"

    # Episode should not be done after 1 step
    assert not any(terms.values()), "Episode ended too early (termination)"
    assert not any(truncs.values()), "Episode ended too early (truncation)"


def test_single_agent_behavior():
    """Edge case test: 1-agent environment."""
    agent_configs, env_config = make_dummy_config(n_agents=1)
    env = OrbitalEnv(agent_configs, env_config)

    obs, infos = env.reset()
    assert len(obs) == 1

    aid = next(iter(obs))
    assert isinstance(obs[aid], np.ndarray)
    assert obs[aid].shape[0] == env.observation_space(aid).shape[0]

    # One step
    actions = {aid: np.zeros(3, dtype=np.float32)}
    obs_1, rewards, terms, truncs, infos = env.step(actions)

    assert not terms[aid]
    assert aid in obs_1 and aid in rewards and aid in terms


def test_observation_values_are_finite():
    """Check that obs doesn't contain NaN/inf after step."""
    agent_configs, env_config = make_dummy_config(n_agents=4)
    env = OrbitalEnv(agent_configs, env_config)

    obs, infos = env.reset()

    for ob in obs.values():
        assert np.all(np.isfinite(ob)), "NaN or Inf in initial observation"

    actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agents}
    obs, _, _, _, _ = env.step(actions)

    for ob in obs.values():
        assert np.all(np.isfinite(ob)), "NaN or Inf in step observation"


def test_action_clipping_and_render():
    agent_configs, env_config = make_dummy_config(n_agents=2)
    env = OrbitalEnv(agent_configs, env_config)
    env.reset()

    # Action exceeds max_delta_v (0.1 km/s)
    large_action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    actions = {"agent_0": large_action}

    obs, rewards, terms, truncs, infos = env.step(actions)

    # Check used delta-v is clipped to max_delta_v (0.1)
    used_dv = env._agent_states["agent_0"].get_used_delta_v().to_value(u.km / u.s)
    assert np.isclose(used_dv, 0.1, atol=1e-6)

    # Test render (smoke test)
    env.render()


def test_empty_actions():
    agent_configs, env_config = make_dummy_config(n_agents=2)
    env = OrbitalEnv(agent_configs, env_config)
    env.reset()

    # Passing empty dict for actions
    obs, rewards, terms, truncs, infos = env.step({})
    assert len(rewards) == 2
    # Rewards could be positive due to shaping (proximity), but types should be float
    assert all(isinstance(r, float) for r in rewards.values())


def test_action_space_contains():
    agent_configs, env_config = make_dummy_config(n_agents=1)
    env = OrbitalEnv(agent_configs, env_config)
    space = env.action_space("agent_0")
    assert space.shape == (3,)
    assert space.contains(np.array([0.05, 0.05, 0.05], dtype=np.float32))
    assert not space.contains(np.array([0.2, 0.0, 0.0], dtype=np.float32))


def test_telemetry_helpers():
    agent_configs, env_config = make_dummy_config(n_agents=2)
    env = OrbitalEnv(agent_configs, env_config)
    env.reset()

    agent_ids = env.agents
    aid0, aid1 = agent_ids

    position = env.get_position_km(aid0)
    assert isinstance(position, np.ndarray)
    assert position.shape == (3,)
    assert np.all(np.isfinite(position))

    initial_remaining_dv = env.get_remaining_delta_v_kms(aid0)
    assert np.isclose(initial_remaining_dv, 10.0, atol=1e-6)

    initial_distances = env.get_pairwise_distances_km()
    assert set(initial_distances.keys()) == {f"{aid0}__{aid1}"}
    expected_initial_distance = float(
        compute_eci_distance(
            env._agent_states[aid0].orbit_state,
            env._agent_states[aid1].orbit_state,
        )
    )
    assert np.isclose(initial_distances[f"{aid0}__{aid1}"], expected_initial_distance, atol=1e-6)

    env.step({aid0: np.array([0.1, 0.0, 0.0], dtype=np.float32), aid1: np.zeros(3, dtype=np.float32)})

    updated_remaining_dv = env.get_remaining_delta_v_kms(aid0)
    assert np.isclose(updated_remaining_dv, 9.9, atol=1e-6)

    updated_distances = env.get_pairwise_distances_km()
    expected_updated_distance = float(
        compute_eci_distance(
            env._agent_states[aid0].orbit_state,
            env._agent_states[aid1].orbit_state,
        )
    )
    assert np.isclose(updated_distances[f"{aid0}__{aid1}"], expected_updated_distance, atol=1e-6)


def test_tnw_maneuver_frame_applies_expected_eci_burn():
    agent_configs, env_config = make_dummy_config(n_agents=1, maneuver_frame="TNW")
    env = OrbitalEnv(agent_configs, env_config)
    env.reset()

    agent_id = env.agents[0]
    agent_state = env._agent_states[agent_id].orbit_state
    burn_time = agent_state.epoch + env.timestep

    # Build an independent reference state at maneuver time (before burn).
    shadow_state = OrbitState(agent_state.get_keplerian(), agent_state.epoch)
    shadow_state.propagate_to(burn_time)
    _, v_before = shadow_state.get_rv()

    raw_action = np.array([0.2, -0.01, 0.0], dtype=np.float32)  # exceeds max_delta_v
    clipped = raw_action * (env.max_delta_v / np.linalg.norm(raw_action))
    expected_dv_eci = shadow_state.tnw_to_eci(clipped * u.km / u.s)

    env.step({agent_id: raw_action})
    _, v_after = env._agent_states[agent_id].orbit_state.get_rv()

    measured_dv = (v_after - v_before).to_value(u.km / u.s)
    assert np.allclose(measured_dv, expected_dv_eci.to_value(u.km / u.s), atol=1e-8)


def test_invalid_maneuver_frame_raises():
    agent_configs, env_config = make_dummy_config(n_agents=1, maneuver_frame="BAD_FRAME")
    with np.testing.assert_raises(ValueError):
        OrbitalEnv(agent_configs, env_config)
