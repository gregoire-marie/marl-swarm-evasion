import numpy as np
from astropy import units as u
from src.main.python.environment.orbital_env import OrbitalEnv


def make_dummy_config(n_agents=2):
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
            "delta_v": 10.0
        }
        agent_configs[agent_id] = config

    env_config = {
        "timestep_sec": 10,
        "episode_length": 10,
        "start_time": "2025-01-01 00:00:00",
        "max_delta_v_kms": 0.1,
    }

    return agent_configs, env_config


def test_orbital_env_reset_and_step():
    agent_configs, env_config = make_dummy_config(n_agents=3)
    env = OrbitalEnv(agent_configs, env_config)

    obs_0 = env.reset()

    # --- Type and structure checks ---
    assert isinstance(obs_0, dict)
    assert set(obs_0.keys()) == set(env.agents)

    for agent_id, ob in obs_0.items():
        # Type
        assert isinstance(ob, np.ndarray)
        assert ob.ndim == 1

        # Shape matches declared observation space
        expected_obs_dim = env.observation_space(agent_id).shape[0]
        assert ob.shape[0] == expected_obs_dim, f"{agent_id} obs shape mismatch"

        # In observation space bounds
        assert env.observation_space(agent_id).contains(ob), f"{agent_id} obs not in observation space"

    # Store pre-step state
    pre_dvs = {aid: env._agent_states[aid].get_total_delta_v().to_value(u.km / u.s) for aid in env.agents}
    pre_obs = {aid: ob.copy() for aid, ob in obs_0.items()}

    # Build zero-action dictionary (no Δv)
    actions = {
        agent_id: np.zeros(3, dtype=np.float32)
        for agent_id in env.agents
    }

    obs_1, rewards, dones, infos = env.step(actions)

    # --- Output structure ---
    assert set(obs_1.keys()) == set(env.agents)
    assert set(rewards.keys()) == set(env.agents)
    assert set(dones.keys()) == set(env.agents).union({"__all__"})
    assert set(infos.keys()) == set(env.agents)

    # --- Post-step checks ---
    for agent_id in env.agents:
        # Types
        assert isinstance(obs_1[agent_id], np.ndarray)
        assert isinstance(rewards[agent_id], float)
        assert isinstance(dones[agent_id], bool)
        assert isinstance(infos[agent_id], dict)

        # Observation should have changed due to propagation
        delta_obs = np.linalg.norm(obs_1[agent_id] - pre_obs[agent_id])
        assert delta_obs > 1e-3, f"Observation for {agent_id} did not change after propagation"

        # Δv should not have changed
        post_dv = env._agent_states[agent_id].get_total_delta_v().to_value(u.km / u.s)
        assert np.isclose(post_dv, pre_dvs[agent_id], atol=1e-6), f"Δv changed for {agent_id} without action"

        # Observation space compliance post-step
        assert env.observation_space(agent_id).contains(obs_1[agent_id]), f"{agent_id} post-step obs out of bounds"

    # Episode should not be done after 1 step
    assert not dones["__all__"], "Episode ended too early"


def test_single_agent_behavior():
    """Edge case test: 1-agent environment."""
    agent_configs, env_config = make_dummy_config(n_agents=1)
    env = OrbitalEnv(agent_configs, env_config)

    obs = env.reset()
    assert len(obs) == 1

    aid = next(iter(obs))
    assert isinstance(obs[aid], np.ndarray)
    assert obs[aid].shape[0] == env.observation_space(aid).shape[0]

    # One step
    actions = {aid: np.zeros(3, dtype=np.float32)}
    obs_1, rewards, dones, infos = env.step(actions)

    assert not dones["__all__"]
    assert aid in obs_1 and aid in rewards and aid in dones


def test_observation_values_are_finite():
    """Check that obs doesn't contain NaN/inf after step."""
    agent_configs, env_config = make_dummy_config(n_agents=4)
    env = OrbitalEnv(agent_configs, env_config)

    obs = env.reset()

    for ob in obs.values():
        assert np.all(np.isfinite(ob)), "NaN or Inf in initial observation"

    actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agents}
    obs, _, _, _ = env.step(actions)

    for ob in obs.values():
        assert np.all(np.isfinite(ob)), "NaN or Inf in step observation"
