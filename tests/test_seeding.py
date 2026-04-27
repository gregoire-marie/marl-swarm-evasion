import os
import random as _random

import numpy as np
import pytest
from astropy import units as u

from src.main.python.utils.random import set_global_seed
from src.main.python.environment.orbital_env import OrbitalEnv


def _snapshot_rng_state():
    """Capture Python, NumPy RNG states and PYTHONHASHSEED env var."""
    py_state = _random.getstate()
    np_state = np.random.get_state()
    env_phs = os.environ.get("PYTHONHASHSEED")
    return py_state, np_state, env_phs


def _restore_rng_state(py_state, np_state, env_phs):
    """Restore previously captured RNG states and PYTHONHASHSEED env var."""
    _random.setstate(py_state)
    np.random.set_state(np_state)
    if env_phs is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = env_phs


def test_set_global_seed_reproducible_and_sets_env():
    py_state, np_state, env_phs = _snapshot_rng_state()
    try:
        s = 123
        r1 = set_global_seed(s)
        # sequences from Python's random and NumPy must be reproducible with same seed
        py_seq_1 = [_random.random() for _ in range(5)]
        np_seq_1 = np.random.random(5).tolist()

        r2 = set_global_seed(s)
        py_seq_2 = [_random.random() for _ in range(5)]
        np_seq_2 = np.random.random(5).tolist()

        assert py_seq_1 == py_seq_2, "Python RNG not reproducible for same seed"
        assert np.allclose(np_seq_1, np_seq_2), "NumPy RNG not reproducible for same seed"

        # Environment variable must be set to the seed string
        assert os.environ.get("PYTHONHASHSEED") == str(s)

        # Return value indicates which components were seeded (torch optional)
        assert r1.get("python") is True and r2.get("python") is True
        assert r1.get("numpy") is True and r2.get("numpy") is True
        assert "torch" in r1 and isinstance(r1["torch"], bool)
    finally:
        _restore_rng_state(py_state, np_state, env_phs)


def test_set_global_seed_different_seeds_diverge():
    py_state, np_state, env_phs = _snapshot_rng_state()
    try:
        set_global_seed(1)
        py_a = [_random.random() for _ in range(3)]
        np_a = np.random.random(3).tolist()

        set_global_seed(2)
        py_b = [_random.random() for _ in range(3)]
        np_b = np.random.random(3).tolist()

        # Very high probability to differ
        assert py_a != py_b, "Python RNG sequences unexpectedly equal for different seeds"
        assert not np.allclose(np_a, np_b), "NumPy RNG sequences unexpectedly equal for different seeds"
    finally:
        _restore_rng_state(py_state, np_state, env_phs)


def _make_dummy_env(n_agents=2):
    base_alt = 500.0  # km
    base_a = (6378.0 + base_alt) * u.km
    agent_configs = {}
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        agent_configs[agent_id] = {
            "role": "interceptor" if i % 2 == 0 else "target",
            "init_orbit": (
                base_a,
                0.00033 * u.one,
                51.6 * u.deg,
                (i * 10.0) * u.deg,
                0 * u.deg,
                0 * u.deg,
            ),
            "init_delta_v": 10000.0,
        }
    env_config = {
        "timestep_sec": 10,
        "episode_length": 5,
        "start_time": "2025-01-01 00:00:00",
        "max_delta_v_mps": 100.0,
    }
    return OrbitalEnv(agent_configs, env_config)


def test_orbital_env_reset_seeds_globals_and_is_reproducible():
    py_state, np_state, env_phs = _snapshot_rng_state()
    try:
        env = _make_dummy_env(n_agents=2)
        seed = 4242
        obs0, infos0 = env.reset(seed=seed)
        assert isinstance(obs0, dict)
        assert isinstance(infos0, dict)
        # env stores the seed
        assert getattr(env, "_seed", None) == seed
        # PYTHONHASHSEED reflects the provided seed
        assert os.environ.get("PYTHONHASHSEED") == str(seed)

        # Immediately after reset, NumPy random should be deterministic for same seed
        np_draw_1 = np.random.random(4).tolist()

        # Re-seed via reset with the same seed and draw again
        env.reset(seed=seed)
        np_draw_2 = np.random.random(4).tolist()

        assert np.allclose(np_draw_1, np_draw_2), "NumPy draws differ after same-seed env.reset()"

        # Different seed should change the sequence
        env.reset(seed=seed + 1)
        np_draw_3 = np.random.random(4).tolist()
        assert not np.allclose(np_draw_1, np_draw_3), "NumPy draws unexpectedly equal after different seed"
    finally:
        _restore_rng_state(py_state, np_state, env_phs)


def test_try_seed_torch_with_mocks(mocker):
    # Mock torch module
    mock_torch = mocker.Mock()
    # Also mock backends.cudnn
    mock_torch.backends.cudnn = mocker.Mock()
    
    mocker.patch.dict("sys.modules", {"torch": mock_torch})
    
    from src.main.python.utils.random import _try_seed_torch
    
    # Case 1: CUDA available
    mock_torch.cuda.is_available.return_value = True
    res = _try_seed_torch(42)
    assert res is True
    mock_torch.manual_seed.assert_called_with(42)
    mock_torch.cuda.manual_seed_all.assert_called_with(42)
    assert mock_torch.backends.cudnn.deterministic is True
    
    # Case 2: CUDA not available
    mock_torch.cuda.is_available.return_value = False
    _try_seed_torch(42)
    
    # Case 3: Exception during seeding
    mock_torch.manual_seed.side_effect = Exception("Fail")
    res = _try_seed_torch(42)
    assert res is False

def test_try_seed_torch_cudnn_fail(mocker):
    # Mock torch module
    mock_torch = mocker.Mock()
    mocker.patch.dict("sys.modules", {"torch": mock_torch})
    
    # Mock backends.cudnn to raise exception when setting deterministic
    mock_cudnn = mocker.Mock()
    mock_torch.backends.cudnn = mock_cudnn
    
    # Use PropertyMock to raise exception on assignment
    type(mock_cudnn).deterministic = mocker.PropertyMock(side_effect=Exception("Cudnn Fail"))
    
    from src.main.python.utils.random import _try_seed_torch
    # Should still return True because of the try-except pass around cudnn setup
    res = _try_seed_torch(42)
    assert res is True

def test_set_global_seed_numpy_fail(mocker):
    # Mock numpy.random.seed to fail
    mocker.patch("numpy.random.seed", side_effect=Exception("NP Fail"))
    from src.main.python.utils.random import set_global_seed
    
    res = set_global_seed(42)
    assert res["numpy"] is False
