import argparse

from src.main.python.utils.rllib_setup import (
    OrbitalRunSpec,
    DEFAULT_MAX_DELTA_V_MPS,
    run_spec_from_args,
    run_spec_from_rllib_env_config,
)


def test_run_spec_from_rllib_env_config_uses_default_max_delta_v_fallback():
    spec = run_spec_from_rllib_env_config(
        {
            "n_interceptors": 1,
            "n_targets": 1,
            "seed": 7,
            "orbital_env_config": {
                "timestep_sec": 60.0,
                "episode_length": 100,
                "start_time": "2025-01-01 00:00:00",
                "maneuver_frame": "tnw",
                "freeze_targets": True,
            },
        }
    )

    assert spec == OrbitalRunSpec(
        n_interceptors=1,
        n_targets=1,
        timestep=60.0,
        episode_length=100,
        start_time="2025-01-01 00:00:00",
        max_delta_v_mps=DEFAULT_MAX_DELTA_V_MPS,
        maneuver_frame="TNW",
        freeze_targets=True,
        seed=7,
    )


def test_run_spec_from_args_overrides_checkpoint_defaults():
    base_spec = OrbitalRunSpec(
        n_interceptors=1,
        n_targets=1,
        timestep=60.0,
        episode_length=100,
        start_time="2025-01-01 00:00:00",
        max_delta_v_mps=100.0,
        maneuver_frame="ECI",
        freeze_targets=False,
        seed=42,
    )
    args = argparse.Namespace(
        n_interceptors=None,
        n_targets=3,
        timestep=None,
        episode_length=250,
        start_time=None,
        max_delta_v_mps=20.0,
        maneuver_frame="tnw",
        freeze_targets=True,
        seed=99,
    )

    spec = run_spec_from_args(args, base_spec=base_spec)

    assert spec == OrbitalRunSpec(
        n_interceptors=1,
        n_targets=3,
        timestep=60.0,
        episode_length=250,
        start_time="2025-01-01 00:00:00",
        max_delta_v_mps=20.0,
        maneuver_frame="TNW",
        freeze_targets=True,
        seed=99,
    )
