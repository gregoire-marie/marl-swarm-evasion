import argparse
import numpy as np
import pytest

import src.main.python.utils.rllib_setup as rllib_setup
from src.main.python.utils.rllib_setup import (
    ACTIVE_CURRICULUM_STAGE_INDEX,
    OrbitalRunSpec,
    DEFAULT_MAX_DELTA_V_MPS,
    RUN_PARAMETERS_FILENAME,
    _batch_single_item,
    _get_module_device,
    _unbatch_single_item,
    build_curriculum_stage_env_config_patch,
    build_orbital_env_config,
    build_policies_to_train,
    build_policy_setup,
    build_rllib_env_config,
    compute_deterministic_module_action,
    create_raw_env,
    create_rllib_env,
    find_run_parameters_path,
    get_orbital_env_name,
    get_observation_slot_counts,
    load_run_parameters_from_checkpoint,
    parse_maneuver_frame,
    register_orbital_env,
    rllib_policy_mapping_fn,
    run_parameters_from_spec,
    run_spec_from_args,
    run_spec_from_rllib_env_config,
    run_spec_from_run_parameters,
    save_run_parameters,
    trainable_policies_for_stage,
    validate_run_spec,
)
from src.main.python.experiment.curriculum import CurriculumConfig
from ray.rllib.core.columns import Columns


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


def test_parse_maneuver_frame_normalizes_and_rejects_invalid_values():
    assert parse_maneuver_frame(" tnw ") == "TNW"

    with pytest.raises(argparse.ArgumentTypeError, match="Unsupported maneuver frame"):
        parse_maneuver_frame("LVLH")


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        ("n_interceptors", 0, "--n-interceptors must be >= 1."),
        ("n_targets", 0, "--n-targets must be >= 1."),
        ("episode_length", 0, "--episode-length must be >= 1."),
        ("max_delta_v_mps", 0.0, "--max-delta-v-mps must be > 0."),
        ("timestep", 0.0, "--timestep must be > 0."),
    ],
)
def test_validate_run_spec_rejects_invalid_ranges(field_name, value, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        validate_run_spec(OrbitalRunSpec(**{field_name: value}))


def test_run_spec_from_args_ignores_missing_namespace_attributes():
    args = argparse.Namespace(n_interceptors=2)

    spec = run_spec_from_args(args)

    assert spec == OrbitalRunSpec(n_interceptors=2)


def test_build_env_configs_include_expected_fields():
    spec = OrbitalRunSpec(
        n_interceptors=2,
        n_targets=3,
        timestep=120.0,
        episode_length=250,
        start_time="2026-01-01 00:00:00",
        max_delta_v_mps=15.0,
        maneuver_frame="TNW",
        freeze_targets=True,
        seed=9,
    )

    assert build_orbital_env_config(spec) == {
        "timestep_sec": 120.0,
        "episode_length": 250,
        "start_time": "2026-01-01 00:00:00",
        "max_delta_v_mps": 15.0,
        "maneuver_frame": "TNW",
        "freeze_targets": True,
    }
    assert build_rllib_env_config(spec) == {
        "n_interceptors": 2,
        "n_targets": 3,
        "seed": 9,
        "orbital_env_config": build_orbital_env_config(spec),
    }


def test_curriculum_env_configs_include_active_stage_index():
    curriculum = CurriculumConfig.from_mapping(
        {
            "N_max": 1,
            "M_max": 1,
            "stages": [
                {
                    "stage_id": "S1",
                    "n_interceptors": 1,
                    "n_targets": 1,
                    "disabled_actions": ["targets"],
                    "frozen_policies": [],
                    "trainable_policies": ["interceptor_policy"],
                    "maneuver_frame": "ECI",
                    "propagator": "keplerian",
                    "initial_condition_distribution": "pursuit_evasion",
                    "max_delta_v_mps": 10.0,
                    "episode_length": 5,
                },
            ],
        }
    )
    spec = OrbitalRunSpec()

    orbital_config = build_orbital_env_config(spec, curriculum_config=curriculum)
    rllib_config = build_rllib_env_config(spec, curriculum_config=curriculum)

    assert orbital_config[ACTIVE_CURRICULUM_STAGE_INDEX] == 0
    assert rllib_config[ACTIVE_CURRICULUM_STAGE_INDEX] == 0
    assert rllib_config["orbital_env_config"][ACTIVE_CURRICULUM_STAGE_INDEX] == 0
    assert build_curriculum_stage_env_config_patch(2) == {
        ACTIVE_CURRICULUM_STAGE_INDEX: 2,
        "orbital_env_config": {ACTIVE_CURRICULUM_STAGE_INDEX: 2},
    }


def test_run_parameters_roundtrip_and_observation_slot_counts(tmp_path):
    spec = OrbitalRunSpec(
        n_interceptors=2,
        n_targets=3,
        timestep=120.0,
        episode_length=250,
        start_time="2026-01-01 00:00:00",
        max_delta_v_mps=15.0,
        maneuver_frame="tnw",
        freeze_targets=True,
        seed=9,
    )

    normalized_spec = validate_run_spec(spec)
    run_parameters = run_parameters_from_spec(spec)

    assert run_parameters["maneuver_frame"] == "TNW"
    assert run_parameters["observation_slot_counts"] == {
        "interceptors": 2,
        "targets": 3,
        "total_agents": 5,
        "features_per_agent": 11,
        "observation_dim": 55,
    }
    assert get_observation_slot_counts(normalized_spec) == run_parameters["observation_slot_counts"]
    assert run_spec_from_run_parameters(run_parameters) == normalized_spec

    output_path = save_run_parameters(spec, str(tmp_path))
    assert output_path == str(tmp_path / RUN_PARAMETERS_FILENAME)
    assert (tmp_path / RUN_PARAMETERS_FILENAME).is_file()
    assert run_spec_from_run_parameters(run_parameters) == normalized_spec


def test_load_run_parameters_from_checkpoint_searches_parent_dirs(tmp_path):
    base_spec = OrbitalRunSpec(timestep=12.0, episode_length=34, max_delta_v_mps=56.0)
    trial_dir = tmp_path / "experiment" / "trial"
    checkpoint_dir = trial_dir / "checkpoint_000001"
    checkpoint_dir.mkdir(parents=True)
    run_parameters_path = tmp_path / "experiment" / RUN_PARAMETERS_FILENAME
    run_parameters_path.write_text(
        '{"n_interceptors": 4, "n_targets": 2, "seed": 13}\n',
        encoding="utf-8",
    )

    assert find_run_parameters_path(str(checkpoint_dir)) == str(tmp_path / "experiment" / RUN_PARAMETERS_FILENAME)

    loaded_spec, loaded_path = load_run_parameters_from_checkpoint(str(checkpoint_dir), base_spec=base_spec)

    assert loaded_spec == OrbitalRunSpec(
        n_interceptors=4,
        n_targets=2,
        timestep=12.0,
        episode_length=34,
        max_delta_v_mps=56.0,
        seed=13,
    )
    assert loaded_path == str(tmp_path / "experiment" / RUN_PARAMETERS_FILENAME)


def test_build_agent_configs_delegates_to_scenario_builder(monkeypatch):
    captured = {}

    def fake_scenario_builder(*, n_interceptors, n_targets, seed):
        captured.update(
            n_interceptors=n_interceptors,
            n_targets=n_targets,
            seed=seed,
        )
        return {"interceptor_0": {"role": "interceptor"}}

    monkeypatch.setattr(rllib_setup, "pursuit_evasion_scenario", fake_scenario_builder)

    spec = OrbitalRunSpec(n_interceptors=4, n_targets=2, seed=13)
    agent_configs = rllib_setup.build_agent_configs(spec)

    assert agent_configs == {"interceptor_0": {"role": "interceptor"}}
    assert captured == {"n_interceptors": 4, "n_targets": 2, "seed": 13}


def test_create_raw_env_and_create_rllib_env_use_expected_factories(monkeypatch):
    spec = OrbitalRunSpec()

    class FakeOrbitalEnv:
        metadata = {"name": "fake-orbital-env"}

        def __init__(self, *, agent_configs, env_config):
            self.agent_configs = agent_configs
            self.env_config = env_config

    wrapped = {}
    fake_raw_env = object()

    monkeypatch.setattr(rllib_setup, "OrbitalEnv", FakeOrbitalEnv)
    monkeypatch.setattr(
        rllib_setup,
        "build_agent_configs",
        lambda input_spec: {"spec": input_spec.n_interceptors},
    )

    raw_env = create_raw_env(spec)
    assert raw_env.agent_configs == {"spec": 1}
    assert raw_env.env_config == build_orbital_env_config(spec)

    monkeypatch.setattr(rllib_setup, "run_spec_from_rllib_env_config", lambda config: spec)

    def fake_create_raw_env(input_spec, *, curriculum_config=None, active_curriculum_stage_index=0):
        wrapped["active_stage_index"] = active_curriculum_stage_index
        return fake_raw_env

    monkeypatch.setattr(rllib_setup, "create_raw_env", fake_create_raw_env)

    class FakeParallelPettingZooEnv:
        def __init__(self, env):
            wrapped["env"] = env
            self.env = env

    monkeypatch.setattr(rllib_setup, "ParallelPettingZooEnv", FakeParallelPettingZooEnv)

    wrapped_env = create_rllib_env({"seed": 5})

    assert wrapped["env"] is fake_raw_env
    assert wrapped["active_stage_index"] == 0
    assert wrapped_env.env is fake_raw_env


def test_register_orbital_env_registers_factory(monkeypatch):
    registered = {}

    class FakeOrbitalEnv:
        metadata = {"name": "orbital-test-env"}

    def fake_register_env(name, factory):
        registered["name"] = name
        registered["factory"] = factory

    monkeypatch.setattr(rllib_setup, "OrbitalEnv", FakeOrbitalEnv)
    monkeypatch.setattr(rllib_setup, "register_env", fake_register_env)

    env_name = register_orbital_env()

    assert get_orbital_env_name() == "orbital-test-env"
    assert env_name == "orbital-test-env"
    assert registered == {
        "name": "orbital-test-env",
        "factory": rllib_setup.create_rllib_env,
    }


def test_build_policy_setup_builds_policy_specs(monkeypatch):
    spec = OrbitalRunSpec()

    class FakeOrbitalEnv:
        def __init__(self, *, agent_configs, env_config):
            self.agent_configs = agent_configs
            self.env_config = env_config

        def observation_space(self, agent_id):
            return f"obs:{agent_id}"

        def action_space(self, agent_id):
            return f"act:{agent_id}"

    monkeypatch.setattr(
        rllib_setup,
        "build_agent_configs",
        lambda _: {
            "interceptor_0": {"role": "interceptor"},
            "target_0": {"role": "target"},
        },
    )
    monkeypatch.setattr(rllib_setup, "OrbitalEnv", FakeOrbitalEnv)

    policy_setup = build_policy_setup(spec)

    assert set(policy_setup["policies"]) == {"interceptor_policy", "target_policy"}
    assert policy_setup["interceptor_obs_space"] == "obs:interceptor_0"
    assert policy_setup["target_obs_space"] == "obs:target_0"
    assert policy_setup["policies"]["interceptor_policy"].observation_space == "obs:interceptor_0"
    assert policy_setup["policies"]["interceptor_policy"].action_space == "act:interceptor_0"
    assert policy_setup["policies"]["target_policy"].observation_space == "obs:target_0"
    assert policy_setup["policies"]["target_policy"].action_space == "act:target_0"


def test_build_policy_setup_rejects_empty_agent_configs(monkeypatch):
    class FakeOrbitalEnv:
        def __init__(self, *, agent_configs, env_config):
            self.agent_configs = agent_configs
            self.env_config = env_config

    monkeypatch.setattr(rllib_setup, "build_agent_configs", lambda _: {})
    monkeypatch.setattr(rllib_setup, "OrbitalEnv", FakeOrbitalEnv)

    with pytest.raises(ValueError, match="No policies were created"):
        build_policy_setup(OrbitalRunSpec())


def test_build_policies_to_train_and_policy_mapping():
    assert build_policies_to_train(
        ["interceptor_policy", "target_policy"],
        freeze_targets=True,
    ) == ["interceptor_policy"]
    assert build_policies_to_train(["target_policy"], freeze_targets=True) == ["target_policy"]
    assert rllib_policy_mapping_fn("target_0") == "target_policy"


def test_trainable_policies_for_stage_uses_curriculum_stage():
    curriculum = CurriculumConfig.from_mapping(
        {
            "N_max": 1,
            "M_max": 1,
            "stages": [
                {
                    "stage_id": "S1",
                    "n_interceptors": 1,
                    "n_targets": 1,
                    "disabled_actions": ["targets"],
                    "frozen_policies": [],
                    "trainable_policies": ["interceptor_policy"],
                    "maneuver_frame": "ECI",
                    "propagator": "keplerian",
                    "initial_condition_distribution": "pursuit_evasion",
                    "max_delta_v_mps": 10.0,
                    "episode_length": 5,
                    "advance_when": {
                        "min_iterations": 1,
                        "consecutive_iterations": 1,
                        "conditions": [{"metric": "intercept_success_rate", "operator": ">", "threshold": 0.8}],
                    },
                },
                {
                    "stage_id": "S2",
                    "n_interceptors": 1,
                    "n_targets": 1,
                    "disabled_actions": [],
                    "frozen_policies": ["interceptor_policy"],
                    "trainable_policies": ["target_policy"],
                    "maneuver_frame": "ECI",
                    "propagator": "keplerian",
                    "initial_condition_distribution": "pursuit_evasion",
                    "max_delta_v_mps": 10.0,
                    "episode_length": 5,
                },
            ],
        }
    )

    assert trainable_policies_for_stage(curriculum, 0) == ["interceptor_policy"]
    assert trainable_policies_for_stage(curriculum, 1) == ["target_policy"]

    with pytest.raises(ValueError, match="Unknown curriculum stage index"):
        trainable_policies_for_stage(curriculum, 2)


def test_batch_and_unbatch_helpers_handle_nested_structures():
    item = {
        "obs": np.array([1.0, 2.0]),
        "parts": [np.array([3.0]), (4.0, 5.0)],
    }

    batched = _batch_single_item(item)
    unbatched = _unbatch_single_item(batched)

    assert batched["obs"].shape == (1, 2)
    assert batched["parts"][0].shape == (1, 1)
    assert batched["parts"][1][0].shape == (1,)
    assert np.array_equal(unbatched["obs"], np.array([1.0, 2.0]))
    assert np.array_equal(unbatched["parts"][0], np.array([3.0]))
    assert unbatched["parts"][1] == (4.0, 5.0)


def test_get_module_device_handles_present_and_missing_parameters():
    class Param:
        device = "cpu"

    class DeviceModule:
        def parameters(self):
            return iter([Param()])

    class EmptyModule:
        def parameters(self):
            return iter(())

    class BrokenModule:
        parameters = None

    assert _get_module_device(DeviceModule()) == "cpu"
    assert _get_module_device(EmptyModule()) is None
    assert _get_module_device(BrokenModule()) is None
    assert _get_module_device(object()) is None


def test_compute_deterministic_module_action_uses_actions_and_state(monkeypatch):
    captured = {}

    class FakeModule:
        action_space = "space"

        def parameters(self):
            return iter(())

        def forward_inference(self, batch):
            captured["batch"] = batch
            return {
                Columns.ACTIONS: np.array([[0.25, -0.5]], dtype=np.float32),
                Columns.STATE_OUT: {"memory": np.array([[3.0, 4.0]], dtype=np.float32)},
            }

    monkeypatch.setattr(rllib_setup, "convert_to_torch_tensor", lambda batch, device=None: batch)
    monkeypatch.setattr(rllib_setup, "convert_to_numpy", lambda value: value)
    monkeypatch.setattr(
        rllib_setup.space_utils,
        "unsquash_action",
        lambda action, action_space: action + 1.0,
    )

    action, next_state = compute_deterministic_module_action(
        FakeModule(),
        {"vector": np.array([1.0, 2.0], dtype=np.float32)},
        normalize_actions=True,
        clip_actions=False,
        module_state=(np.array([9.0], dtype=np.float32),),
    )

    assert captured["batch"][Columns.OBS]["vector"].shape == (1, 2)
    assert captured["batch"][Columns.STATE_IN][0].shape == (1, 1)
    assert np.allclose(action, np.array([1.25, 0.5], dtype=np.float32))
    assert np.allclose(next_state["memory"], np.array([3.0, 4.0], dtype=np.float32))


def test_compute_deterministic_module_action_uses_action_distribution_and_clipping(monkeypatch):
    class FakeDistribution:
        def to_deterministic(self):
            return self

        def sample(self):
            return np.array([[2.0, -2.0]], dtype=np.float32)

    class FakeDistributionClass:
        @classmethod
        def from_logits(cls, logits):
            assert logits == "logits"
            return FakeDistribution()

    class FakeModule:
        action_space = "space"

        def parameters(self):
            return iter(())

        def forward_inference(self, batch):
            return {Columns.ACTION_DIST_INPUTS: "logits"}

        def get_inference_action_dist_cls(self):
            return FakeDistributionClass

    monkeypatch.setattr(rllib_setup, "convert_to_torch_tensor", lambda batch, device=None: batch)
    monkeypatch.setattr(rllib_setup, "convert_to_numpy", lambda value: value)
    monkeypatch.setattr(
        rllib_setup.space_utils,
        "clip_action",
        lambda action, action_space: np.clip(action, -1.0, 1.0),
    )

    action, next_state = compute_deterministic_module_action(
        FakeModule(),
        np.array([0.0, 1.0], dtype=np.float32),
        normalize_actions=False,
        clip_actions=True,
    )

    assert np.allclose(action, np.array([1.0, -1.0], dtype=np.float32))
    assert next_state is None


def test_compute_deterministic_module_action_requires_actions_or_logits(monkeypatch):
    class FakeModule:
        action_space = "space"

        def parameters(self):
            return iter(())

        def forward_inference(self, batch):
            return {}

    monkeypatch.setattr(rllib_setup, "convert_to_torch_tensor", lambda batch, device=None: batch)

    with pytest.raises(KeyError, match="must return either"):
        compute_deterministic_module_action(
            FakeModule(),
            np.array([0.0], dtype=np.float32),
            normalize_actions=False,
            clip_actions=False,
        )
