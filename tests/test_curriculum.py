import argparse

import pytest

from src.main.python.experiment.curriculum import CurriculumConfig, CurriculumTrainingConfig, CurriculumTrainingParameters


def valid_curriculum_dict():
    return {
        "N_max": 2,
        "M_max": 2,
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
                "episode_length": 50,
                "advance_when": {
                    "min_iterations": 2,
                    "consecutive_iterations": 3,
                    "conditions": [
                        {"metric": "intercept_success_rate", "operator": ">", "threshold": 0.8}
                    ],
                },
            },
            {
                "stage_id": "S2",
                "n_interceptors": 1,
                "n_targets": 1,
                "disabled_actions": [],
                "frozen_policies": ["interceptor_policy"],
                "trainable_policies": ["target_policy"],
                "maneuver_frame": "TNW",
                "propagator": "keplerian",
                "initial_condition_distribution": "pursuit_evasion",
                "max_delta_v_mps": 20.0,
                "episode_length": 75,
            },
        ],
    }


def test_curriculum_config_loads_and_validates_stages():
    config = CurriculumConfig.from_mapping(valid_curriculum_dict())

    assert config.N_max == 2
    assert config.M_max == 2
    assert [stage.stage_id for stage in config.stages] == ["S1", "S2"]
    assert config.stages[0].stage_index == 0
    assert config.stages[0].advance_when.consecutive_iterations == 3
    assert config.stages[1].advance_when is None
    assert config.stages[1].maneuver_frame == "TNW"


def test_curriculum_training_config_loads_json_owned_training_parameters():
    data = valid_curriculum_dict()
    data.update(
        {
            "iterations": 12,
            "batch-size": 256,
            "lr": 1e-4,
            "gamma": 0.95,
            "num-epochs": 3,
            "num-workers": 0,
            "num-gpus": 0.0,
            "checkpoint-freq": 5,
            "resume": True,
            "name": "curriculum_smoke",
            "local-dir": "/tmp/ray-results",
            "ray-num-cpus": 2,
            "torch-num-threads": 1,
        }
    )

    config = CurriculumTrainingConfig.from_mapping(data)
    namespace = config.training.to_namespace()

    assert config.curriculum.N_max == 2
    assert config.training.num_workers == 0
    assert config.training.ray_num_cpus == 2
    assert config.training.torch_num_threads == 1
    assert namespace == argparse.Namespace(
        iterations=12,
        batch_size=256,
        lr=1e-4,
        gamma=0.95,
        num_epochs=3,
        seed=None,
        num_workers=0,
        num_gpus=0.0,
        checkpoint_freq=5,
        resume=True,
        name="curriculum_smoke",
        local_dir="/tmp/ray-results",
        ray_num_cpus=2,
        torch_num_threads=1,
    )
    assert config.to_dict()["num-workers"] == 0
    assert config.to_dict()["ray-num-cpus"] == 2
    assert config.to_dict()["torch-num-threads"] == 1


def test_curriculum_training_parameters_accept_snake_case_aliases():
    params = CurriculumTrainingParameters.from_mapping(
        {
            "batch_size": 128,
            "num_epochs": 2,
            "num_workers": 0,
            "num_gpus": 0,
            "checkpoint_freq": 0,
            "ray_num_cpus": 1,
            "torch_num_threads": 1,
        }
    )

    assert params.batch_size == 128
    assert params.num_epochs == 2
    assert params.num_workers == 0
    assert params.checkpoint_freq == 0
    assert params.ray_num_cpus == 1
    assert params.torch_num_threads == 1


def test_curriculum_training_parameters_reject_conflicting_aliases():
    with pytest.raises(ValueError, match="Conflicting values"):
        CurriculumTrainingParameters.from_mapping({"num-workers": 1, "num_workers": 2})


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("ray-num-cpus", 0, "ray-num-cpus"),
        ("torch-num-threads", 0, "torch-num-threads"),
        ("num-workers", -1, "num-workers"),
    ],
)
def test_curriculum_training_parameters_reject_invalid_resource_limits(field_name, value, message):
    with pytest.raises(ValueError, match=message):
        CurriculumTrainingParameters.from_mapping({field_name: value})


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.update(N_max=0), "N_max"),
        (lambda data: data["stages"][0].update(n_interceptors=3), "n_interceptors"),
        (lambda data: data["stages"][0].update(disabled_actions=["bogus"]), "disabled_actions"),
        (lambda data: data["stages"][0].update(frozen_policies=["bogus_policy"]), "frozen_policies"),
        (
            lambda data: data["stages"][0].update(
                disabled_actions=[],
                frozen_policies=["interceptor_policy"],
                trainable_policies=["interceptor_policy"],
            ),
            "both frozen and trainable",
        ),
        (lambda data: data["stages"][0].update(propagator="sgp4"), "unsupported propagator"),
        (
            lambda data: data["stages"][0].update(initial_condition_distribution="unknown"),
            "unsupported initial_condition_distribution",
        ),
        (lambda data: data["stages"][0].pop("advance_when"), "requires advance_when"),
    ],
)
def test_curriculum_config_rejects_invalid_stage_definitions(mutator, message):
    data = valid_curriculum_dict()
    mutator(data)

    with pytest.raises(ValueError, match=message):
        CurriculumConfig.from_mapping(data)


def test_disabled_team_cannot_require_policy_actions():
    data = valid_curriculum_dict()
    data["stages"][0]["frozen_policies"] = ["target_policy"]

    with pytest.raises(ValueError, match="disabled targets"):
        CurriculumConfig.from_mapping(data)


def test_final_stage_cannot_define_advance_when():
    data = valid_curriculum_dict()
    data["stages"][1]["advance_when"] = {
        "min_iterations": 1,
        "consecutive_iterations": 1,
        "conditions": [{"metric": "collision_rate", "operator": "<", "threshold": 0.1}],
    }

    with pytest.raises(ValueError, match="Final curriculum stage"):
        CurriculumConfig.from_mapping(data)
