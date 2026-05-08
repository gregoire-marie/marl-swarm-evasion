import pytest

from src.main.python.experiment.curriculum import CurriculumConfig


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
