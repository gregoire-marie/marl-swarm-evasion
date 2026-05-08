import numpy as np
import pytest
from ray.rllib.policy.sample_batch import SampleBatch

from src.main.python.experiment.curriculum import CurriculumConfig
from src.main.python.utils.callbacks import CurriculumCallbacks, OrbitalPhysicsCallbacks
from src.main.python.utils.rllib_setup import (
    CURRICULUM_N_INTERCEPTORS,
    CURRICULUM_N_TARGETS,
    CURRICULUM_STAGE_ID,
    CURRICULUM_STAGE_INDEX,
)


class FakeMetricsLogger:
    def __init__(self):
        self.values = {}

    def log_value(self, name, value):
        self.values[name] = value


class NewApiEpisode:
    def __init__(self, infos):
        self._infos = infos

    def get_infos(self, index):
        assert index == -1
        return self._infos


class OldApiEpisode:
    def __init__(self, info):
        self._info = info
        self.custom_metrics = {}

    def last_info_for(self):
        return self._info


class BrokenNewApiEpisode:
    def get_infos(self, index):
        raise RuntimeError("boom")


class BrokenOldApiEpisode:
    def last_info_for(self):
        raise RuntimeError("boom")


class FakeConfig:
    def __init__(self, env_config):
        self.env_config = env_config


class FakeOrbitalEnv:
    def __init__(self):
        self.tasks = []
        self._task = None

    def set_task(self, task):
        self.tasks.append(task["stage_id"])
        self._task = task

    def get_task(self):
        class Task:
            pass

        task = Task()
        task.stage_id = self._task["stage_id"]
        return task


class FakeEnvRunnerGroup:
    def __init__(self, envs):
        self.envs = envs

    def foreach_env(self, fn):
        return [[fn(env)] for env in self.envs]


class FakeAlgorithm:
    def __init__(self, curriculum):
        self.config = FakeConfig({"curriculum": curriculum.to_dict()})
        self.envs = [FakeOrbitalEnv(), FakeOrbitalEnv()]
        self.env_runner_group = FakeEnvRunnerGroup(self.envs)


def curriculum_for_callback_tests():
    return CurriculumConfig.from_mapping(
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
                        "min_iterations": 2,
                        "consecutive_iterations": 2,
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
                    "advance_when": {
                        "min_iterations": 1,
                        "consecutive_iterations": 1,
                        "conditions": [{"metric": "collision_rate", "operator": "<", "threshold": 0.1}],
                        "plateau": {"metric": "episode_return_mean", "window": 3, "min_delta": 0.01},
                    },
                },
                {
                    "stage_id": "S3",
                    "n_interceptors": 1,
                    "n_targets": 1,
                    "disabled_actions": [],
                    "frozen_policies": [],
                    "trainable_policies": ["interceptor_policy", "target_policy"],
                    "maneuver_frame": "ECI",
                    "propagator": "keplerian",
                    "initial_condition_distribution": "pursuit_evasion",
                    "max_delta_v_mps": 10.0,
                    "episode_length": 5,
                },
            ],
        }
    )


def test_on_episode_end_logs_metrics_with_new_api():
    callback = OrbitalPhysicsCallbacks()
    logger = FakeMetricsLogger()
    episode = NewApiEpisode(
        {
            "interceptor_0": {
                "flags": {
                    "intercept_success": True,
                    "interceptors_coll": False,
                    "targets_coll": True,
                    "no_fuel": False,
                    "reentry": True,
                },
                "step": 12,
            }
        }
    )

    callback.on_episode_end(episode=episode, env_index=0, metrics_logger=logger)

    assert logger.values == {
        "intercept_success_rate": 1.0,
        "interceptors_collision_rate": 0.0,
        "targets_collision_rate": 1.0,
        "target_survival_rate": 0.0,
        "collision_rate": 1.0,
        "out_of_fuel_rate": 0.0,
        "reentry_rate": 1.0,
        "episode_steps": 12.0,
    }


def test_on_episode_end_populates_custom_metrics_with_old_api():
    callback = OrbitalPhysicsCallbacks()
    episode = OldApiEpisode(
        {
            "flags": {
                "intercept_success": False,
                "no_fuel": True,
            },
            "step": 7,
        }
    )

    callback.on_episode_end(episode=episode, env_index=0)

    assert episode.custom_metrics == {
        "intercept_success_rate": 0.0,
        "target_survival_rate": 1.0,
        "out_of_fuel_rate": 1.0,
        "episode_steps": 7.0,
    }


def test_on_episode_end_ignores_missing_flags():
    callback = OrbitalPhysicsCallbacks()
    logger = FakeMetricsLogger()
    episode = NewApiEpisode({"interceptor_0": {"step": 5}})

    callback.on_episode_end(episode=episode, env_index=0, metrics_logger=logger)

    assert logger.values == {}


def test_on_episode_end_swallows_new_api_errors():
    callback = OrbitalPhysicsCallbacks()

    callback.on_episode_end(
        episode=BrokenNewApiEpisode(),
        env_index=0,
        metrics_logger=FakeMetricsLogger(),
    )


def test_on_episode_end_swallows_old_api_errors():
    callback = OrbitalPhysicsCallbacks()

    callback.on_episode_end(
        episode=BrokenOldApiEpisode(),
        env_index=0,
    )


def test_curriculum_callback_initializes_and_advances_after_consecutive_success():
    curriculum = curriculum_for_callback_tests()
    algorithm = FakeAlgorithm(curriculum)
    callback = CurriculumCallbacks()

    callback.on_algorithm_init(algorithm=algorithm)
    assert [env.tasks for env in algorithm.envs] == [["S1"], ["S1"]]

    result = {"env_runners": {"intercept_success_rate": 1.0}}
    callback.on_train_result(algorithm=algorithm, result=result)
    assert result["curriculum/stage_id"] == "S1"
    assert result["curriculum/stage_index"] == 0
    assert callback.current_stage_index == 0

    result = {"env_runners": {"intercept_success_rate": 1.0}}
    callback.on_train_result(algorithm=algorithm, result=result)
    assert result["curriculum/stage_id"] == "S1"

    result = {"env_runners": {"intercept_success_rate": 1.0}}
    callback.on_train_result(algorithm=algorithm, result=result)
    assert result["curriculum/stage_id"] == "S2"
    assert result["curriculum/stage_index"] == 1
    assert result["curriculum/stage_transition_count"] == 1
    assert [env.tasks for env in algorithm.envs] == [["S1", "S2"], ["S1", "S2"]]


def test_curriculum_callback_plateau_transition():
    curriculum = curriculum_for_callback_tests()
    algorithm = FakeAlgorithm(curriculum)
    callback = CurriculumCallbacks()
    callback.on_algorithm_init(algorithm=algorithm)

    callback.current_stage_index = 1
    callback.iterations_in_stage = 0
    for value in [10.0, 10.005, 10.006]:
        result = {"env_runners": {"episode_return_mean": value, "collision_rate": 0.0}}
        callback.on_train_result(algorithm=algorithm, result=result)

    assert callback.current_stage_index == 2
    assert result["curriculum/stage_id"] == "S3"


def test_curriculum_callback_adds_batch_audit_metadata():
    callback = CurriculumCallbacks()
    batch = SampleBatch(
        {
            SampleBatch.INFOS: np.asarray(
                [
                    {
                        CURRICULUM_STAGE_ID: "S1",
                        CURRICULUM_STAGE_INDEX: 0,
                        CURRICULUM_N_INTERCEPTORS: 1,
                        CURRICULUM_N_TARGETS: 2,
                    },
                    {
                        CURRICULUM_STAGE_ID: "S1",
                        CURRICULUM_STAGE_INDEX: 0,
                        CURRICULUM_N_INTERCEPTORS: 1,
                        CURRICULUM_N_TARGETS: 2,
                    },
                ],
                dtype=object,
            )
        }
    )

    callback.on_postprocess_trajectory(
        worker=None,
        episode=None,
        agent_id="interceptor_0",
        policy_id="interceptor_policy",
        policies={},
        postprocessed_batch=batch,
        original_batches={},
    )

    assert np.array_equal(batch[CURRICULUM_STAGE_INDEX], np.array([0, 0], dtype=np.int32))
    assert np.array_equal(batch[CURRICULUM_N_INTERCEPTORS], np.array([1, 1], dtype=np.int32))
    assert list(batch[CURRICULUM_STAGE_ID]) == ["S1", "S1"]


def test_curriculum_callback_requires_env_runner_group():
    curriculum = curriculum_for_callback_tests()
    algorithm = FakeAlgorithm(curriculum)
    algorithm.env_runner_group = None
    callback = CurriculumCallbacks()

    with pytest.raises(RuntimeError, match="env_runner_group.foreach_env"):
        callback.on_algorithm_init(algorithm=algorithm)
