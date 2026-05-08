import pytest

from src.main.python.experiment.curriculum import CurriculumConfig
from src.main.python.utils.callbacks import CurriculumCallbacks, OrbitalPhysicsCallbacks
from src.main.python.utils.rllib_setup import (
    ACTIVE_CURRICULUM_STAGE_INDEX,
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
        self.policies_to_train = None

    def environment(self, *, env_config):
        for key, value in env_config.items():
            if key == "orbital_env_config":
                self.env_config.setdefault("orbital_env_config", {}).update(value)
            else:
                self.env_config[key] = value

    def multi_agent(self, *, policies_to_train):
        self.policies_to_train = list(policies_to_train)


class FakeEnvRunner:
    def __init__(self, curriculum, *, worker_id, has_env=True):
        self.worker_id = worker_id
        self.config = FakeConfig(
            {
                "curriculum": curriculum.to_dict(),
                "orbital_env_config": {"curriculum": curriculum.to_dict()},
                ACTIVE_CURRICULUM_STAGE_INDEX: 0,
            }
        )
        self.env = object() if has_env else None
        self.make_env_calls = 0

    def make_env(self):
        self.make_env_calls += 1


class FakeEnvRunnerGroup:
    def __init__(self, local_env_runner, remote_env_runners):
        self.local_env_runner = local_env_runner
        self.remote_env_runners = {
            env_runner.worker_id: env_runner for env_runner in remote_env_runners
        }
        self.calls = []

    def foreach_env_runner(
        self,
        func,
        *,
        remote_worker_ids=None,
        local_env_runner=True,
    ):
        self.calls.append(
            {
                "remote_worker_ids": remote_worker_ids,
                "local_env_runner": local_env_runner,
            }
        )
        results = []
        if local_env_runner and self.local_env_runner is not None:
            results.append(func(self.local_env_runner))
        worker_ids = remote_worker_ids or list(self.remote_env_runners)
        for worker_id in worker_ids:
            results.append(func(self.remote_env_runners[worker_id]))
        return results


class FakeLearner:
    def __init__(self):
        self.config = FakeConfig({})


class FakeLearnerGroup:
    def __init__(self):
        self.learners = [FakeLearner()]
        self.calls = []

    def foreach_learner(self, *, func, timeout_seconds=None):
        self.calls.append(timeout_seconds)
        for learner in self.learners:
            func(learner)


class FakeAlgorithm:
    def __init__(self, curriculum):
        self.config = FakeConfig(
            {
                "curriculum": curriculum.to_dict(),
                "orbital_env_config": {"curriculum": curriculum.to_dict()},
                ACTIVE_CURRICULUM_STAGE_INDEX: 0,
            }
        )
        self.local_env_runner = FakeEnvRunner(curriculum, worker_id=0, has_env=False)
        self.remote_env_runners = [
            FakeEnvRunner(curriculum, worker_id=1),
            FakeEnvRunner(curriculum, worker_id=2),
        ]
        self.env_runner_group = FakeEnvRunnerGroup(
            self.local_env_runner,
            self.remote_env_runners,
        )
        self.learner_group = FakeLearnerGroup()


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
    assert algorithm.env_runner_group.calls == []
    assert algorithm.learner_group.calls == []
    assert [runner.make_env_calls for runner in algorithm.remote_env_runners] == [0, 0]

    result = {"env_runners": {"intercept_success_rate": 1.0}}
    callback.on_train_result(algorithm=algorithm, result=result)
    assert result["curriculum/stage_id"] == "S1"
    assert result["curriculum/stage_index"] == 0
    assert callback.current_stage_index == 0
    assert algorithm.env_runner_group.calls == []

    result = {"env_runners": {"intercept_success_rate": 1.0}}
    callback.on_train_result(algorithm=algorithm, result=result)
    assert result["curriculum/stage_id"] == "S1"
    assert algorithm.env_runner_group.calls == []

    result = {"env_runners": {"intercept_success_rate": 1.0}}
    callback.on_train_result(algorithm=algorithm, result=result)
    assert result["curriculum/stage_id"] == "S2"
    assert result["curriculum/stage_index"] == 1
    assert result["curriculum/stage_transition_count"] == 1
    assert algorithm.learner_group.learners[0].config.policies_to_train == ["target_policy"]
    assert algorithm.env_runner_group.calls == [
        {"remote_worker_ids": None, "local_env_runner": True}
    ]
    assert algorithm.local_env_runner.config.policies_to_train == ["target_policy"]
    assert algorithm.local_env_runner.make_env_calls == 0
    assert [runner.config.env_config[ACTIVE_CURRICULUM_STAGE_INDEX] for runner in algorithm.remote_env_runners] == [
        1,
        1,
    ]
    assert [runner.make_env_calls for runner in algorithm.remote_env_runners] == [1, 1]


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
    assert algorithm.learner_group.learners[0].config.policies_to_train == [
        "interceptor_policy",
        "target_policy",
    ]


def test_curriculum_callback_reapplies_stage_to_recreated_env_runners():
    curriculum = curriculum_for_callback_tests()
    algorithm = FakeAlgorithm(curriculum)
    callback = CurriculumCallbacks()
    callback.on_algorithm_init(algorithm=algorithm)
    callback.current_stage_index = 1

    callback.on_env_runners_recreated(
        algorithm=algorithm,
        env_runner_group=algorithm.env_runner_group,
        env_runner_indices=[2],
        is_evaluation=False,
    )

    assert algorithm.env_runner_group.calls == [
        {"remote_worker_ids": [2], "local_env_runner": False}
    ]
    assert algorithm.remote_env_runners[0].config.env_config[ACTIVE_CURRICULUM_STAGE_INDEX] == 0
    assert algorithm.remote_env_runners[0].make_env_calls == 0
    assert algorithm.remote_env_runners[1].config.env_config[ACTIVE_CURRICULUM_STAGE_INDEX] == 1
    assert algorithm.remote_env_runners[1].config.policies_to_train == ["target_policy"]
    assert algorithm.remote_env_runners[1].make_env_calls == 1


def test_curriculum_callback_requires_env_runner_group_on_transition():
    curriculum = curriculum_for_callback_tests()
    algorithm = FakeAlgorithm(curriculum)
    algorithm.env_runner_group = None
    callback = CurriculumCallbacks()
    callback.on_algorithm_init(algorithm=algorithm)

    with pytest.raises(RuntimeError, match="env_runner_group.foreach_env"):
        result = {"env_runners": {"intercept_success_rate": 1.0}}
        callback.on_train_result(algorithm=algorithm, result=result)
        callback.on_train_result(algorithm=algorithm, result=result)
        callback.on_train_result(algorithm=algorithm, result=result)
