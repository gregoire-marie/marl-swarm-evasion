from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID, EpisodeType

from src.main.python.experiment.curriculum import CurriculumConfig, CurriculumTask
from src.main.python.utils.rllib_setup import (
    build_curriculum_stage_env_config_patch,
    get_active_curriculum_stage_index,
    trainable_policies_for_stage,
)


def configure_env_runner_for_curriculum_stage(
    env_runner: Any,
    *,
    env_config_patch: Mapping[str, Any],
    trainable_policies: List[str],
) -> Optional[int]:
    env_runner.config.environment(env_config=dict(env_config_patch))
    env_runner.config.multi_agent(policies_to_train=list(trainable_policies))
    if getattr(env_runner, "env", None) is not None:
        env_runner.make_env()
    return get_active_curriculum_stage_index(getattr(env_runner.config, "env_config", None))


def configure_learner_for_curriculum_stage(
    learner: Any,
    *,
    trainable_policies: List[str],
) -> None:
    learner.config.multi_agent(policies_to_train=list(trainable_policies))


class OrbitalPhysicsCallbacks(DefaultCallbacks):
    """
    Custom callbacks for RLlib to log orbital-specific metrics.
    
    This class extracts 'flags' from the environment's info dictionary
    at the end of each episode and logs them as custom metrics for TensorBoard.
    """

    def on_episode_end(
        self,
        *,
        episode: EpisodeType,
        env_runner: Optional[Any] = None,
        metrics_logger: Optional[Any] = None,
        env: Optional[Any] = None,
        env_index: int,
        rl_module: Optional[Any] = None,
        # Deprecated arguments for backward compatibility
        worker: Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ):
        """
        Called when an episode is done.
        """
        # Extract flags from the last info
        flags = {}
        episode_steps = 0

        if hasattr(episode, "get_infos"):
            # MultiAgentEpisode (new API stack)
            try:
                last_infos_dict = episode.get_infos(-1)
                if last_infos_dict:
                    # Pick any agent's info since flags are global
                    agent_info = next(iter(last_infos_dict.values()))
                    flags = agent_info.get("flags", {})
                    episode_steps = agent_info.get("step", 0)
            except Exception:
                pass
        elif hasattr(episode, "last_info_for"):
            # Old Episode object
            try:
                last_info = episode.last_info_for()
                if last_info:
                    flags = last_info.get("flags", {})
                    episode_steps = last_info.get("step", 0)
            except Exception:
                pass

        if not flags:
            return

        # Prepare metrics to log
        metrics = {}
        if "intercept_success" in flags:
            metrics["intercept_success_rate"] = float(flags["intercept_success"])
        if "interceptors_coll" in flags:
            metrics["interceptors_collision_rate"] = float(flags["interceptors_coll"])
        if "targets_coll" in flags:
            metrics["targets_collision_rate"] = float(flags["targets_coll"])
        if "intercept_success" in flags:
            metrics["target_survival_rate"] = 1.0 - float(flags["intercept_success"])
        if "interceptors_coll" in flags or "targets_coll" in flags:
            metrics["collision_rate"] = float(
                bool(flags.get("interceptors_coll", False)) or bool(flags.get("targets_coll", False))
            )
        if "no_fuel" in flags:
            metrics["out_of_fuel_rate"] = float(flags["no_fuel"])
        if "reentry" in flags:
            metrics["reentry_rate"] = float(flags["reentry"])

        metrics["episode_steps"] = float(episode_steps)

        # Log metrics using the appropriate method
        if metrics_logger is not None:
            for name, value in metrics.items():
                metrics_logger.log_value(name, value)
        elif hasattr(episode, "custom_metrics"):
            for name, value in metrics.items():
                episode.custom_metrics[name] = value


class CurriculumCallbacks(OrbitalPhysicsCallbacks):
    """Stateful RLlib callback that owns curriculum progression."""

    def __init__(self):
        super().__init__()
        self.curriculum_config: Optional[CurriculumConfig] = None
        self.current_stage_index = 0
        self.iterations_in_stage = 0
        self.stage_transition_count = 0
        self._consecutive_success_count = 0
        self._metric_history: Dict[str, List[float]] = defaultdict(list)
        self._initialized = False

    def on_algorithm_init(
        self,
        *,
        algorithm: Any,
        metrics_logger: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self._ensure_initialized(algorithm)

    def on_train_result(
        self,
        *,
        algorithm: Any,
        metrics_logger: Optional[Any] = None,
        result: dict,
        **kwargs,
    ) -> None:
        self._ensure_initialized(algorithm)
        self.iterations_in_stage += 1

        task = self._current_task()
        should_advance = self._should_advance(task, result)
        if should_advance and self.current_stage_index < len(self.curriculum_config.stages) - 1:
            self.current_stage_index += 1
            self.iterations_in_stage = 0
            self.stage_transition_count += 1
            self._consecutive_success_count = 0
            self._metric_history.clear()
            task = self._current_task()
            self._apply_stage_to_algorithm(algorithm, task)

        self._log_curriculum_metrics(task, result, metrics_logger)

    def on_env_runners_recreated(
        self,
        *,
        algorithm: Any,
        env_runner_group: Any,
        env_runner_indices: List[int],
        is_evaluation: bool,
        **kwargs,
    ) -> None:
        self._ensure_initialized(algorithm)
        task = self._current_task()
        self._apply_stage_to_env_runners(
            env_runner_group,
            task,
            remote_worker_ids=env_runner_indices,
            local_env_runner=False,
        )

    def _ensure_initialized(self, algorithm: Any) -> None:
        if self._initialized:
            return
        env_config = getattr(getattr(algorithm, "config", None), "env_config", None)
        if env_config is None and hasattr(algorithm, "config"):
            env_config = getattr(algorithm.config, "get", lambda *_: None)("env_config", None)
        curriculum_data = None
        if isinstance(env_config, Mapping):
            curriculum_data = env_config.get("curriculum") or dict(env_config.get("orbital_env_config", {})).get(
                "curriculum"
            )
        if curriculum_data is None:
            raise ValueError("CurriculumCallbacks requires curriculum metadata in RLlib env_config.")
        self.curriculum_config = CurriculumConfig.from_mapping(curriculum_data)
        self.current_stage_index = get_active_curriculum_stage_index(env_config)
        self.iterations_in_stage = 0
        self.stage_transition_count = 0
        self._consecutive_success_count = 0
        self._metric_history.clear()
        self._initialized = True

    def _current_task(self) -> CurriculumTask:
        if self.curriculum_config is None:
            raise ValueError("Curriculum callback has not been initialized.")
        return self.curriculum_config.task_by_index(self.current_stage_index)

    def _should_advance(self, task: CurriculumTask, result: Mapping[str, Any]) -> bool:
        advance_when = task.advance_when
        if advance_when is None:
            return False
        for condition in advance_when.conditions:
            metric_value = _lookup_metric(result, condition.metric)
            if metric_value is not None:
                self._metric_history[condition.metric].append(float(metric_value))
        if advance_when.plateau is not None:
            plateau_value = _lookup_metric(result, advance_when.plateau.metric)
            if plateau_value is not None:
                self._metric_history[advance_when.plateau.metric].append(float(plateau_value))

        if self.iterations_in_stage < advance_when.min_iterations:
            self._consecutive_success_count = 0
            return False

        conditions_met = True
        for condition in advance_when.conditions:
            metric_value = _lookup_metric(result, condition.metric)
            conditions_met = conditions_met and metric_value is not None and condition.evaluate(float(metric_value))

        if advance_when.plateau is not None:
            values = self._metric_history[advance_when.plateau.metric]
            conditions_met = conditions_met and advance_when.plateau.evaluate(values)

        if conditions_met:
            self._consecutive_success_count += 1
        else:
            self._consecutive_success_count = 0
        return self._consecutive_success_count >= advance_when.consecutive_iterations

    def _apply_stage_to_algorithm(self, algorithm: Any, task: CurriculumTask) -> None:
        self._sync_trainability(algorithm, task)
        self._apply_stage_to_env_runners(
            getattr(algorithm, "env_runner_group", None),
            task,
        )

    def _sync_trainability(self, algorithm: Any, task: CurriculumTask) -> None:
        trainable_policies = trainable_policies_for_stage(self.curriculum_config, task.stage_index)
        learner_group = getattr(algorithm, "learner_group", None)
        if learner_group is None or not hasattr(learner_group, "foreach_learner"):
            return
        learner_group.foreach_learner(
            func=partial(
                configure_learner_for_curriculum_stage,
                trainable_policies=trainable_policies,
            ),
            timeout_seconds=0.0,
        )

    def _apply_stage_to_env_runners(
        self,
        env_runner_group: Any,
        task: CurriculumTask,
        *,
        remote_worker_ids: Optional[List[int]] = None,
        local_env_runner: bool = True,
    ) -> None:
        if env_runner_group is None or not hasattr(env_runner_group, "foreach_env_runner"):
            raise RuntimeError("Installed RLlib does not expose algorithm.env_runner_group.foreach_env_runner().")

        env_runner_group.foreach_env_runner(
            func=partial(
                configure_env_runner_for_curriculum_stage,
                env_config_patch=build_curriculum_stage_env_config_patch(task.stage_index),
                trainable_policies=trainable_policies_for_stage(self.curriculum_config, task.stage_index),
            ),
            remote_worker_ids=remote_worker_ids,
            local_env_runner=local_env_runner,
        )

    def _log_curriculum_metrics(
        self,
        task: CurriculumTask,
        result: dict,
        metrics_logger: Optional[Any],
    ) -> None:
        metrics = {
            "curriculum/stage_index": int(task.stage_index),
            "curriculum/n_interceptors": int(task.n_interceptors),
            "curriculum/n_targets": int(task.n_targets),
            "curriculum/iterations_in_stage": int(self.iterations_in_stage),
            "curriculum/stage_transition_count": int(self.stage_transition_count),
        }
        result["curriculum/stage_id"] = task.stage_id
        result.update(metrics)
        if metrics_logger is not None:
            for name, value in metrics.items():
                metrics_logger.log_value(name, value)


def _lookup_metric(result: Mapping[str, Any], metric_name: str) -> Optional[float]:
    candidates = [
        result,
        result.get("env_runners", {}) if isinstance(result.get("env_runners"), Mapping) else {},
        result.get("custom_metrics", {}) if isinstance(result.get("custom_metrics"), Mapping) else {},
        result.get("sampler_results", {}) if isinstance(result.get("sampler_results"), Mapping) else {},
    ]
    for candidate in candidates:
        if metric_name in candidate:
            return float(candidate[metric_name])
        mean_name = f"{metric_name}_mean"
        if mean_name in candidate:
            return float(candidate[mean_name])
    return None
