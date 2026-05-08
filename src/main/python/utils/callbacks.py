from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional
import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID, EpisodeType

from src.main.python.experiment.curriculum import CurriculumConfig, CurriculumTask
from src.main.python.utils.rllib_setup import (
    CURRICULUM_N_INTERCEPTORS,
    CURRICULUM_N_TARGETS,
    CURRICULUM_STAGE_ID,
    CURRICULUM_STAGE_INDEX,
)


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
        self._broadcast_task(algorithm, self._current_task())

    def on_postprocess_trajectory(
        self,
        *,
        worker: Any,
        episode: Any,
        agent_id: Any,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[Any, Any],
        **kwargs,
    ) -> None:
        infos = postprocessed_batch.get(SampleBatch.INFOS)
        if infos is None:
            infos = postprocessed_batch.get("infos")
        if infos is None:
            raise ValueError("Curriculum batch metadata requires infos in postprocessed batches.")

        stage_ids = []
        stage_indices = []
        n_interceptors = []
        n_targets = []
        for info in infos:
            if not isinstance(info, Mapping):
                raise ValueError("Curriculum batch info entries must be mappings.")
            missing = {
                CURRICULUM_STAGE_ID,
                CURRICULUM_STAGE_INDEX,
                CURRICULUM_N_INTERCEPTORS,
                CURRICULUM_N_TARGETS,
            } - set(info)
            if missing:
                raise ValueError(f"Curriculum batch info missing field(s): {sorted(missing)}")
            stage_ids.append(info[CURRICULUM_STAGE_ID])
            stage_indices.append(info[CURRICULUM_STAGE_INDEX])
            n_interceptors.append(info[CURRICULUM_N_INTERCEPTORS])
            n_targets.append(info[CURRICULUM_N_TARGETS])

        postprocessed_batch[CURRICULUM_STAGE_ID] = np.asarray(stage_ids, dtype=object)
        postprocessed_batch[CURRICULUM_STAGE_INDEX] = np.asarray(stage_indices, dtype=np.int32)
        postprocessed_batch[CURRICULUM_N_INTERCEPTORS] = np.asarray(n_interceptors, dtype=np.int32)
        postprocessed_batch[CURRICULUM_N_TARGETS] = np.asarray(n_targets, dtype=np.int32)

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
            self._broadcast_task(algorithm, task)

        self._log_curriculum_metrics(task, result, metrics_logger)

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
        self.current_stage_index = 0
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

    def _broadcast_task(self, algorithm: Any, task: CurriculumTask) -> None:
        env_runner_group = getattr(algorithm, "env_runner_group", None)
        if env_runner_group is None or not hasattr(env_runner_group, "foreach_env"):
            raise RuntimeError("Installed RLlib does not expose algorithm.env_runner_group.foreach_env().")

        task_dict = task.to_dict()

        def assign_task(env):
            orbital_env = _unwrap_orbital_env(env)
            if orbital_env is None:
                raise RuntimeError("Could not find OrbitalEnv.set_task() on RLlib rollout environment.")
            orbital_env.set_task(task_dict)
            return orbital_env.get_task().stage_id

        env_runner_group.foreach_env(assign_task)

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


def _unwrap_orbital_env(env: Any):
    if hasattr(env, "set_task") and hasattr(env, "get_task"):
        return env
    if hasattr(env, "par_env"):
        par_env = env.par_env
        if hasattr(par_env, "set_task") and hasattr(par_env, "get_task"):
            return par_env
    if hasattr(env, "unwrapped"):
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "set_task") and hasattr(unwrapped, "get_task"):
            return unwrapped
    return None
