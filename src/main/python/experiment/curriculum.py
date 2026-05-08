import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from src.main.python.utils.constants import DEFAULT_START_TIME


KNOWN_TEAMS = {"interceptors", "targets"}
KNOWN_POLICIES = {"interceptor_policy", "target_policy"}
SUPPORTED_PROPAGATORS = {"keplerian"}
SUPPORTED_INITIAL_CONDITION_DISTRIBUTIONS = {"pursuit_evasion"}
SUPPORTED_MANEUVER_FRAMES = {"ECI", "TNW"}


@dataclass(frozen=True)
class MetricCondition:
    metric: str
    operator: str
    threshold: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MetricCondition":
        missing = {"metric", "operator", "threshold"} - set(data)
        if missing:
            raise ValueError(f"Metric condition missing required field(s): {sorted(missing)}")
        operator = str(data["operator"])
        if operator not in {">", ">=", "<", "<=", "=="}:
            raise ValueError(f"Unsupported metric condition operator: {operator}")
        return cls(
            metric=str(data["metric"]),
            operator=operator,
            threshold=float(data["threshold"]),
        )

    def evaluate(self, value: float) -> bool:
        if self.operator == ">":
            return value > self.threshold
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == "<":
            return value < self.threshold
        if self.operator == "<=":
            return value <= self.threshold
        if self.operator == "==":
            return value == self.threshold
        raise ValueError(f"Unsupported metric condition operator: {self.operator}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class PlateauCondition:
    metric: str
    window: int
    min_delta: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PlateauCondition":
        missing = {"metric", "window", "min_delta"} - set(data)
        if missing:
            raise ValueError(f"Plateau condition missing required field(s): {sorted(missing)}")
        window = int(data["window"])
        if window <= 1:
            raise ValueError("Plateau condition window must be > 1.")
        min_delta = float(data["min_delta"])
        if min_delta < 0:
            raise ValueError("Plateau condition min_delta must be >= 0.")
        return cls(metric=str(data["metric"]), window=window, min_delta=min_delta)

    def evaluate(self, values: Sequence[float]) -> bool:
        if len(values) < self.window:
            return False
        window_values = list(values[-self.window :])
        return max(window_values) - min(window_values) <= self.min_delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "window": self.window,
            "min_delta": self.min_delta,
        }


@dataclass(frozen=True)
class AdvanceWhen:
    min_iterations: int
    consecutive_iterations: int
    conditions: Tuple[MetricCondition, ...]
    plateau: Optional[PlateauCondition] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AdvanceWhen":
        missing = {"min_iterations", "consecutive_iterations"} - set(data)
        if missing:
            raise ValueError(f"advance_when missing required field(s): {sorted(missing)}")
        min_iterations = int(data["min_iterations"])
        consecutive_iterations = int(data["consecutive_iterations"])
        if min_iterations < 0:
            raise ValueError("advance_when.min_iterations must be >= 0.")
        if consecutive_iterations <= 0:
            raise ValueError("advance_when.consecutive_iterations must be >= 1.")

        raw_conditions = data.get("conditions", [])
        if not isinstance(raw_conditions, (list, tuple)):
            raise ValueError("advance_when.conditions must be a list.")
        conditions = tuple(MetricCondition.from_mapping(condition) for condition in raw_conditions)
        plateau = PlateauCondition.from_mapping(data["plateau"]) if "plateau" in data else None
        if not conditions and plateau is None:
            raise ValueError("advance_when must define at least one metric condition or plateau condition.")
        return cls(
            min_iterations=min_iterations,
            consecutive_iterations=consecutive_iterations,
            conditions=conditions,
            plateau=plateau,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "min_iterations": self.min_iterations,
            "consecutive_iterations": self.consecutive_iterations,
            "conditions": [condition.to_dict() for condition in self.conditions],
        }
        if self.plateau is not None:
            data["plateau"] = self.plateau.to_dict()
        return data


@dataclass(frozen=True)
class CurriculumTask:
    stage_id: str
    n_interceptors: int
    n_targets: int
    disabled_actions: Tuple[str, ...]
    frozen_policies: Tuple[str, ...]
    trainable_policies: Tuple[str, ...]
    maneuver_frame: str
    propagator: str
    initial_condition_distribution: str
    max_delta_v_mps: float
    episode_length: int
    stage_index: int = 0
    advance_when: Optional[AdvanceWhen] = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        stage_index: int,
        n_max: int,
        m_max: int,
        require_advance_when: bool,
    ) -> "CurriculumTask":
        required = {
            "stage_id",
            "n_interceptors",
            "n_targets",
            "disabled_actions",
            "frozen_policies",
            "trainable_policies",
            "maneuver_frame",
            "propagator",
            "initial_condition_distribution",
            "max_delta_v_mps",
            "episode_length",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"Curriculum stage missing required field(s): {sorted(missing)}")

        advance_when = None
        if "advance_when" in data:
            advance_when = AdvanceWhen.from_mapping(data["advance_when"])
        elif require_advance_when:
            raise ValueError(f"Curriculum stage {data['stage_id']} requires advance_when.")

        task = cls(
            stage_id=str(data["stage_id"]),
            n_interceptors=int(data["n_interceptors"]),
            n_targets=int(data["n_targets"]),
            disabled_actions=_validated_names(data["disabled_actions"], KNOWN_TEAMS, "disabled_actions"),
            frozen_policies=_validated_names(data["frozen_policies"], KNOWN_POLICIES, "frozen_policies"),
            trainable_policies=_validated_names(data["trainable_policies"], KNOWN_POLICIES, "trainable_policies"),
            maneuver_frame=parse_curriculum_maneuver_frame(str(data["maneuver_frame"])),
            propagator=str(data["propagator"]),
            initial_condition_distribution=str(data["initial_condition_distribution"]),
            max_delta_v_mps=float(data["max_delta_v_mps"]),
            episode_length=int(data["episode_length"]),
            stage_index=stage_index,
            advance_when=advance_when,
        )
        task.validate(n_max=n_max, m_max=m_max)
        return task

    def validate(self, *, n_max: int, m_max: int) -> None:
        if not self.stage_id:
            raise ValueError("Curriculum stage_id must not be empty.")
        if self.n_interceptors < 0 or self.n_interceptors > n_max:
            raise ValueError(f"{self.stage_id}: n_interceptors must be in [0, {n_max}].")
        if self.n_targets < 0 or self.n_targets > m_max:
            raise ValueError(f"{self.stage_id}: n_targets must be in [0, {m_max}].")
        if self.n_interceptors + self.n_targets <= 0:
            raise ValueError(f"{self.stage_id}: at least one physical slot must be active.")
        if self.max_delta_v_mps <= 0:
            raise ValueError(f"{self.stage_id}: max_delta_v_mps must be > 0.")
        if self.episode_length <= 0:
            raise ValueError(f"{self.stage_id}: episode_length must be >= 1.")
        if self.propagator not in SUPPORTED_PROPAGATORS:
            raise ValueError(
                f"{self.stage_id}: unsupported propagator '{self.propagator}'. "
                f"Supported: {sorted(SUPPORTED_PROPAGATORS)}."
            )
        if self.initial_condition_distribution not in SUPPORTED_INITIAL_CONDITION_DISTRIBUTIONS:
            raise ValueError(
                f"{self.stage_id}: unsupported initial_condition_distribution "
                f"'{self.initial_condition_distribution}'. Supported: "
                f"{sorted(SUPPORTED_INITIAL_CONDITION_DISTRIBUTIONS)}."
            )
        frozen = set(self.frozen_policies)
        trainable = set(self.trainable_policies)
        overlap = frozen & trainable
        if overlap:
            raise ValueError(f"{self.stage_id}: policies cannot be both frozen and trainable: {sorted(overlap)}")

        controllable_policies = self.controllable_policies()
        if not controllable_policies:
            raise ValueError(f"{self.stage_id}: at least one team must be controllable.")
        missing_policies = controllable_policies - (frozen | trainable)
        if missing_policies:
            raise ValueError(
                f"{self.stage_id}: every controllable policy must be frozen or trainable; "
                f"missing {sorted(missing_policies)}."
            )

        disabled = set(self.disabled_actions)
        if "interceptors" in disabled and "interceptor_policy" in frozen | trainable:
            raise ValueError(f"{self.stage_id}: disabled interceptors cannot require interceptor_policy actions.")
        if "targets" in disabled and "target_policy" in frozen | trainable:
            raise ValueError(f"{self.stage_id}: disabled targets cannot require target_policy actions.")

    def controllable_policies(self) -> set:
        policies = set()
        if self.n_interceptors > 0 and "interceptors" not in self.disabled_actions:
            policies.add("interceptor_policy")
        if self.n_targets > 0 and "targets" not in self.disabled_actions:
            policies.add("target_policy")
        return policies

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "stage_id": self.stage_id,
            "n_interceptors": self.n_interceptors,
            "n_targets": self.n_targets,
            "disabled_actions": list(self.disabled_actions),
            "frozen_policies": list(self.frozen_policies),
            "trainable_policies": list(self.trainable_policies),
            "maneuver_frame": self.maneuver_frame,
            "propagator": self.propagator,
            "initial_condition_distribution": self.initial_condition_distribution,
            "max_delta_v_mps": self.max_delta_v_mps,
            "episode_length": self.episode_length,
            "stage_index": self.stage_index,
        }
        if self.advance_when is not None:
            data["advance_when"] = self.advance_when.to_dict()
        return data


@dataclass(frozen=True)
class CurriculumConfig:
    N_max: int
    M_max: int
    stages: Tuple[CurriculumTask, ...]
    timestep: float = 60.0
    start_time: str = DEFAULT_START_TIME
    seed: int = 42

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CurriculumConfig":
        missing = {"N_max", "M_max", "stages"} - set(data)
        if missing:
            raise ValueError(f"Curriculum config missing required field(s): {sorted(missing)}")
        n_max = int(data["N_max"])
        m_max = int(data["M_max"])
        if n_max <= 0:
            raise ValueError("N_max must be >= 1.")
        if m_max <= 0:
            raise ValueError("M_max must be >= 1.")
        raw_stages = data["stages"]
        if not isinstance(raw_stages, (list, tuple)) or not raw_stages:
            raise ValueError("Curriculum config requires a non-empty stages list.")
        stages = tuple(
            CurriculumTask.from_mapping(
                stage,
                stage_index=index,
                n_max=n_max,
                m_max=m_max,
                require_advance_when=index < len(raw_stages) - 1,
            )
            for index, stage in enumerate(raw_stages)
        )
        stage_ids = [stage.stage_id for stage in stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("Curriculum stage_id values must be unique.")
        if stages[-1].advance_when is not None:
            raise ValueError("Final curriculum stage must not define advance_when.")
        return cls(
            N_max=n_max,
            M_max=m_max,
            stages=stages,
            timestep=float(data.get("timestep", 60.0)),
            start_time=str(data.get("start_time", DEFAULT_START_TIME)),
            seed=int(data.get("seed", 42)),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "CurriculumConfig":
        with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
            return cls.from_mapping(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "N_max": self.N_max,
            "M_max": self.M_max,
            "timestep": self.timestep,
            "start_time": self.start_time,
            "seed": self.seed,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def task_by_index(self, stage_index: int) -> CurriculumTask:
        try:
            return self.stages[int(stage_index)]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unknown curriculum stage index: {stage_index}") from exc


def _validated_names(values: Iterable[Any], known_values: set, field_name: str) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list.")
    normalized = tuple(str(value) for value in values)
    unknown = set(normalized) - known_values
    if unknown:
        raise ValueError(f"{field_name} contains unknown value(s): {sorted(unknown)}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return normalized


def parse_curriculum_maneuver_frame(value: str) -> str:
    frame = str(value).strip().upper()
    if frame not in SUPPORTED_MANEUVER_FRAMES:
        raise ValueError(
            f"Unsupported maneuver frame '{value}'. Supported frames: {sorted(SUPPORTED_MANEUVER_FRAMES)}."
        )
    return frame
