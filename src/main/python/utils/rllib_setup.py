import argparse
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.tune.registry import register_env

from src.main.python.environment.orbital_env import OrbitalEnv
from src.main.python.environment.scenarios import pursuit_evasion_scenario
from src.main.python.utils.helpers import policy_mapping_fn

SUPPORTED_MANEUVER_FRAMES = ("ECI", "TNW")
DEFAULT_START_TIME = "2025-01-01 00:00:00"
DEFAULT_MAX_DELTA_V_KMS = 0.02
LEGACY_ENV_DEFAULT_MAX_DELTA_V_KMS = 0.1


@dataclass(frozen=True)
class OrbitalRunSpec:
    n_interceptors: int = 1
    n_targets: int = 1
    timestep: float = 60.0
    episode_length: int = 100
    start_time: str = DEFAULT_START_TIME
    max_delta_v_kms: float = DEFAULT_MAX_DELTA_V_KMS
    maneuver_frame: str = "ECI"
    freeze_targets: bool = False
    seed: int = 42


def parse_maneuver_frame(value: str) -> str:
    frame = str(value).strip().upper()
    if frame not in SUPPORTED_MANEUVER_FRAMES:
        raise argparse.ArgumentTypeError(
            f"Unsupported maneuver frame '{value}'. Supported frames: {list(SUPPORTED_MANEUVER_FRAMES)}."
        )
    return frame


def validate_run_spec(spec: OrbitalRunSpec) -> OrbitalRunSpec:
    normalized_spec = replace(spec, maneuver_frame=parse_maneuver_frame(spec.maneuver_frame))
    if normalized_spec.n_interceptors <= 0:
        raise ValueError("--n-interceptors must be >= 1.")
    if normalized_spec.n_targets <= 0:
        raise ValueError("--n-targets must be >= 1.")
    if normalized_spec.episode_length <= 0:
        raise ValueError("--episode-length must be >= 1.")
    if normalized_spec.max_delta_v_kms <= 0:
        raise ValueError("--max-delta-v-kms must be > 0.")
    if normalized_spec.timestep <= 0:
        raise ValueError("--timestep must be > 0.")
    return normalized_spec


def run_spec_from_args(args: Any, *, base_spec: Optional[OrbitalRunSpec] = None) -> OrbitalRunSpec:
    overrides = {}
    for spec_field in fields(OrbitalRunSpec):
        if not hasattr(args, spec_field.name):
            continue
        value = getattr(args, spec_field.name)
        if value is not None:
            overrides[spec_field.name] = value
    return validate_run_spec(replace(base_spec or OrbitalRunSpec(), **overrides))


def run_spec_from_rllib_env_config(env_config: Optional[Mapping[str, Any]]) -> OrbitalRunSpec:
    config = dict(env_config or {})
    orbital_env_config = dict(config.get("orbital_env_config", {}))
    return validate_run_spec(
        OrbitalRunSpec(
            n_interceptors=int(config.get("n_interceptors", 1)),
            n_targets=int(config.get("n_targets", 1)),
            timestep=float(orbital_env_config.get("timestep_sec", 60.0)),
            episode_length=int(orbital_env_config.get("episode_length", 100)),
            start_time=str(orbital_env_config.get("start_time", DEFAULT_START_TIME)),
            # Older checkpoints may not store this field and relied on OrbitalEnv's
            # internal default instead.
            max_delta_v_kms=float(orbital_env_config.get("max_delta_v_kms", LEGACY_ENV_DEFAULT_MAX_DELTA_V_KMS)),
            maneuver_frame=str(orbital_env_config.get("maneuver_frame", "ECI")),
            freeze_targets=bool(orbital_env_config.get("freeze_targets", False)),
            seed=int(config.get("seed", 42)),
        )
    )


def build_agent_configs(spec: OrbitalRunSpec) -> Dict[str, Dict[str, Any]]:
    return pursuit_evasion_scenario(
        n_interceptors=spec.n_interceptors,
        n_targets=spec.n_targets,
        seed=spec.seed,
    )


def build_orbital_env_config(spec: OrbitalRunSpec) -> Dict[str, Any]:
    return {
        "timestep_sec": spec.timestep,
        "episode_length": spec.episode_length,
        "start_time": spec.start_time,
        "max_delta_v_kms": spec.max_delta_v_kms,
        "maneuver_frame": spec.maneuver_frame,
        "freeze_targets": spec.freeze_targets,
    }


def build_rllib_env_config(spec: OrbitalRunSpec) -> Dict[str, Any]:
    return {
        "n_interceptors": spec.n_interceptors,
        "n_targets": spec.n_targets,
        "seed": spec.seed,
        "orbital_env_config": build_orbital_env_config(spec),
    }


def create_raw_env(spec: OrbitalRunSpec) -> OrbitalEnv:
    return OrbitalEnv(
        agent_configs=build_agent_configs(spec),
        env_config=build_orbital_env_config(spec),
    )


def get_orbital_env_name() -> str:
    return OrbitalEnv.metadata.get("name", "orbital_env_v0")


def create_rllib_env(config: Dict[str, Any]) -> ParallelPettingZooEnv:
    spec = run_spec_from_rllib_env_config(config)
    return ParallelPettingZooEnv(create_raw_env(spec))


def register_orbital_env() -> str:
    env_name = get_orbital_env_name()
    register_env(env_name, create_rllib_env)
    return env_name


def build_policy_setup(spec: OrbitalRunSpec) -> Dict[str, Any]:
    agent_configs = build_agent_configs(spec)
    probe_env = OrbitalEnv(
        agent_configs=agent_configs,
        env_config=build_orbital_env_config(spec),
    )
    interceptor_id = next((aid for aid, cfg in agent_configs.items() if cfg["role"] == "interceptor"), None)
    target_id = next((aid for aid, cfg in agent_configs.items() if cfg["role"] == "target"), None)

    policies: Dict[str, PolicySpec] = {}
    interceptor_obs_space = None
    target_obs_space = None

    if interceptor_id is not None:
        interceptor_obs_space = probe_env.observation_space(interceptor_id)
        interceptor_act_space = probe_env.action_space(interceptor_id)
        policies["interceptor_policy"] = PolicySpec(
            observation_space=interceptor_obs_space,
            action_space=interceptor_act_space,
        )

    if target_id is not None:
        target_obs_space = probe_env.observation_space(target_id)
        target_act_space = probe_env.action_space(target_id)
        policies["target_policy"] = PolicySpec(
            observation_space=target_obs_space,
            action_space=target_act_space,
        )

    if not policies:
        raise ValueError("No policies were created. Check scenario agent configuration.")

    return {
        "policies": policies,
        "interceptor_obs_space": interceptor_obs_space,
        "target_obs_space": target_obs_space,
    }


def build_policies_to_train(policy_ids, *, freeze_targets: bool) -> list:
    policies_to_train = [policy_id for policy_id in policy_ids if policy_id != "target_policy" or not freeze_targets]
    return policies_to_train or list(policy_ids)


def rllib_policy_mapping_fn(agent_id: str, *unused_args: Any, **unused_kwargs: Any) -> str:
    return policy_mapping_fn(agent_id)


def _batch_single_item(item: Any) -> Any:
    if isinstance(item, np.ndarray):
        return np.expand_dims(item, axis=0)
    if isinstance(item, dict):
        return {key: _batch_single_item(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_batch_single_item(value) for value in item]
    if isinstance(item, tuple):
        return tuple(_batch_single_item(value) for value in item)
    return np.expand_dims(np.asarray(item), axis=0)


def _unbatch_single_item(item: Any) -> Any:
    if isinstance(item, dict):
        return {key: _unbatch_single_item(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_unbatch_single_item(value) for value in item]
    if isinstance(item, tuple):
        return tuple(_unbatch_single_item(value) for value in item)
    return item[0]


def _get_module_device(module: Any) -> Any:
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def compute_deterministic_module_action(
    module: Any,
    observation: Any,
    *,
    normalize_actions: bool,
    clip_actions: bool,
    module_state: Any = None,
) -> tuple:
    input_batch = {Columns.OBS: _batch_single_item(observation)}
    if module_state is not None:
        input_batch[Columns.STATE_IN] = _batch_single_item(module_state)

    forward_outputs = module.forward_inference(
        convert_to_torch_tensor(input_batch, device=_get_module_device(module))
    )

    if Columns.ACTIONS in forward_outputs:
        actions = forward_outputs[Columns.ACTIONS]
    elif Columns.ACTION_DIST_INPUTS in forward_outputs:
        action_dist = module.get_inference_action_dist_cls().from_logits(
            forward_outputs[Columns.ACTION_DIST_INPUTS]
        )
        actions = action_dist.to_deterministic().sample()
    else:
        raise KeyError(
            "RLModule.forward_inference() must return either "
            f"'{Columns.ACTIONS}' or '{Columns.ACTION_DIST_INPUTS}'."
        )

    action = _unbatch_single_item(convert_to_numpy(actions))
    if normalize_actions:
        action = space_utils.unsquash_action(action, module.action_space)
    elif clip_actions:
        action = space_utils.clip_action(action, module.action_space)

    next_state = None
    if Columns.STATE_OUT in forward_outputs:
        next_state = _unbatch_single_item(convert_to_numpy(forward_outputs[Columns.STATE_OUT]))

    return action, next_state
