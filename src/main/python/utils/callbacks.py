from typing import Dict, Optional, Union, Any
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID, EpisodeType

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
