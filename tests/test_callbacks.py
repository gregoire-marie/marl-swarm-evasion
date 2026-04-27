from src.main.python.utils.callbacks import OrbitalPhysicsCallbacks


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
