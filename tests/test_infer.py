import gymnasium as gym
import numpy as np
import torch

from ray.rllib.core.columns import Columns
from src.main.python.utils.rllib_setup import compute_deterministic_module_action


class _FakeDeterministicDist:
    def __init__(self, logits):
        self.logits = logits

    @classmethod
    def from_logits(cls, logits):
        return cls(logits)

    def to_deterministic(self):
        return self

    def sample(self):
        return self.logits


class _FakeModule(torch.nn.Module):
    def __init__(self, forward_outputs, *, stateful=False):
        super().__init__()
        self._anchor = torch.nn.Parameter(torch.zeros(1))
        self._forward_outputs = forward_outputs
        self._stateful = stateful
        self.action_space = gym.spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(3,),
            dtype=np.float32,
        )

    def forward_inference(self, batch):
        obs = batch[Columns.OBS]
        assert isinstance(obs, torch.Tensor)
        assert tuple(obs.shape) == (1, 14)
        if self._stateful:
            assert Columns.STATE_IN in batch
        return self._forward_outputs

    def get_inference_action_dist_cls(self):
        return _FakeDeterministicDist


def test_compute_module_action_unsquashes_deterministic_logits():
    module = _FakeModule(
        {Columns.ACTION_DIST_INPUTS: torch.tensor([[0.5, -0.5, 0.0]], dtype=torch.float32)}
    )

    action, next_state = compute_deterministic_module_action(
        module,
        np.zeros(14, dtype=np.float32),
        normalize_actions=True,
        clip_actions=False,
    )

    np.testing.assert_allclose(action, np.array([0.05, -0.05, 0.0], dtype=np.float32))
    assert next_state is None


def test_compute_module_action_preserves_state_outputs():
    module = _FakeModule(
        {
            Columns.ACTIONS: torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
            Columns.STATE_OUT: {"h": torch.tensor([[1.0, 2.0]], dtype=torch.float32)},
        },
        stateful=True,
    )

    action, next_state = compute_deterministic_module_action(
        module,
        np.zeros(14, dtype=np.float32),
        normalize_actions=False,
        clip_actions=False,
        module_state={"h": np.array([0.0, 0.0], dtype=np.float32)},
    )

    np.testing.assert_allclose(action, np.array([0.01, 0.02, 0.03], dtype=np.float32))
    assert isinstance(next_state, dict)
    np.testing.assert_allclose(next_state["h"], np.array([1.0, 2.0], dtype=np.float32))
