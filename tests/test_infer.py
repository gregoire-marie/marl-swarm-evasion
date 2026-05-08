import gymnasium as gym
import numpy as np
import torch

from argparse import Namespace
from ray.rllib.core.columns import Columns
from app.infer import _unwrap_angle_deg, plot_inference
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


def test_unwrap_angle_deg_removes_wrap_discontinuity():
    unwrapped = _unwrap_angle_deg([350.0, 355.0, 1.0, 5.0])

    np.testing.assert_allclose(unwrapped, np.array([350.0, 355.0, 361.0, 365.0]))


def test_plot_inference_writes_orbital_elements_png(tmp_path):
    agent_ids = ["interceptor_0", "target_0"]
    times_min = [0.0, 1.0, 2.0]
    step_times_min = [1.0, 2.0]
    orbital_elements = {
        aid: {
            "a_m": [7_000_000.0, 7_000_010.0, 7_000_020.0],
            "e": [0.001, 0.0011, 0.0012],
            "i_deg": [51.0, 51.1, 51.2],
            "raan_deg": [350.0, 355.0, 1.0],
            "argp_deg": [10.0, 11.0, 12.0],
            "M_deg": [20.0, 21.0, 22.0],
        }
        for aid in agent_ids
    }
    inference_data = {
        "spec": Namespace(max_delta_v_mps=20.0),
        "agent_ids": agent_ids,
        "times_min": times_min,
        "step_times_min": step_times_min,
        "trajectories_m": {
            aid: [
                np.array([7_000_000.0, 0.0, 0.0]),
                np.array([7_000_000.0, 1_000.0, 0.0]),
                np.array([7_000_000.0, 2_000.0, 0.0]),
            ]
            for aid in agent_ids
        },
        "rewards": {aid: [1.0, 2.0] for aid in agent_ids},
        "fuel_mps": {aid: [100.0, 99.0, 98.0] for aid in agent_ids},
        "action_magnitudes_mps": {aid: [0.5, 0.7] for aid in agent_ids},
        "orbital_elements": orbital_elements,
        "distances_m": {"interceptor_0__target_0": [10_000.0, 9_000.0, 8_000.0]},
    }

    plot_inference(str(tmp_path), inference_data)

    assert (tmp_path / "orbital_elements.png").is_file()
