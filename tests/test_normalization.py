import numpy as np
import pytest
from src.main.python.utils.normalization import normalize_angle, denormalize_angle

def test_normalize_angle():
    # 0 -> 0
    assert np.isclose(normalize_angle(0.0), 0.0)
    # pi -> 1.0 (or -1.0 depending on wrap)
    assert np.isclose(abs(normalize_angle(np.pi)), 1.0)
    # -pi -> -1.0 (or 1.0)
    assert np.isclose(abs(normalize_angle(-np.pi)), 1.0)
    # 2pi -> 0
    assert np.isclose(normalize_angle(2 * np.pi), 0.0)
    # pi/2 -> 0.5
    assert np.isclose(normalize_angle(np.pi / 2), 0.5)

def test_denormalize_angle():
    assert np.isclose(denormalize_angle(1.0), np.pi)
    assert np.isclose(denormalize_angle(0.0), 0.0)
    assert np.isclose(denormalize_angle(-0.5), -np.pi / 2)
