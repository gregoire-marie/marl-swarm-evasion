import numpy as np

from src.main.python.environment.reward_engine import (
    objective_d_shaping_generator_v2,
    zero_d_shaping_generator_v2,
)


def test_objective_d_shaping_generator_v2_behavior():
    d_safe = 10.0
    alpha = 2.0
    fn = objective_d_shaping_generator_v2(d_safe=d_safe, alpha=alpha)

    # Outside safe distance → zero
    assert fn(20.0) == 0.0
    assert fn(d_safe) == 0.0

    # Inside safe distance → negative quadratic
    val_mid = fn(5.0)
    expected_mid = -alpha * (1.0 - 0.5) ** 2
    assert np.isclose(val_mid, expected_mid, rtol=1e-12)

    val_zero = fn(0.0)
    expected_zero = -alpha * (1.0 - 0.0) ** 2
    assert np.isclose(val_zero, expected_zero, rtol=1e-12)
    assert val_zero < val_mid < 0.0


def test_zero_d_shaping_generator_v2_behavior():
    beta = 10.0
    eps = 1e-3
    d_max = 1.0
    fn = zero_d_shaping_generator_v2(beta=beta, eps=eps, d_max=d_max)

    # Reciprocal decay until cutoff
    near = fn(0.0)
    mid = fn(0.5)
    far = fn(0.999)
    cutoff = fn(1.0)

    assert near > mid > far > 0.0
    assert np.isclose(near, beta / (0.0 + eps))
    # At and beyond d_max → zero
    assert cutoff == 0.0
    assert fn(5.0) == 0.0
