import numpy as np
from astropy import units as u
from astropy.time import Time

from src.main.python.orbital_meca.orbits import (
    compute_instantaneous_delta_v,
    compute_eci_distance,
)
from src.main.python.agents.orbit_state import OrbitState


def elements_a_b():
    a = 6771.0 * u.km
    e = 0.0001 * u.one
    inc = 51.6 * u.deg
    # Slightly different RAAN and argp
    el_a = (a, e, inc, 0 * u.deg, 0 * u.deg)
    el_b = (a * 1.02, e, inc, 5.0 * u.deg, 2.0 * u.deg)
    return el_a, el_b


def test_compute_instantaneous_delta_v_shapes_and_units():
    el_a, el_b = elements_a_b()
    M_burn = 30.0 * u.deg
    epoch = Time("2025-01-01 00:00:00", scale="utc")

    dv_vec, dv_mag = compute_instantaneous_delta_v(el_a, el_b, M_burn, epoch)

    # Vector units and shape
    assert hasattr(dv_vec, "unit")
    assert dv_vec.unit == (u.m / u.s)
    assert dv_vec.shape == (3,)

    # Magnitude units
    assert hasattr(dv_mag, "unit")
    assert dv_mag.unit == (u.m / u.s)

    # Consistency between vector and magnitude
    dv_vec_mag = np.linalg.norm(dv_vec.to_value(u.m / u.s))
    assert np.isclose(dv_vec_mag, dv_mag.to_value(u.m / u.s), rtol=1e-6)


def test_compute_eci_distance_symmetry_and_nonnegativity():
    epoch = Time("2025-01-01 00:00:00", scale="utc")
    elements1 = (
        6771.0 * u.km, 0.0002 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 0 * u.deg
    )
    elements2 = (
        (6771.0 + 50) * u.km, 0.0002 * u.one, 51.6 * u.deg,
        10.0 * u.deg, 0 * u.deg, 120.0 * u.deg
    )

    s1 = OrbitState(elements1, epoch)
    s2 = OrbitState(elements2, epoch)

    d12 = compute_eci_distance(s1, s2)
    d21 = compute_eci_distance(s2, s1)

    assert d12 >= 0.0
    assert np.isfinite(d12)
    assert np.isclose(d12, d21, rtol=1e-12)
