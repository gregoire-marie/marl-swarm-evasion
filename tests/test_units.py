import pytest
import numpy as np
from astropy import units as u
from src.main.python.utils.units import ensure_quantity, assert_unit, to_m, to_mps, to_deg, to_value

def test_ensure_quantity():
    # From float
    q = ensure_quantity(10.0, u.m)
    assert q.value == 10.0
    assert q.unit == u.m
    
    # From other unit
    q2 = ensure_quantity(100.0 * u.cm, u.m)
    assert q2.value == 1.0
    assert q2.unit == u.m

def test_assert_unit():
    assert_unit(10.0 * u.m, u.cm)
    with pytest.raises(u.UnitConversionError):
        assert_unit(10.0 * u.m, u.s)

def test_convenience_helpers():
    assert to_m(5).unit == u.m
    assert to_mps(5).unit == u.m / u.s
    assert to_deg(5).unit == u.deg

def test_to_value():
    q = 1000.0 * u.mm
    assert to_value(q, u.m) == 1.0
