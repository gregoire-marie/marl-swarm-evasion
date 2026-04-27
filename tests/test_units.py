import pytest
import numpy as np
from astropy import units as u
from src.main.python.utils.units import ensure_quantity, assert_unit, to_km, to_mps, to_deg, to_value

def test_ensure_quantity():
    # From float
    q = ensure_quantity(10.0, u.km)
    assert q.value == 10.0
    assert q.unit == u.km
    
    # From other unit
    q2 = ensure_quantity(1000.0 * u.m, u.km)
    assert q2.value == 1.0
    assert q2.unit == u.km

def test_assert_unit():
    assert_unit(10.0 * u.km, u.m)
    with pytest.raises(u.UnitConversionError):
        assert_unit(10.0 * u.km, u.s)

def test_convenience_helpers():
    assert to_km(5).unit == u.km
    assert to_mps(5).unit == u.m / u.s
    assert to_deg(5).unit == u.deg

def test_to_value():
    q = 1.0 * u.km
    assert to_value(q, u.m) == 1000.0
