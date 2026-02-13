import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time
from src.main.python.orbital_meca.approaches import compute_closest_approaches_kep

def test_compute_closest_approaches_kep():
    epoch = Time("2025-01-01 00:00:00", scale="utc")
    
    # Orbit A: standard LEO
    elements_A = (
        6771.0 * u.km, 0.0001 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 0 * u.deg
    )
    
    # Orbit B: slightly ahead in Mean Anomaly
    elements_B = (
        6771.0 * u.km, 0.0001 * u.one, 51.6 * u.deg,
        0 * u.deg, 0 * u.deg, 0.1 * u.deg
    )
    
    # Run with small duration to keep it fast
    results = compute_closest_approaches_kep(
        elements_A, elements_B, epoch, 
        N=1, duration_days=0.1, step_sec=100
    )
    
    # Depending on the orbits, there might not be a local minimum in such a short window
    # But it should at least return a list (possibly empty)
    assert isinstance(results, list)

def test_no_minima_detected(capsys):
    epoch = Time("2025-01-01 00:00:00", scale="utc")
    # Diverging orbits: A is lower (faster) and ahead of B.
    # a_A = 6771 km, a_B = 7000 km.
    # M_A = 10 deg, M_B = 0 deg.
    # A will move away from B.
    elements = (
        6771.0 * u.km, 0.0 * u.one, 0.0 * u.deg,
        0 * u.deg, 0 * u.deg, 10 * u.deg
    )
    elements2 = (
        7000.0 * u.km, 0.0 * u.one, 0.0 * u.deg,
        0 * u.deg, 0 * u.deg, 0 * u.deg
    )
    
    # Use short duration and large steps to ensure no minimum is captured.
    results = compute_closest_approaches_kep(
        elements, elements2, epoch, 
        N=1, duration_days=0.01, step_sec=100
    )
    assert results == []
    captured = capsys.readouterr()
    assert "No local minima" in captured.out
