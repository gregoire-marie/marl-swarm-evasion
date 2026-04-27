from __future__ import annotations

"""
Unit guardrails and helpers.

This module centralizes thin wrappers around Astropy to enforce canonical units
across the codebase, as per the project guidelines:
 - Distances in m
 - Orbital velocities in m/s
 - Delta-v in m/s
 - Angles in degrees externally; radians for internal unwrapped arrays
 - Time handled by astropy.time (Time/TimeDelta)

Use these helpers in new physics code paths to avoid unit mixups and ensure
consistent conversions at API boundaries.
"""

from typing import Any
from astropy import units as u


def ensure_quantity(value: Any, unit: u.UnitBase) -> u.Quantity:
    """
    Ensure a value is an astropy Quantity with the given unit. If `value` is a plain
    number, interpret it in the provided unit. If it's a Quantity, convert to the unit.

    Parameters
    ----------
    value : Any
        Either a float/int or astropy Quantity.
    unit : astropy Unit
        Target unit (e.g., u.m, u.m / u.s).

    Returns
    -------
    Quantity
        Value expressed in the specified unit.
    """
    if hasattr(value, "to"):
        return value.to(unit)
    return float(value) * unit


def assert_unit(q: u.Quantity, unit: u.UnitBase) -> None:
    """
    Assert that Quantity `q` is convertible to `unit`. Raises UnitConversionError otherwise.
    """
    # Will raise if incompatible
    _ = q.to(unit)


def to_m(x: Any) -> u.Quantity:
    """Return value as Quantity in meters."""
    return ensure_quantity(x, u.m)


def to_mps(x: Any) -> u.Quantity:
    """Return value as Quantity in meters per second (m/s)."""
    return ensure_quantity(x, u.m / u.s)


def to_deg(x: Any) -> u.Quantity:
    """Return value as Quantity in degrees (deg)."""
    return ensure_quantity(x, u.deg)


def to_value(q: u.Quantity, unit: u.UnitBase) -> float:
    """
    Convert Quantity to float value in the specified unit. Use at RL edges to
    strip units intentionally.
    """
    return float(q.to_value(unit))
