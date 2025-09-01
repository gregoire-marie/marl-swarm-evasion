from __future__ import annotations

from typing import Tuple

from astropy import units as u
from astropy.units import Quantity
from astropy.time import Time

from poliastro.bodies import Earth
from poliastro.twobody.orbit import Orbit
from poliastro.twobody.angles import (
    M_to_E,
    E_to_nu,
    nu_to_E,
    E_to_M,
)

from src.main.python.utils.helpers import get_logger

log = get_logger("OrbitState")


class OrbitState:
    """
    A wrapper around poliastro's Orbit object that enables convenience operations
    like propagation and delta-v application in the ECI frame, while maintaining
    a consistent Keplerian interface using mean anomaly.

    Notes:
        - Internally converts mean anomaly M → eccentric anomaly E → true anomaly ν.
        - Conversion steps use float-based units for compatibility across Astropy/Poliastro.
        - Outputs Keplerian elements with **mean anomaly (M)**, not true anomaly (ν).
    """

    def __init__(self, elements: Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity], epoch: Time):
        """
        Initialize the OrbitState with Keplerian elements and epoch.

        Args:
            elements (tuple): Orbital elements in order:
                - a (Quantity[km]): Semi-major axis
                - e (Quantity[unitless]): Eccentricity
                - i (Quantity[deg]): Inclination
                - RAAN (Quantity[deg]): Right ascension of ascending node
                - argp (Quantity[deg]): Argument of perigee
                - M (Quantity[deg]): Mean anomaly
            epoch (Time): Epoch at which the orbit is defined.
        """
        self.epoch: Time = epoch
        self.orbit: Orbit = self._build_orbit_from_elements(elements, epoch)

    def _build_orbit_from_elements(self, elements, epoch: Time) -> Orbit:
        """
        Construct a Poliastro Orbit from classical elements using mean anomaly.

        Args:
            elements (tuple): See `__init__` for element ordering and units.
            epoch (Time): Epoch of the orbit.

        Returns:
            Orbit: Poliastro orbit object.
        """
        a, ecc, inc, raan, argp, M = elements

        # Ensure proper units for angle conversions (M: Quantity[rad], e: Quantity[one])
        e_q = ecc.to(u.one) if hasattr(ecc, "to") else ecc * u.one
        M_q = M.to(u.rad) if hasattr(M, "to") else M * u.rad

        E = M_to_E(M_q, e_q)
        nu = E_to_nu(E, e_q)

        return Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)

    def propagate_to(self, new_epoch: Time) -> None:
        """
        Propagate the orbit to a new epoch using Keplerian motion.

        Args:
            new_epoch (Time): Future time to which the orbit should be propagated.

        Returns:
            None
        """
        if new_epoch <= self.epoch:
            return

        # Use Quantity time-of-flight for compatibility across poliastro versions
        delta_seconds = (new_epoch - self.epoch).to_value(u.s)
        tof = delta_seconds * u.s
        self.orbit = self.orbit.propagate(tof)
        self.epoch = new_epoch

    def apply_delta_v(self, dv_vector: Quantity, time: Time) -> None:
        """
        Apply an instantaneous delta-v at a given epoch.

        Args:
            dv_vector (Quantity[km/s]): 3D delta-v vector in ECI frame.
            time (Time): Time at which the delta-v is applied.

        Returns:
            None
        """
        if time != self.epoch:
            self.propagate_to(time)

        r, v = self.orbit.rv()
        v_new = v + dv_vector

        self.orbit = Orbit.from_vectors(Earth, r, v_new, epoch=time)
        self.epoch = time

    def get_rv(self):
        """
        Return the current position and velocity vectors in ECI frame.

        Returns:
            tuple:
                - r (Quantity[km]): Position vector.
                - v (Quantity[km/s]): Velocity vector.
        """
        return self.orbit.rv()

    def get_keplerian(self):
        """
        Get the classical orbital elements at the current epoch using mean anomaly.

        Returns:
            tuple:
                - a (Quantity[km]): Semi-major axis
                - e (Quantity[unitless]): Eccentricity
                - i (Quantity[deg]): Inclination
                - RAAN (Quantity[deg]): Right ascension of ascending node
                - argp (Quantity[deg]): Argument of perigee
                - M (Quantity[deg]): Mean anomaly
        """
        a, e, inc, raan, argp, nu = self.orbit.classical()
        # Convert true anomaly back to mean anomaly (why: keep API consistent with constructor)
        e_q = (e if hasattr(e, "unit") else e * u.one)
        E = nu_to_E(nu, e_q)
        M = E_to_M(E, e_q).to(u.deg)
        return a, e, inc, raan, argp, M

