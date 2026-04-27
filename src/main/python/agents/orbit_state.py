from __future__ import annotations

from typing import Tuple

import numpy as np
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

SUPPORTED_MANEUVER_FRAMES = {"ECI", "TNW"}


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

    @staticmethod
    def _normalize_maneuver_frame(maneuver_frame: str) -> str:
        frame = str(maneuver_frame).strip().upper()
        if frame not in SUPPORTED_MANEUVER_FRAMES:
            raise ValueError(
                f"Unsupported maneuver frame '{maneuver_frame}'. "
                f"Supported frames: {sorted(SUPPORTED_MANEUVER_FRAMES)}."
            )
        return frame

    def _get_tnw_basis_matrix(self) -> np.ndarray:
        """
        Return the TNW basis expressed in ECI components.

        Columns are unit vectors [T, N, W] expressed in ECI.
        """
        r, v = self.orbit.rv()
        r_m = r.to_value(u.m)
        v_mps = v.to_value(u.m / u.s)

        v_norm = np.linalg.norm(v_mps)
        if v_norm == 0.0:
            raise ValueError("Cannot build TNW frame with zero velocity norm.")

        h_vec = np.cross(r_m, v_mps)
        h_norm = np.linalg.norm(h_vec)
        if h_norm == 0.0:
            raise ValueError("Cannot build TNW frame with zero angular-momentum norm.")

        t_hat = v_mps / v_norm
        w_hat = h_vec / h_norm
        n_hat = np.cross(w_hat, t_hat)
        n_norm = np.linalg.norm(n_hat)
        if n_norm == 0.0:
            raise ValueError("Cannot build TNW frame with degenerate basis.")
        n_hat = n_hat / n_norm

        # Re-orthogonalize W after N normalization to reduce numerical drift.
        w_hat = np.cross(t_hat, n_hat)
        w_hat = w_hat / np.linalg.norm(w_hat)

        return np.column_stack((t_hat, n_hat, w_hat))

    def tnw_to_eci(self, dv_vector_tnw: Quantity) -> Quantity:
        """
        Convert a TNW-frame vector to ECI components at current epoch.
        """
        dv_tnw_mps = np.asarray(dv_vector_tnw.to_value(u.m / u.s), dtype=float)
        if dv_tnw_mps.shape != (3,):
            raise ValueError(f"TNW vector must have shape (3,), got {dv_tnw_mps.shape}.")

        basis = self._get_tnw_basis_matrix()
        dv_eci_mps = basis @ dv_tnw_mps
        return dv_eci_mps * u.m / u.s

    def eci_to_tnw(self, dv_vector_eci: Quantity) -> Quantity:
        """
        Convert an ECI-frame vector to TNW components at current epoch.
        """
        dv_eci_mps = np.asarray(dv_vector_eci.to_value(u.m / u.s), dtype=float)
        if dv_eci_mps.shape != (3,):
            raise ValueError(f"ECI vector must have shape (3,), got {dv_eci_mps.shape}.")

        basis = self._get_tnw_basis_matrix()
        dv_tnw_mps = basis.T @ dv_eci_mps
        return dv_tnw_mps * u.m / u.s

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

    def apply_delta_v(self, dv_vector: Quantity, time: Time, maneuver_frame: str = "ECI") -> None:
        """
        Apply an instantaneous delta-v at a given epoch.

        Args:
            dv_vector (Quantity[m/s]): 3D delta-v vector in maneuver frame.
            time (Time): Time at which the delta-v is applied.
            maneuver_frame (str): Local frame for dv_vector, "ECI" or "TNW".

        Returns:
            None
        """
        if time != self.epoch:
            self.propagate_to(time)

        frame = self._normalize_maneuver_frame(maneuver_frame)
        if hasattr(dv_vector, "to"):
            dv_input = dv_vector.to(u.m / u.s)
        else:
            dv_input = np.asarray(dv_vector, dtype=float) * u.m / u.s
        dv_values = np.asarray(dv_input.to_value(u.m / u.s), dtype=float)
        if dv_values.shape != (3,):
            raise ValueError(
                f"Delta-v vector must have shape (3,) in {frame} frame, got {dv_values.shape}."
            )
        dv_input = dv_values * u.m / u.s

        if frame == "TNW":
            dv_eci = self.tnw_to_eci(dv_input)
        else:
            dv_eci = dv_input

        r, v = self.orbit.rv()
        v_new = v.to(u.m / u.s) + dv_eci

        self.orbit = Orbit.from_vectors(Earth, r, v_new, epoch=time)
        self.epoch = time

    def get_rv(self):
        """
        Return the current position and velocity vectors in ECI frame.

        Returns:
            tuple:
                - r (Quantity[km]): Position vector.
                - v (Quantity[m/s]): Velocity vector.
        """
        r, v = self.orbit.rv()
        return r.to(u.km), v.to(u.m / u.s)

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
