from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody.orbit import Orbit
from poliastro.twobody.angles import E_to_nu, M_to_E
import numpy as np

def compute_instantaneous_delta_v(
    elements_a, elements_b, M_burn, epoch=Time("2025-01-01 00:00:00", scale="utc")
):
    """
    Compute the instantaneous delta-v vector and magnitude needed to transfer
    from orbit A to orbit B at a common mean anomaly.

    Parameters
    ----------
    elements_a : tuple
        Tuple of 5 classical orbital elements for orbit A: (a, ecc, inc, raan, argp)
        Units must be astropy quantities.
    elements_b : tuple
        Same as elements_a, but for target orbit B.
    M_burn : astropy.units.Quantity
        Mean anomaly (at time of burn), assumed to be the same for both orbits.
    epoch : astropy.time.Time, optional
        Epoch at which the maneuver is evaluated (default is 2025-01-01).

    Returns
    -------
    delta_v_vector : astropy.units.Quantity
        Velocity difference vector [km/s] from A to B at burn point.
    delta_v_magnitude : astropy.units.Quantity
        Scalar magnitude of delta-v [m/s].
    """

    # Unpack elements
    aA, eccA, incA, raanA, argpA = elements_a
    aB, eccB, incB, raanB, argpB = elements_b

    # Convert M_burn to true anomaly using poliastro 0.17.0-compatible functions
    E_burn = M_to_E(M_burn.to(u.rad).value, eccA.value)     # Scalar rad
    nu_burn = E_to_nu(E_burn, eccA.value) * u.rad           # Quantity rad

    # Construct both orbits at the same true anomaly (same spatial point)
    orbA = Orbit.from_classical(Earth, aA, eccA, incA, raanA, argpA, nu_burn, epoch)
    orbB = Orbit.from_classical(Earth, aB, eccB, incB, raanB, argpB, nu_burn, epoch)

    # Extract velocity vectors
    _, vA = orbA.rv()
    _, vB = orbB.rv()

    # Compute delta-v vector and magnitude
    delta_v_vector = vB - vA
    delta_v_magnitude = delta_v_vector.norm().to(u.m / u.s)

    return delta_v_vector.to(u.km / u.s), delta_v_magnitude


def compute_eci_distance(state_a, state_b):
    """
    Computes the Euclidean distance between two objects in ECI frame.

    Args:
        state_a, state_b: OrbitState objects

    Returns:
        float: Distance in kilometers
    """
    r1, _ = state_a.get_rv()
    r2, _ = state_b.get_rv()
    return np.linalg.norm((r1 - r2).to_value(u.km))
