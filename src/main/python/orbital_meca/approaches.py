import numpy as np
from astropy import units as u
from astropy.time import TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.angles import E_to_nu, M_to_E
from scipy.signal import argrelextrema


def compute_closest_approaches_kep(elements_A, elements_B, epoch, N=3,
                                   duration_days=10, step_sec=10):
    """
    Compute the next N closest approaches between two satellites using Keplerian propagation only.

    Parameters
    ----------
    elements_A, elements_B : tuple
        Tuples of orbital elements (a, ecc, inc, raan, argp, mean_anomaly) with astropy units.
    epoch : astropy.time.Time
        Epoch at which both orbits are defined.
    N : int
        Number of closest approaches to return.
    duration_days : float
        Duration of search window.
    step_sec : float
        Step size for propagation (in seconds).

    Returns
    -------
    List of (time, distance [km]) tuples for the N closest approaches.
    """

    # Unpack elements
    aA, eA, iA, raanA, argpA, MA = elements_A
    aB, eB, iB, raanB, argpB, MB = elements_B

    # Convert mean anomaly to eccentric anomaly (in radians)
    E_A = M_to_E(MA.to(u.rad).value, eA.value)
    E_B = M_to_E(MB.to(u.rad).value, eB.value)

    # Convert eccentric anomaly to true anomaly (in radians)
    nuA = E_to_nu(E_A, eA.value) * u.rad
    nuB = E_to_nu(E_B, eB.value) * u.rad

    # Create the orbits using classical Keplerian elements
    orbA = Orbit.from_classical(Earth, aA, eA, iA, raanA, argpA, nuA, epoch)
    orbB = Orbit.from_classical(Earth, aB, eB, iB, raanB, argpB, nuB, epoch)

    # Generate propagation times
    total_steps = int((duration_days * 86400) // step_sec)
    times = [epoch + TimeDelta(i * step_sec, format="sec") for i in range(total_steps)]

    # Compute inter-satellite distances at each time
    distances = []
    for t in times:
        rA, _ = orbA.propagate(t - epoch).rv()
        rB, _ = orbB.propagate(t - epoch).rv()
        dist = (rA - rB).norm().to(u.km).value
        distances.append(dist)

    distances = np.array(distances)

    # Detect local minima in the distance array
    minima_indices = argrelextrema(distances, np.less, order=5)[0]

    if len(minima_indices) == 0:
        print("No local minima (close approaches) detected.")
        return []

    # Extract times and distances at local minima
    closest_events = [(times[i], distances[i]) for i in minima_indices]

    # Sort by distance and return the closest N events
    closest_sorted = sorted(closest_events, key=lambda x: x[1])
    return closest_sorted[:N]
