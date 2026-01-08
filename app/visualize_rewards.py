"""
Visualize reward shaping functions used in the project.

This script plots:
  - zero dist. interception shaping (distance -> reward for interceptors)
  - objective d evasion shaping (distance -> reward for targets)
  - objective d same-role dispersion shaping (distance -> reward for spacing)
  - Linear fuel penalty (Δv used -> penalty)

It annotates key objective thresholds (collision distance, avoid distance,
same-role spacing) and uses the default weights from src.main.python.utils.constants.

Usage examples:
  - python app/visualize_rewards.py
  - python app/visualize_rewards.py --xmin 1e-3 --xmax 2e3 --logx

Notes:
  - Distance unit is kilometers on the X axis for distance plots.
  - Δv unit is km/s on the X axis for the fuel plot.
  - The objective dist./zero dist. shaping functions saturate or have asymptotic behavior; the
    plots include helpful reference lines to appreciate these shapes.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.main.python.environment.reward_engine import (
    objective_d_shaping_generator,
    zero_d_shaping_generator,
    objective_d_shaping_generator_v2,
    zero_d_shaping_generator_v2,
    linear_reward_generator,
)
from src.main.python.utils.constants import (
    DEFAULT_OBJECTIVES,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_REWARD_WEIGHTS_V2,
)


def build_distance_grid(xmin: float, xmax: float, n: int, logx: bool) -> np.ndarray:
    if logx:
        return np.logspace(np.log10(xmin), np.log10(xmax), n)
    return np.linspace(xmin, xmax, n)


def annotate_thresholds(ax: plt.Axes, thresholds: dict):
    # Vertical lines for key distances
    ax.axvline(thresholds["collision_distance_km"], color="#d62728", ls="--", lw=1.5, label="collision")
    ax.axvline(thresholds["avoid_distance_km"], color="#2ca02c", ls=":", lw=1.5, label="avoid (target)")
    ax.axvline(thresholds["same_role_spacing_km"], color="#1f77b4", ls="-.", lw=1.5, label="spacing (same-role)")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize reward shaping functions.")
    parser.add_argument("--xmin", type=float, default=1e-2, help="Min distance (km) for distance plots.")
    parser.add_argument("--xmax", type=float, default=1e2, help="Max distance (km) for distance plots.")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples for curves.")
    parser.add_argument("--logx", action="store_true", help="Use logarithmic X scale for distance plots.")

    # Fuel grid
    parser.add_argument("--dv_max", type=float, default=5.0, help="Max Δv used (km/s) for fuel plot.")

    args = parser.parse_args()
    return args

def main(version):
    thresholds = {
        "collision_distance_km": DEFAULT_OBJECTIVES["collision_distance_km"],
        "avoid_distance_km": DEFAULT_OBJECTIVES["avoid_distance_km"],
        "same_role_spacing_km": DEFAULT_OBJECTIVES["same_role_spacing_km"],
    }

    if version == "v1":
        # Build shaping functions
        intercept_zero_d = zero_d_shaping_generator(w=DEFAULT_REWARD_WEIGHTS["intercept_shaping"])
        evasion_objective_d = objective_d_shaping_generator(objective=thresholds["avoid_distance_km"],
                                                                   w=DEFAULT_REWARD_WEIGHTS["evasion_shaping"])
        spacing_objective_d_i = objective_d_shaping_generator(objective=thresholds["same_role_spacing_km"],
                                                                     w=DEFAULT_REWARD_WEIGHTS["interceptor_dispersion"])
        spacing_objective_d_t = objective_d_shaping_generator(objective=thresholds["same_role_spacing_km"],
                                                                     w=DEFAULT_REWARD_WEIGHTS["target_dispersion"])
    elif version == "v2":
        # Build shaping functions
        intercept_zero_d = zero_d_shaping_generator_v2(
            beta=DEFAULT_REWARD_WEIGHTS["intercept_shaping"] * DEFAULT_REWARD_WEIGHTS_V2["zero_d_beta"],
            eps=DEFAULT_REWARD_WEIGHTS_V2["zero_d_eps"],
            d_max=DEFAULT_REWARD_WEIGHTS_V2["zero_d_max"],
        )
        evasion_objective_d = objective_d_shaping_generator_v2(
            d_safe=thresholds["avoid_distance_km"],
            alpha=DEFAULT_REWARD_WEIGHTS["evasion_shaping"] * DEFAULT_REWARD_WEIGHTS_V2["objective_d_alpha"],
        )
        spacing_objective_d_i = objective_d_shaping_generator_v2(
            d_safe=thresholds["same_role_spacing_km"],
            alpha=DEFAULT_REWARD_WEIGHTS["interceptor_dispersion"] * DEFAULT_REWARD_WEIGHTS_V2["objective_d_alpha"],
        )
        spacing_objective_d_t = objective_d_shaping_generator_v2(
            d_safe=thresholds["same_role_spacing_km"],
            alpha=DEFAULT_REWARD_WEIGHTS["target_dispersion"] * DEFAULT_REWARD_WEIGHTS_V2["objective_d_alpha"],
        )

    fuel_linear = linear_reward_generator(w=DEFAULT_REWARD_WEIGHTS["fuel_penalty"])

    # Grids
    xdist = build_distance_grid(args.xmin, args.xmax, args.n, args.logx)
    # avoid log(<=0) in objective dist.; clamp tiny values
    xdist_safe = np.clip(xdist, 1e-9, None)
    dv_used = np.linspace(0.0, args.dv_max, args.n)

    # Curves
    y_intercept = np.array([intercept_zero_d(x) for x in xdist_safe])
    y_evasion = np.array([evasion_objective_d(x) for x in xdist_safe])
    y_spacing_i = np.array([spacing_objective_d_i(x) for x in xdist_safe])
    y_spacing_t = np.array([spacing_objective_d_t(x) for x in xdist_safe])
    y_fuel = np.array([fuel_linear(x) for x in dv_used])

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # 1) Intercept (zero dist.) vs distance
    ax = axs[0, 0]
    ax.plot(xdist, y_intercept, label=f"zero dist. intercept (w={DEFAULT_REWARD_WEIGHTS['intercept_shaping']:g})", color="#d62728")
    annotate_thresholds(ax, thresholds)
    ax.set_title("Interceptor vs Target distance — zero dist. shaping (minimize distance)")
    ax.set_xlabel("distance (km)")
    ax.set_ylabel("reward")
    if args.logx:
        ax.set_xscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    if version == "v1":
        ax.text(0.02, 0.02, "r = w / (x + 1e-6)", transform=ax.transAxes, fontsize=9, va="bottom")
    elif version == "v2":
        ax.text(0.02, 0.02, "r = beta / (x + eps)", transform=ax.transAxes, fontsize=9, va="bottom")

    # 2) Evasion (objective dist.) vs distance
    ax = axs[0, 1]
    ax.plot(xdist, y_evasion, label=f"objective dist. evasion (obj={thresholds['avoid_distance_km']:.3g} km, w={DEFAULT_REWARD_WEIGHTS['evasion_shaping']:g})", color="#2ca02c")
    annotate_thresholds(ax, thresholds)
    ax.set_title("Target evasion — objective dist. shaping (maximize distance)")
    ax.set_xlabel("distance (km)")
    ax.set_ylabel("reward")
    if args.logx:
        ax.set_xscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    if version == "v1":
        ax.text(0.02, 0.02, "r = tanh((obj/w)*(log(x)/log(obj+1e-6) - 1))", transform=ax.transAxes, fontsize=9, va="bottom")
    elif version == "v2":
        ax.text(0.02, 0.02, f"r = -alpha * (1 - x/d_safe)^2   if x < d_safe\nr = 0 otherwise", transform=ax.transAxes,
                fontsize=9, va="bottom")

    # 3) Same-role spacing (objective dist.) — plot both roles weights for comparison
    ax = axs[1, 0]
    ax.plot(xdist, y_spacing_i, label=f"interceptor spacing (obj={thresholds['same_role_spacing_km']:.3g} km, w={DEFAULT_REWARD_WEIGHTS['interceptor_dispersion']:g})", color="#1f77b4")
    ax.plot(xdist, y_spacing_t, label=f"target spacing (obj={thresholds['same_role_spacing_km']:.3g} km, w={DEFAULT_REWARD_WEIGHTS['target_dispersion']:g})", color="#9467bd")
    annotate_thresholds(ax, thresholds)
    ax.set_title("Same-role dispersion — objective dist. shaping (maximize distance)")
    ax.set_xlabel("distance (km)")
    ax.set_ylabel("reward")
    if args.logx:
        ax.set_xscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    ax.text(0.02, 0.02, "same objective dist. form as evasion; different weights", transform=ax.transAxes, fontsize=9, va="bottom")

    # 4) Fuel penalty (linear) vs Δv used
    ax = axs[1, 1]
    ax.plot(dv_used, y_fuel, label=f"fuel penalty (w={DEFAULT_REWARD_WEIGHTS['fuel_penalty']:g})", color="#ff7f0e")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_title("Fuel usage penalty — linear in Δv used")
    ax.set_xlabel("Δv used (km/s)")
    ax.set_ylabel("reward (penalty if w<0)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()
    ax.text(0.02, 0.02, "r = w * Δv_used", transform=ax.transAxes, fontsize=9, va="bottom")

    fig.suptitle(f"Reward shaping functions and objective thresholds ({version})", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Provide some overall notes in the figure
    notes = (
        "Notes:\n"
        "- zero dist. shaping increases sharply at small distances; keep w moderate.\n"
        "- objective dist. shaping saturates to [-1, 1] via tanh; obj sets the knee.\n"
        "- Fuel penalty is linear; negative w penalizes consumption."
    )
    fig.text(0.02, 0.005, notes, ha="left", va="bottom", fontsize=9)

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main("v1")
    main("v2")
