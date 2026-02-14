# MARL Swarm Interceptor Evasion

![CI](https://github.com/gregoire-marie/marl-swarm-evasion/actions/workflows/ci.yml/badge.svg)

This is a project about swarm interceptor satellites evasion using multi-agent reinforcement learning.

## Overview
A mixed target and interceptor satellite swarms cooperative-competitive environment, enabling multi-agent policy optimization with deep reinforcement learning using algorithms such as [MADDPG](https://arxiv.org/pdf/1706.02275) or [PPO](https://arxiv.org/abs/1707.06347). 

Target satellites learn to evade a swarm of interceptor satellites dynamically learning seek-and-destroy strategies.

## Quick Start
### Requirements
- **Hardware**: A good GPU.
- **Software**: Python 3.9.21 (see `pyproject.toml`)

### Install the env (uv version)
1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

### Run the app
1. **Visualize rewards**:
   ```bash
   python app/visualize_rewards.py
   ```
2. **Run training (PPO)**:
   ```bash
   python app/train.py
   ```

## Testing
Run tests using:
```bash
uv run pytest tests/ --cov=src/main/python --cov-report=term-missing
```

## Conventions

This project standardizes physical units, angles, and time across the codebase for correctness and reproducibility:

- Units and types
  - Internals use astropy quantities (Quantity) end-to-end.
  - Distances are in kilometers (km); velocities and Δv in kilometers per second (km/s).
  - Angles use mean anomaly M in the public API; internally conversions may use true anomaly ν.
  - Time is handled with astropy.time (Time, TimeDelta). Avoid naive datetime.
  - At RL edges (actions/observations), values are plain numpy float arrays. Convert with .to_value(...) at boundaries.

- Actions
  - 3D delta-v vectors in the Earth-Centered Inertial (ECI) frame (km/s).
  - Magnitudes are clipped by `env_config["max_delta_v_kms"]`.

- Observations
  - Flat float vectors containing Keplerian elements and derived scalars (e.g., remaining Δv, pairwise distances).
  - Normalization:
    - Semi-major axis: Centered around 7000 km, scaled by 1000 km.
    - Eccentricity: Already in [0, 1].
    - Angles (i, RAAN, argp, M): Normalized to [-1, 1] (wrapped to $[-\pi, \pi]$ then divided by $\pi$).
    - Remaining Δv: Normalized by the initial fuel budget.
    - Distances: Scaled by 1000 km.

- Distances
  - Pairwise ECI distances and similar quantities are expressed in kilometers (km).

- Seeding and determinism
  - Use utils.random.set_global_seed(seed) or pass seed to env.reset(seed=...) to seed Python, NumPy, and PyTorch (if installed).
  - Tests and examples use fixed epochs and deterministic elements for reproducibility.

## Learn to parametrize

Multiple modes and features are available and parameterizable. # TODO : detail the parameters

### Observation space

- [x] **Keplerian elements and distance**: Each satellite observes its own normalized Keplerian parameters and remaining fuel, as well as the normalized Keplerian parameters and relative distance of all other satellites.
- [ ] **Close approaches**: (Coming soon) Inclusion of time-to-closest-approach and distance-at-closest-approach in observations.

### Action space

- [x] **ECI Δv Maneuvers**: Agents choose 3D delta-v vectors in the ECI frame.
- [ ] **TNW Maneuvers**: (**Not** Planned) Maneuvers defined in the TNW local frame.
- [ ] **Keplerian Target Orbit**: (**Not** Planned) Agents choose a target orbit; the environment computes and applies the required Δv.

### Interceptors objective

- [x] **Seek-and-destroy**: Distance to targets is minimized using reciprocal distance reward shaping.
- [ ] **Approach-and-maintain**: (Planned) Minimize distance then maintain a stable offset.

### Targets objective

- [x] **Evade**: Maintain safety distance from interceptors at low fuel cost.
- [ ] **Station keeping**: (Planned) Stay within an orbital slot while evading interceptors.

## Main features
- **Orbital Environment Builder**: Creates an environment containing target and interceptor satellites. *Based on [`Poliastro`](https://docs.poliastro.space/en/stable/) and [`Astropy`](https://www.astropy.org/)*. Manages:
  - Orbital propagation,
  - Orbital maneuvers,
  - Proximity approach computation,
  - Collision probability estimation.
- **Policy Learning**: Teach interceptors to track targets and targets to evade collisions using fuel-efficient maneuvers. Supports algorithms such as **MADDPG** and **PPO** via RLlib.
- **Parametric Scenarios**: Easily configure orbital regions (LEO/MEO/GEO), swarm size, maneuvering capacity, and mission objectives.
- **Modular Reward Engine**: Pluggable reward shaping functions (reciprocal distance, logarithmic, quadratic) for different mission goals.
- **Visualization Tools**: Utilities to visualize reward landscapes and simulation results.

## Repository Structure
```
marl-swarm-evasion/
│
├── app/                            # Entry points and scripts
│   ├── train.py                    # RLlib training script (PPO)
│   └── visualize_rewards.py        # Reward shaping visualization tool
│
├── docs/                           # Documentation and diagrams
│
├── src/
│   └── main/
│       └── python/
│           ├── agents/             # Agent and orbital state abstractions
│           │   ├── orbit_state.py     # Poliastro wrapper for propagation & Δv
│           │   └── satellite_agent.py # Agent logic & observations
│           │
│           ├── environment/        # PettingZoo-compatible orbital environment
│           │   ├── orbital_env.py     # ParallelEnv implementation
│           │   ├── reward_engine.py   # Modular reward computation
│           │   └── scenarios.py       # Scenario generation templates
│           │
│           ├── orbital_meca/       # Orbital mechanics tools
│           │   ├── approaches.py      # Closest approach calculations
│           │   └── orbits.py          # ECI distance & orbital utilities
│           │
│           └── utils/              # Shared utilities
│               ├── constants.py       # Physical & environment constants
│               ├── helpers.py         # Logging & unit conversions
│               ├── normalization.py   # Observation scaling
│               ├── random.py          # Seeding & reproducibility
│               └── units.py           # Unit guardrails
│
├── tests/                          # Test suite
├── pyproject.toml                  # Dependencies and project metadata
└── README.md
```

## License
All Rights Reserved

Copyright © 2025 Grégoire MARIE

This source code and all associated files are the property of the author.
Unauthorized copying, distribution, modification, or sale of this software,
via any medium, is strictly prohibited without prior written permission.

## Keywords
MARL, Multi-Agent Reinforcement Learning, MADDPG, ASAT, Anti-Satellite