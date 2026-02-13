# MARL Swarm Interceptor Evasion

![CI](https://github.com/gregoire-marie/marl-swarm-evasion/actions/workflows/ci.yml/badge.svg)

This is a project about swarm interceptor satellites evasion using multi-agent reinforcement learning.

## Overview
A mixed target and interceptor satellite swarms cooperative-competitive environment, enabling multi-agent policy optimization with deep reinforcement learning using algorithms such as [MADDPG](https://arxiv.org/pdf/1706.02275) or [PPO](https://arxiv.org/abs/1707.06347). 

Target satellites learn to evade a swarm of interceptor satellites dynamically learning seek-and-destroy strategies.

## Quick Start
### Requirements
- **Hardware**: A good GPU.
- **Software**: Python 3.9 (see pyproject.toml)

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
  - 3D delta-v vectors (ECI) in km/s. Magnitudes are clipped by env_config["max_delta_v_kms"].

- Observations
  - Flat float vectors containing Keplerian elements and derived scalars (e.g., remaining Δv, pairwise distances in km).
  - Values are normalized to [-1, 1] or [0, 1] for training stability (e.g., semi-major axis is centered around 7000km, angles are scaled by $\pi$).

- Distances
  - Pairwise ECI distances and similar quantities are expressed in kilometers (km).

- Seeding and determinism
  - Use utils.random.set_global_seed(seed) or pass seed to env.reset(seed=...) to seed Python, NumPy, and PyTorch (if installed).
  - Tests and examples use fixed epochs and deterministic elements for reproducibility.

## Learn to parametrize

Multiple modes are available for the various features.

### Observation space

- [x] **Parameters and distance**: Each satellite observes the keplerian parameters of all satellites (targets and interceptors), as well as their current distance, and its own remaining delta-V. Best for a prograde chases, specifically approach-and-maintain.
- [ ] **Parameters and close approaches**: Each satellite observes the keplerian parameters of all satellites (targets and interceptors), as well as information relative to the next closest approach to each other satellites (approach distance and velocity, and time remaining to the closest approach), and its own remaining delta-V. Best for retrograde seed-and-destroy.

### Action space

- [x] **TNW Maneuvers**: Agents learn to directly choose the maneuver parameters in the TNW local orbital frame.
- [ ] **Keplerian Target Orbit**: Agents learn to choose on which orbit they shall be next. The optimal maneuver delta-V is then determined and applied to the satellite.

### Interceptors objective

- [x] **Seek-and-destroy**: The distance to targets must be reduced to zero as fast as possible, regardless of relative speed.
- [ ] **Approach-and-maintain (coming)**: The distance to targets must be reduced in a cost-efficient manner, then a set distance must be kept.

### Targets objective

- [x] **Evade**: Kept a non-null distance with all interceptors at low fuel cost, without any bounds of movement.
- [ ] **Station keeping**: Stay inside a given orbit box, while evading interceptors.

## Main features
- **Orbital Environment Builder**: Creates an environment containing target and interceptor satellites. *Based on [`Poliastro`](https://docs.poliastro.space/en/stable/) and [`Astropy`](https://www.astropy.org/)*. Manages:
  - Orbital propagation,
  - Orbital maneuvers,
  - Proximity approach computation,
  - Collision probability estimation.
- **Policy Learning**: Teach the interceptors to reduce the distance with targets and the targets to evade collisions using fuel efficient maneuvers. Supports algorithms such as **MADDPG** and **PPO** via RLlib.
- **Multiple Chase Scenarios**: Aggressor and targets orbits, swarm size, maneuvering capacity, mission objective are all parameterizable.

## Repository Structure
```
marl_interceptor_evasion/
│
├── app/                            # Top-level scripts
│   ├── train.py                    # RLlib training script (PPO)
│   └── visualize_rewards.py        # Reward shaping visualization tool
│
├── docs/                           # Documentation and diagrams
│
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Locked dependencies
│
├── src/
│   └── main/
│       └── python/
│           ├── agents/             # Agent abstractions
│           │   ├── orbit_state.py     # Poliastro wrapper for propagation & Δv
│           │   └── satellite_agent.py # Satellite-level logic & observations
│           │
│           ├── environment/        # PettingZoo-compatible orbital env
│           │   ├── orbital_env.py     # PettingZoo ParallelEnv implementation
│           │   ├── reward_engine.py   # Modular reward computation
│           │   └── scenarios.py       # Scenario generation templates
│           │
│           ├── orbital_meca/       # Low-level orbital mechanics tools
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
├── tests/                          # 100% coverage test suite
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