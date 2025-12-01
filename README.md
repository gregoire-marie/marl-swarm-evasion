# MARL Swarm Interceptor Evasion

![CI](https://github.com/gregoire-marie/marl-swarm-evasion/actions/workflows/ci.yml/badge.svg)

This is a project about swarm interceptor satellites evasion using multi-agent reinforcement learning.

## Overview
A mixed target and interceptor satellite swarms cooperative-competitive environment, enabling multi-agent policy optimization with deep reinforcement learning using the [MADDPG algorithm](https://arxiv.org/pdf/1706.02275). 

Target satellites learn to evade a swarm of interceptor satellites dynamically learning seek-and-destroy strategies.

## Quick Start
### Requirements
- **Hardware**: A good GPU.
- **Software**: Python 3.9 (see environment.yml)

### Install the env (Conda version)
1. Install basic in a new Conda env:
   ```bash
    conda env create -n ENV_NAME -f environment.yml
   
### Run the app
1. **Run the main script**:
   ```bash
   python app/main.py

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
- **MADDPG Policy Learning**: Teach the interceptors to reduce the distance with targets and the targets to evade collisions using a fuel efficient maneuvers. *Based on [`RLlib`](https://docs.ray.io/en/latest/rllib/index.html), with a [`PyTorch`](https://pytorch.org/) backend*.
- **Multiple Chase Scenarios**: Aggressor and targets orbits, swarm size, maneuvering capacity, mission objective are all parameterizable.

## Repository Structure
```
marl_interceptor_evasion/
│
├── app/                            # Top-level scripts
│   └── main.py                     # Entry point for running simulations
│
├── docs/                           # Documentation and diagrams
│
├── environment.yml                 # Conda env with Poliastro, RLlib, etc.
│
├── src/
│   └── main/
│       └── python/
│           ├── environment/        # PettingZoo-compatible orbital env
│           │   ├── orbital_env.py        # PettingZoo.parallel_env
│           │   ├── reward_engine.py      # Modular reward computation
│           │   └── wrappers.py           # Optional preprocessing (SuperSuit)
│           │
│           ├── agents/             # Agent abstractions
│           │   ├── satellite_agent.py    # Satellite-level logic
│           │   ├── orbit_state.py        # Poliastro wrapper with propagation + Δv
│           │   └── maneuver.py           # Maneuver object and tracking
│           │
│           ├── orbital_meca/       # Low-level orbital tools
│           │   ├── orbits.py             # delta-v computation
│           │   └── approaches.py         # closest approach
│           │
│           ├── scenarios/          # Scenario generator & config
│           │   ├── scenario_loader.py    # Load/save scenario configs
│           │   └── initial_conditions.py # Orbital element sampling
│           │
│           ├── models/             # RLlib-compatible models
│           │
│           ├── train/              # Training scripts/configs
│           │   ├── train_rllib.py        # RLlib trainer launcher
│           │   └── config.yaml           # RLlib training configuration
│           │
│           └── utils/
│               ├── constants.py         # Global μ, Earth radius, etc.
│               └── helpers.py           # Unit conversion, logs, etc.
│
└── README.md
```

## License
All Rights Reserved

Copyright © 2025 Your Name

This source code and all associated files are the property of the author.
Unauthorized copying, distribution, modification, or sale of this software,
via any medium, is strictly prohibited without prior written permission.

## Keywords
MARL, Multi-Agent Reinforcement Learning, MADDPG, ASAT, Anti-Satellite