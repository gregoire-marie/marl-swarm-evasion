# MARL Swarm Interceptor Evasion

This is a project about swarm interceptor satellites evasion using multi-agent reinforcement learning.

## Overview
A mixed target and interceptor satellite swarms cooperative-competitive environment, enabling multi-agent policy optimization with deep reinforcement learning using the [MADDPG algorithm](https://arxiv.org/pdf/1706.02275). 

Target satellites learn to evade a swarm of interceptor satellites dynamically learning seek-and-destroy strategies.

## Quick Start
### Requirements
- **Hardware**: A good GPU.
- **Software**: Python 3.9 (see environment.yaml)

### Install the env (Conda version)
1. Install basic in a new Conda env:
   ```bash
    conda env create -n ENV_NAME -f environment.yaml
   
### Run the app
1. **Run the main script**:
   ```bash
   python app/main.py

## Learn to parametrize

### Observation space

1. **Parameters and distance**: Each satellite observes the keplerian parameters of all satellites (targets and interceptors), as well as their current distance, and its own remaining delta-V. Best for a prograde chases, specifically approach-and-maintain.
2. **Parameters and close approaches**: Each satellite observes the keplerian parameters of all satellites (targets and interceptors), as well as information relative to the next closest approach to each other satellites (approach distance and velocity, and time remaining to the closest approach), and its own remaining delta-V. Best for retrograde seed-and-destroy.

### Action space

Two modes are available.
1. **Keplerian Target Orbit**: Agents learn to choose on which orbit they shall be next. The optimal maneuver delta-V is then determined and applied to the satellite.
2. **TNW Maneuvers**: Agents learn to directly choose the maneuver parameters in the TNW local orbital frame.

### Interceptors objective

1. **Seek-and-destroy**: The distance to targets must be reduced to zero as fast as possible, regardless of relative speed.
2. **Approach-and-maintain (coming)**: The distance to targets must be reduced in a cost-efficient manner, then a set distance must be kept.

### Targets objective

1. **Evade**: Kept a non-null distance with all interceptors at low fuel cost, without any bounds of movement.
2. **Station keeping**: Stay inside a given orbit box, while evading interceptors.

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
├── environment.yaml                # Conda env with Poliastro, RLlib, etc.
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
Released under the **MIT License**.

## Keywords
MARL, Multi-Agent Reinforcement Learning, MADDPG, ASAT, Anti-Satellite