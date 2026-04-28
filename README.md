# MARL Swarm Interceptor Evasion

![CI](https://github.com/gregoire-marie/marl-swarm-evasion/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gregoire-marie/08984c6d4e53525aafa600104ee7070d/raw/coverage.json)
![Python Version](https://img.shields.io/badge/python-3.9.21-blue.svg)
![License](https://img.shields.io/badge/license-All%20rights%20reserved-red)

This is a project about swarm interceptor satellites evasion using multi-agent reinforcement learning.

## Overview
A mixed target and interceptor satellite swarms cooperative-competitive environment, enabling multi-agent policy optimization with deep reinforcement learning. The current training entry point uses [PPO](https://arxiv.org/abs/1707.06347) through Ray RLlib.

[MADDPG](https://arxiv.org/pdf/1706.02275) is a relevant future algorithm for this setting, but it is not implemented yet.

Target satellites learn to evade a swarm of interceptor satellites dynamically learning seek-and-destroy strategies.
Maneuvers are supported in both the inertial `ECI` frame and the local `TNW` frame.

See [docs/architecture.md](docs/architecture.md) for diagrams of the training/inference architecture and environment step lifecycle.

## Quick Start
### Requirements
- **Hardware**: A good GPU.
- **Software**: Python 3.9.21 (see `pyproject.toml`)

### Install the env (uv)
1. Install dependencies using `uv`:
   ```bash
   make install
   ```

### Run the app
1. **Visualize rewards**:
   ```bash
   uv run python app/visualize_rewards.py
   ```
2. **Run training (PPO)**:
   ```bash
   # Train with custom parameters
   uv run python app/train.py --name ppo_3i_1t_tnw --n-interceptors 3 --n-targets 1 --iterations 100 --num-workers 4 --maneuver-frame TNW
   ```
   
3. **Run inference**:
   ```bash
   # Inference with custom parameters
   uv run python app/infer.py checkpoint --n-interceptors 3 --n-targets 1 --episode-length 250 --maneuver-frame TNW
   ```

## Testing
Run tests using:
```bash
make test
```

## Conventions

This project standardizes physical units, angles, and time across the codebase for correctness and reproducibility:

- Units and types
  - Internals use astropy quantities (Quantity) end-to-end.
  - Distances, orbital velocities, and Δv are in SI units: meters (m) and meters per second (m/s).
  - Angles use mean anomaly M in the public API; internally conversions may use true anomaly ν.
  - Time is handled with astropy.time (Time, TimeDelta). Avoid naive datetime.
  - At RL edges (actions/observations), values are plain numpy float arrays. Convert with .to_value(...) at boundaries.

- Actions
  - 3D delta-v maneuver vectors in the `ECI` or `TNW` frame (m/s).
  - In `TNW` mode, actions are converted to ECI at burn epoch before propagation.
  - Magnitudes are clipped by `env_config["max_delta_v_mps"]`.

- Observations
  - Flat float vectors containing Keplerian elements and derived scalars (e.g., remaining Δv, pairwise distances).
  - Normalization:
    - Semi-major axis: Centered around 7,000,000 m, scaled by 1,000,000 m.
    - Eccentricity: Already in [0, 1].
    - Angles (i, RAAN, argp, M): Normalized to [-1, 1] (wrapped to $[-\pi, \pi]$ then divided by $\pi$).
    - Remaining Δv: Normalized by the initial fuel budget.
    - Distances: Scaled by 1,000,000 m.

- Distances
  - Pairwise ECI distances and similar quantities are expressed in meters (m).

- Seeding and determinism
  - Use utils.random.set_global_seed(seed) or pass seed to env.reset(seed=...) to seed Python, NumPy, and PyTorch (if installed).
  - Tests and examples use fixed epochs and deterministic elements for reproducibility.

## Training

The `app/train.py` script is the main entry point for training the agents.

### Basic Usage

```bash
uv run python app/train.py [OPTIONS]
```

### Common Examples

- **Small scale training**:
  ```bash
  uv run python app/train.py --name ppo_small_1v1 --n-interceptors 1 --n-targets 1 --freeze-targets
  ```
- **Fully parametrized training**:
  ```bash
  uv run python app/train.py --name ppo_small_1v1 --n-interceptors 1 --n-targets 1 --freeze-targets --maneuver-frame tnw --timestep 60.0 --episode-length 100 --iterations 20 --batch-size 1000 --lr 0.0001 --gamma 0.99 --num-epochs 10 --seed 42 --num-workers 8 --checkpoint-freq 10 --local-dir "~/results/marl-swarm-evasion/ray_results"
  ```
- **Train with TNW maneuvers**:
  ```bash
  uv run python app/train.py --name ppo_tnw_2v1 --n-interceptors 2 --n-targets 1 --maneuver-frame tnw
  ```
- **Resume from a previous run**:
  ```bash
  uv run python app/train.py --resume --name ppo_tnw_2v1 --local-dir ~/results/marl-swarm-evasion/ray_results
  ```

### Command-line Arguments

| Argument | Type | Default                                    | Description                                                                                                                                                              |
| :--- | :--- |:-------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Scenario** | |                                            |                                                                                                                                                                          |
| `--n-interceptors` | int | 1                                          | Number of interceptor agents.                                                                                                                                            |
| `--n-targets` | int | 1                                          | Number of target agents.                                                                                                                                                 |
| `--timestep` | float | 60.0                                       | Simulation timestep in seconds.                                                                                                                                          |
| `--episode-length` | int | 100                                        | Maximum number of steps per episode (any collision causes an early termination).                                                                                         |
| `--max-delta-v-mps` | float | 20.0                                       | Maximum single-maneuver delta-v in m/s.                                                                                                                                  |
| `--maneuver-frame` | str | `eci`                                      | Maneuver frame used for actions: `eci` or `tnw`.                                                                                                                         |
| `--freeze-targets` | flag | -                                          | Force targets to apply zero Δv at each step.                                                                                                                             |
| **Training** | |                                            |                                                                                                                                                                          |
| `--iterations` | int | 20                                         | Number of training iterations.                                                                                                                                           |
| `--batch-size` | int | 4000                                       | Training batch size : number of environment timesteps (across all workers) before a weight update. Cause: at least `batch-size`/`episode-length` episodes are performed. |
| `--lr` | float | 5e-5                                       | Learning rate.                                                                                                                                                           |
| `--gamma` | float | 0.99                                       | Discount factor.                                                                                                                                                         |
| `--num-epochs` | int | 10                                         | Number of SGD epochs applied to each training batch (`num_epochs` in RLlib PPO).                                                                                       |
| `--seed` | int | 42                                         | Random seed.                                                                                                                                                             |
| **Execution** | |                                            |                                                                                                                                                                          |
| `--num-workers` | int | 1                                          | Number of rollout workers (parallel envs).                                                                                                                               |
| `--num-gpus` | float | 0                                          | Number of GPUs (can be fractional).                                                                                                                                      |
| `--checkpoint-freq`| int | 1                                          | Frequency of checkpointing.                                                                                                                                              |
| `--resume` | flag | -                                          | Resume training from the last checkpoint.                                                                                                                                |
| `--name` | str | -                                          | Name of the experiment (used as the results subdirectory).                                                                                                              |
| `--local-dir` | str | `~/results/marl-swarm-evasion/ray_results` | Directory for results and checkpoints.                                                                                                                                   |

## Monitoring

You can monitor the training progress in real-time using **TensorBoard**. This allows you to track not only the rewards but also domain-specific success metrics.

### Launching TensorBoard

Point TensorBoard to your results directory (default is `~/results/marl-swarm-evasion/ray_results`):

```bash
tensorboard --logdir ~/results/marl-swarm-evasion/ray_results
```

### Dashboard Layout

Use this page to track the most important metrics:

*   **Success and Failures**: `intercept_success_rate` and `out_of_fuel_rate`.
*   **Collisions**: `interceptors_collision_rate` and `targets_collision_rate`.
*   **Episode Metrics**: Average steps per episode `episode_steps` (custom) and episode length `episode_len_mean` (default).
*   **Training Performance**: Mean episode return.

### Key Metrics to Watch

In the TensorBoard dashboard, you will find several categories of metrics:

1.  **Default Ray RLlib Metrics**:
    *   `ray/tune/env_runners/episode_return_mean`: Overall performance of all agents.
    *   `ray/tune/info/learner/<policy_id>/learner_stats/policy_loss`: Training stability.

2.  **Custom Orbital Metrics** (found under `ray/tune/env_runners/`):
    *   `intercept_success_rate`: Percentage of episodes where an interceptor successfully reached a target.
    *   `interceptors_collision_rate`: Rate of collisions between interceptors.
    *   `targets_collision_rate`: Rate of collisions between targets.
    *   `out_of_fuel_rate`: Percentage of episodes ending because agents ran out of Δv budget.
    *   `reentry_rate`: Percentage of episodes ending because agents reentered the atmosphere.
    *   `episode_steps`: Average number of steps per episode (shorter episodes often indicate early collisions or successes).

These metrics provide a direct view of whether your agents are actually learning the desired orbital behaviors or just maximizing rewards through unintended shortcuts.

## Inference and Visualization

After training your agents, you can run an inference session to visualize the orbital situation and agent behaviors.

### Running Inference

Use the `app/infer.py` script to load a checkpoint and run a single episode:

```bash
uv run python app/infer.py checkpoint --n-interceptors 1 --n-targets 1 --episode-length 100 --maneuver-frame tnw
```

### Command-line Arguments (Inference)

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `checkpoint` | str | - | **Required**. Path to a run directory with `checkpoint_*` folders, or directly to a specific checkpoint directory. |
| `--n-interceptors` | int | checkpoint config | Override the number of interceptor agents. |
| `--n-targets` | int | checkpoint config | Override the number of target agents. |
| `--timestep` | float | checkpoint config | Override the simulation timestep in seconds. |
| `--episode-length` | int | checkpoint config | Override the number of steps per episode. |
| `--max-delta-v-mps` | float | checkpoint config | Override the maximum single-maneuver delta-v in m/s. |
| `--freeze-targets` / `--no-freeze-targets` | bool | checkpoint config | Override whether target agents are forced to apply zero Δv. |
| `--maneuver-frame` | str | checkpoint config | Override the maneuver frame used for actions: `eci` or `tnw`. |
| `--seed` | int | checkpoint config | Override the random seed for the scenario. |
| `--out-dir` | str | `$checkpoint/inference` | Directory to save generated plots. |

### Generated Plots

The script produces several plots in the output directory:

1.  **`trajectories_3d.png`**: A 3D view of the orbital trajectories for all agents, with Earth for reference.
2.  **`metrics_over_time.png`**: Time-series of rewards, remaining fuel (Δv), and action magnitudes for each agent.
3.  **`distances.png`**: Relative distances between all pairs of agents over time (log scale), with the collision threshold highlighted.

## Learn to parametrize

### Observation space

- [x] **Keplerian elements and distance**: Each satellite observes its own normalized Keplerian parameters and remaining fuel, as well as the normalized Keplerian parameters and relative distance of all other satellites.
- [ ] **Close approaches**: (Coming soon) Inclusion of time-to-closest-approach and distance-at-closest-approach in observations.

### Action space

- [x] **ECI Δv Maneuvers**: Agents can choose 3D delta-v vectors in the ECI frame.
- [x] **TNW Maneuvers**: Agents can choose 3D delta-v vectors in the TNW local frame (converted to ECI at burn epoch).
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
  - Pairwise distance checks,
  - Closest-approach utilities.
- **Not implemented yet**: Collision probability estimation is not wired into the environment.
- **Policy Learning**: Teach interceptors to track targets and targets to evade collisions using fuel-efficient maneuvers. The implemented training path uses **PPO** via RLlib.
- **Not implemented yet**: MADDPG support is not currently available.
- **Parametric Scenarios**: Configure LEO pursuit-evasion scenarios with swarm size, maneuvering capacity, and mission objectives.
- **Not implemented yet**: MEO/GEO scenario builders are not currently available.
- **Modular Reward Engine**: Pluggable reward shaping functions (reciprocal distance, logarithmic, quadratic) for different mission goals.
- **Visualization Tools**: Utilities to visualize reward landscapes and simulation results.

## Repository Structure
```
marl-swarm-evasion/
│
├── app/                            # Entry points and scripts
│   ├── train.py                    # RLlib training script (PPO)
│   ├── infer.py                    # Inference and visualization script
│   └── visualize_rewards.py        # Reward shaping visualization tool
│
├── docs/                           # Documentation and diagrams
│   ├── architecture.md             # Runtime architecture and step lifecycle
│   └── class_diagram.md            # Earlier class/workflow sketch
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
│               ├── callbacks.py       # RLlib metrics callbacks
│               ├── constants.py       # Physical & environment constants
│               ├── helpers.py         # Logging & unit conversions
│               ├── normalization.py   # Observation scaling
│               ├── random.py          # Seeding & reproducibility
│               ├── rllib_setup.py     # RLlib config, policies, env factories
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
MARL, Multi-Agent Reinforcement Learning, PPO, MADDPG-planned, ASAT, Anti-Satellite
