# MARL Swarm Interceptor Evasion

## Overview
An environment enabling many-to-many optimization in space.
And MARL models to learn multiple interceptor evasion by a swarm of targets.

## Features
- **Orbital environment**: Based on [`Poliastro`](https://docs.poliastro.space/en/stable/) and [`Astropy`](https://www.astropy.org/).
- **MARL Algorithms**: Based on [`PyTorch`](https://pytorch.org/).
- **Chase scenarios**: Multiple aggressor initial orbits and objectives through parameterizable objective functions.

## Repository Structure
```
├── app/                        # Launchable scripts
├── docs/                       # Documentation
└── src/
    └── main/
        └── python/
            ├── plot/           # Plotting utilities & real-time visualization
            ├── environment/    # Orbital dynamics environment
            └── models/         # MARL Models
```

## Quick Start
### Requirements
- **Hardware**: A good GPU.
- **Software**: 
  - Python 3.9 (see requirements.txt)
  - 

### Install the env (Conda version)
1. Install basic dependencies and PyTorch:
   ```bash
    conda env create -f environment.yml
2. Install RLlib and Gymnasium:
   ```bash
    pip install "ray[rllib]" gymnasium

### Run the app
1. **Run the main script**:
   ```bash
   python app/main.py

## License
Released under the **MIT License**.