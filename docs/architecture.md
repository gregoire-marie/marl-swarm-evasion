# Architecture

This document summarizes the runtime shape of the project. The important boundary is
between the reinforcement-learning interface, which exchanges plain NumPy arrays, and
the orbital mechanics core, which keeps physical quantities unit-aware with Astropy.

## Runtime Architecture

```mermaid
flowchart TD
    Train["app/train.py<br/>PPO training entry point"]
    Infer["app/infer.py<br/>checkpoint inference and plots"]

    RLlib["Ray RLlib PPO<br/>multi-agent policies"]
    Wrapper["ParallelPettingZooEnv<br/>RLlib/PettingZoo adapter"]
    Env["OrbitalEnv<br/>PettingZoo ParallelEnv"]

    Scenarios["environment.scenarios<br/>interceptors and targets"]
    Agent["SatelliteAgent<br/>role, fuel budget, observation"]
    OrbitState["OrbitState<br/>poliastro Orbit wrapper"]
    Physics["orbital_meca<br/>propagation, ECI/TNW burns, distances"]
    Rewards["reward_engine<br/>intercept, evasion, fuel, reentry"]
    Callbacks["OrbitalPhysicsCallbacks<br/>TensorBoard metrics"]

    Train --> RLlib
    Infer --> RLlib
    RLlib --> Wrapper
    Wrapper --> Env
    Scenarios --> Env
    Env --> Agent
    Agent --> OrbitState
    OrbitState --> Physics
    Env --> Rewards
    Env --> Callbacks

    Env -.->|observations: normalized float32 vectors| RLlib
    RLlib -.->|actions: 3D delta-v vectors| Env
```

## Environment Step Lifecycle

```mermaid
sequenceDiagram
    participant Policy as RLlib policy
    participant Env as OrbitalEnv
    participant Agent as SatelliteAgent
    participant Orbit as OrbitState
    participant Reward as reward_engine

    Policy->>Env: actions[agent_id] = 3D delta-v
    Env->>Env: advance simulation clock

    loop each active agent
        Env->>Env: default missing action to zero vector
        Env->>Env: clip action by max_delta_v_mps
        alt freeze_targets enabled and role is target
            Env->>Agent: apply_action([0, 0, 0], time)
        else target can maneuver
            Env->>Agent: apply_action(delta-v, time, frame)
        end
        Agent->>Orbit: apply_delta_v(delta-v, time, ECI or TNW)
        Orbit->>Orbit: convert TNW to ECI when needed
        Orbit->>Orbit: rebuild orbit from updated velocity
    end

    loop each active agent
        Env->>Agent: propagate_to(current_time)
        Agent->>Orbit: propagate_to(current_time)
    end

    Env->>Reward: compute_rewards(agent_states)
    Reward-->>Env: rewards and termination flags

    loop each active agent
        Env->>Agent: get_observation(all_agents)
        Agent->>Orbit: read Keplerian state and pairwise distances
        Agent-->>Env: 7N normalized float32 observation
    end

    Env-->>Policy: observations, rewards, terminations, truncations, infos
```

## Main Code Paths

- `app/train.py` is the standard PPO entry point; shared PPO/Tune setup lives in
  `src/main/python/experiment/train.py`.
- `app/curriculum_train.py` is the callback-based curriculum entry point. It
  loads the ordered curriculum config, uses one continuous RLlib/Tune run, and
  delegates stage transitions to `CurriculumCallbacks.on_train_result()`.
- `app/infer.py` loads a checkpoint, computes deterministic actions, steps the
  same environment, and produces trajectory/metric plots.
- `src/main/python/utils/rllib_setup.py` centralizes run configuration, fixed
  multi-agent policy setup, environment creation, curriculum batch trainability,
  and inference action helpers.
- `src/main/python/environment/orbital_env.py` owns the multi-agent simulation
  loop and the RL-facing spaces.
- `src/main/python/agents/satellite_agent.py` owns per-agent state, fuel usage,
  and observation construction.
- `src/main/python/agents/orbit_state.py` wraps Poliastro and keeps maneuver,
  propagation, and Keplerian conversion behavior isolated.
- `src/main/python/environment/reward_engine.py` computes cooperative-competitive
  rewards and terminal condition flags from the current constellation state.
