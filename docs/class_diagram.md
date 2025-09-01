This is the workflow:

```
┌────────────────────────────┐
│     OrbitalEnv (PZ)        │  ◀─────── PettingZoo.parallel_env
├────────────────────────────┤
│ agents: Dict[str, Agent]   │
│ step(actions)              │
│ reset()                    │
│ observation_space()        │
│ action_space()             │
└────────────┬───────────────┘
             │ uses
             ▼
┌────────────────────────────┐       ┌────────────────────────────┐
│      SatelliteAgent        │──────▶│       OrbitState           │
├────────────────────────────┤       ├────────────────────────────┤
│ id                         │       │ poliastro.Orbit            │
│ OrbitState                 │       │ propagate_to(t)            │
│ apply_action()             │       │ apply_delta_v()            │
│ get_observation()          │       └────────────────────────────┘
└────────────┬───────────────┘
             │ uses                  ┌────────────────────────────┐
             ▼                       │       EventDetector        │
  ┌──────────────────────────┐       ├────────────────────────────┤
  │  Maneuver Planner (opt)  │       │ compute_closest_approaches │
  └──────────────────────────┘       └────────────────────────────┘
             ▲
             │
             │ wrapped by
             ▼
    ┌───────────────────────┐
    │   PettingZooEnv       │ ◀─── RLlib `PettingZooEnv` wrapper
    └───────────────────────┘
             │
             ▼
    ┌───────────────────────┐
    │        RLlib          │
    └───────────────────────┘
```
