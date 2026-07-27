# Architecture

The Python source stays in the top-level `sim` package so examples can remain
small and research workflows can use an editable install. The package is split
by responsibility:

```text
sim/
├── core/          simulation clock, lifecycle, snapshots, backend protocol
├── agents/        agents, typed registries, and dynamics
├── environment/   boundaries, obstacles, terrain, and initial placement
├── mobility/      shared controller contracts and PID primitive
├── math/          stateless geometry and radio calculations
├── metrics.py     on-demand coverage/connectivity summaries
├── evsm/          EVSM algorithm, controller, monitor, and scenario adapter
├── sdqn/          actions, observations, rewards, environment, policy, trainer
├── network/       optional ns-3 process/protocol integration
├── gui/           Matplotlib viewers
└── utils/         file logging and other small utilities
```

## Dependency direction

The core knows how to advance agents and an optional network backend. It does
not import EVSM, SDQN, TensorFlow, Matplotlib, metrics, or ns-3:

```text
environment + agents ──> core <── network backend
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
               EVSM                SDQN
                 │                   │
                 └──────> GUI <──────┘
```

Algorithm packages assemble or adapt the core; they do not implement a second
physics clock. Metrics and rendering consume the resulting state and can run at
a lower cadence than physics.

## Simulation lifecycle

`Simulator.reset()` initializes every registered agent and returns a
`SimulationSnapshot`. `Simulator.step()` first freezes every agent's
observations, then advances every agent exactly once, updates the optional
backend, and returns another snapshot. This two-phase transition prevents an
agent from observing neighbors that have already moved in the same tick. The
simulator's convenience state properties point at that same snapshot, so
viewers, rewards, and callers cannot observe different steps.

Snapshots are the public, detached transition results. Registry entries are
live implementation objects for scenario adapters and should not be mutated by
callers between transitions.

Initial state is normally passed as a mapping:

```python
snapshot = simulator.reset(
    {
        "gcs": gcs_states,
        "drone": drone_states,
        "user": user_states,
    }
)
```

Scenario-level adapters expose this lifecycle directly:

- `EVSMSimulator` is a configured `Simulator`, not a wrapper around one.
- `SDQNEnvironment` is the only transition implementation for SDQN training
  and evaluation.
- `SDQNSimulator` adds policy action selection to that environment.

## Optional integrations

Flat simulations do not import terrain, GUI, or machine-learning dependencies.
Terrain support loads only when a DEM is requested, and online tile dependencies
load only when a satellite background is requested. DQN/Keras imports happen
only when a policy wrapper is constructed. The core talks to network backends
through a small protocol; network-disabled runs never construct or launch ns-3.
The bundled ns-3 adapter maps contiguous agent IDs directly to node IDs and
supports at most 78 nodes, which keeps its position messages within the bridge's
1,024-byte datagram limit.

## Public imports

Import stable concepts from package surfaces and implementation-specific
classes from their domain:

```python
from sim.core import Simulator
from sim.environment import Environment
from sim.evsm import EVSMConfig, EVSMSimulator
from sim.sdqn.environment import SDQNEnvironment, SDQNEnvironmentConfig
```

Internal modules should use relative imports and avoid importing from aggregate
packages that re-export optional or higher-level functionality.

## Migration from the previous layout

The project is pre-1.0, so obsolete implementations were removed instead of
being kept as ambiguous compatibility layers:

| Previous API | Maintained API |
| --- | --- |
| `sim.simulators.MultiAgentSimulator` | `sim.core.Simulator` |
| legacy EVSM simulator/controller/monitor | `sim.evsm` |
| `GymEnvironment`, `GymConfig` | `SDQNEnvironment`, `SDQNEnvironmentConfig` |
| legacy SDQN simulator/trainer | `sim.sdqn.SDQNSimulator`, `sim.sdqn.SDQNTrainer` |
| simulator metrics/network helpers | `sim.metrics`, `sim.network.NetworkManager` |
| mobility placement helpers | `sim.environment.placement` |
