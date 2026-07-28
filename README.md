# UAV Swarm Network Simulator

[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen?style=flat-square)](#tests)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![MIT license](https://img.shields.io/github/license/pabloramesc/uav-swarm-sim?style=flat-square)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=261230)](https://docs.astral.sh/ruff/)
[![Last commit](https://img.shields.io/github/last-commit/pabloramesc/uav-swarm-sim?style=flat-square)](https://github.com/pabloramesc/uav-swarm-sim/commits/main/)
[![GitHub stars](https://img.shields.io/github/stars/pabloramesc/uav-swarm-sim?style=flat-square&logo=github)](https://github.com/pabloramesc/uav-swarm-sim/stargazers)

A Python simulator for autonomous UAV swarms restoring wireless coverage in
emergency scenarios. It combines multi-agent dynamics, obstacles and terrain,
Matplotlib visualization, optional NS-3 networking, and two decentralized
deployment algorithms:

- **EVSM** — Extended Virtual Spring Mesh control.
- **SDQN** — multi-agent deep Q-learning with centralized training and
  decentralized execution.

This repository accompanies the master's thesis
[UAV Swarm Network Simulator for Emergency Communications](docs/UAV_Swarm_Network_Simulator_for_Emergency_Communications.pdf).

## Architecture

There is one simulation lifecycle. `sim.core.Simulator` owns the clock and
advances every agent once per step; algorithm packages configure or extend that
core instead of wrapping another simulator.

```text
sim/
├── core/          clock, lifecycle, snapshots, network-backend protocol
├── agents/        agents, registries, and dynamics
├── environment/   boundaries, obstacles, terrain, generation, placement
├── mobility/      generic controller and PID primitives
├── math/          stateless geometry and radio calculations
├── metrics.py     on-demand coverage and connectivity metrics
├── evsm/          EVSM controller, monitor, and simulator
├── sdqn/          actions, frames, rewards, environment, policy, trainer
├── network/       optional NS-3 integration
├── gui/           Matplotlib viewers
└── utils/         logging helpers
```

The important contracts are:

- `Simulator.reset(...)` returns an immutable state snapshot.
- `Simulator.step()` freezes shared observations, advances every agent once,
  and returns the matching snapshot.
- `EVSMSimulator` is a configured core simulator.
- `SDQNEnvironment` is the single transition path shared by training and
  inference.
- Metrics, rendering, TensorFlow, and NS-3 remain outside the generic core.

See [docs/architecture.md](docs/architecture.md) for dependency and lifecycle
details.

## Swarming algorithms

The project implements two decentralized deployment strategies. Both operate
on the same agents, environment, dynamics, and metrics, so their behavior can
be compared without changing the underlying simulation model.

### EVSM — Extended Virtual Spring Mesh

EVSM extends the virtual-spring approach proposed by
[Derr et al. (2011)](https://doi.org/10.1109/TIE.2011.2130492) with explicit
boundary and obstacle avoidance. Each drone treats selected neighbors as
virtual springs. The acute-angle test selects a sparse planar spring mesh,
while the springs' natural length grows toward the configured separation to
deploy the swarm progressively.

Damping stabilizes the formation, and drones on the edge of the mesh receive
an exploration force toward uncovered space. Nearby boundaries and obstacles
override that expansion with a repulsive force. Horizontal EVSM forces are
combined with a PID altitude controller that follows the requested height
above local terrain.

The controller is independent of how neighbor positions arrive:

- **Ideal communication** reads the current positions directly from the agent
  registry.
- **Network communication** uses positions received through `SwarmLink` over
  the NS-3 ad-hoc network. Delayed or expired broadcasts can temporarily alter
  the spring topology.

| Ideal communication | NS-3 communication |
| --- | --- |
| ![EVSM with ideal communication](videos/evsm_sim_ideal.gif) | ![EVSM with network communication](videos/evsm_sim_network.gif) |

These GIFs were generated from the same seeded 120-second scenario using the
current `examples.evsm_video` entry point. The network capture runs Python and
NS-3 in synchronized real time; both are displayed as a 10× time-lapse.

### SDQN — Swarming Deep Q-Network

SDQN applies Deep Q-Learning using Centralized Training with Decentralized
Execution (CTDE). During training, every drone contributes experience to one
shared policy. During execution, each drone selects an action from its own
local observation, without requiring a centralized controller.

SDQN supports Cartesian and log-polar two-channel observations:

- obstacle occupancy around the observing drone;
- nearby users and whether the current swarm covers them.

Cartesian geometry provides uniform spatial resolution. Log-polar geometry
dedicates more cells to nearby detail while retaining a wide view of distant
features. A trained model must use the same representation it was trained
with. Each step selects one of five horizontal actions: hold, up, down, left,
or right.

Coverage rewards can be global, fractional, or based on each drone's marginal
contribution. Boundary, obstacle, and inter-drone collisions are penalized and
terminate the shared episode. The TensorFlow/Keras DQN implementation is
provided by the bundled [dqn-lab](https://github.com/pabloramesc/dqn-lab)
submodule.

#### Cartesian observation

![SDQN Cartesian observation](videos/sdqn_grid_frame.gif)

#### Log-polar observation

![SDQN log-polar observation](videos/sdqn_logpolar_frame.gif)

Both fresh 10-second GIFs use the current `examples.sdqn_video` entry point
with the same seed, scenario, and reproducible random action sequence. This
isolates the difference between observation geometries without implying that
an untracked trained model was used. Pass `--model` instead of
`--random-policy` for policy-driven inference.

## Installation

Python 3.12 is the supported version.

For the core library:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For all Python examples, including visualization, terrain, and SDQN:

```bash
git submodule update --init --recursive libs/dqn-lab
python -m pip install -r requirements.txt
python -m pip install -e libs/dqn-lab
python -m pip install -e .
```

The simulator and EVSM do not import TensorFlow. Machine-learning dependencies
are loaded only when an SDQN model is constructed.

## Examples

Every maintained example is finite and safe to import.

```bash
# Interactive EVSM deployment
python -m examples.evsm_demo

# EVSM over the included Barcelona elevation map
python -m examples.evsm_terrain

# Add an online satellite-tile background
python -m examples.evsm_terrain --satellite

# Render EVSM to MP4
python -m examples.evsm_video --duration 120 --speedup 10

# Run EVSM with positions exchanged through NS-3
# (requires the NS-3 setup described below)
python -m examples.evsm_demo --network --duration 120

# Render the NS-3 communications simulation
python -m examples.evsm_video \
  --network --duration 120 --fps 10 --speedup 10 \
  --output outputs/evsm_network.mp4

# Train a log-polar SDQN model
python -m examples.train_sdqn \
  --model data/models/sdqn_logpolar.keras \
  --log data/logs/sdqn_logpolar.csv

# Train using Cartesian observations
python -m examples.train_sdqn \
  --representation cartesian \
  --model data/models/sdqn_cartesian.keras \
  --log data/logs/sdqn_cartesian.csv

# Evaluate the log-polar model created above
python -m examples.sdqn_demo \
  --model data/models/sdqn_logpolar.keras \
  --steps 300

# Render SDQN inference to MP4
python -m examples.sdqn_video \
  --model data/models/sdqn_logpolar.keras

# Render a Cartesian SDQN model using the same video pipeline
python -m examples.sdqn_video \
  --model data/models/sdqn_cartesian.keras \
  --representation cartesian \
  --output outputs/sdqn_cartesian.mp4

# Reproduce the two README observation GIFs without a trained model
python -m examples.sdqn_video \
  --random-policy --representation cartesian \
  --output videos/sdqn_grid_frame.gif
python -m examples.sdqn_video \
  --random-policy --representation logpolar \
  --output videos/sdqn_logpolar_frame.gif
```

SDQN models and training logs are generated locally under `data/models/` and
`data/logs/`; they are intentionally not tracked. The scripts expose all
configurable options through `--help`.

## Using EVSM

```python
from sim.environment import Environment
from sim.evsm import EVSMConfig, EVSMSimulator

environment = Environment()
environment.set_rectangular_boundary((0.0, 0.0), (1_000.0, 1_000.0))

simulator = EVSMSimulator(
    environment,
    num_drones=9,
    num_users=20,
    config=EVSMConfig(),
    seed=7,
)
snapshot = simulator.reset(home=(100.0, 100.0), spacing=10.0)
snapshot = simulator.step()
simulator.close()
```

## Tests

Run the lightweight unit suite with:

```bash
python -m unittest discover -s tests -t . -v
```

The tests do not launch NS-3 or load TensorFlow. The latest verified result is
shown by the badge at the top of this README.

## NS-3 integration

The simulator combines the Python multi-agent model with the
[NS-3 network simulator](https://www.nsnam.org/) in C++. Python remains
authoritative for motion, obstacles, controllers, and metrics; NS-3 models the
802.11 ad-hoc channel, IP stack, routing, packet delivery, and network timing.

The two processes communicate through the project-specific **SimBridge**
protocol over local UDP sockets. The C++ bridge listens on port `9000`, while
the Python client listens for replies and delivered packets on port `9001`.

<div align="center">
  <img src="images/simbridge-diagram.png" width="55%" alt="SimBridge architecture">
</div>

At runtime:

1. `NetworkManager` sends agent positions and ingress packets from Python.
2. The C++ `SimBridge` polls commands, updates NS-3 mobility models, and injects
   packets into the appropriate node.
3. NS-3 receive callbacks return delivered packets to Python, while request
   replies expose node addresses, positions, and simulation time.
4. Each `SwarmLink` processes the resulting packets and maintains its local,
   timeout-aware view of neighboring drones and users.

NS-3 uses its real-time best-effort scheduler. Synchronized simulator steps
wait for both wall time and NS-3 time; if the network process cannot catch up
within the configured tolerance, the Python side raises a timeout instead of
silently consuming stale network state. The current single-datagram position
protocol supports at most 78 nodes.

To prepare the bundled pinned NS-3 checkout:

```bash
cd ns3
bash setup.sh
```

The script initializes the pinned NS-3 submodule, copies the bridge source into
its scratch tree, and builds only the required `main` bridge target. Install
the platform build prerequisites from the
[NS-3 installation guide](https://www.nsnam.org/docs/installation/html/index.html)
first. The network-enabled commands in [Examples](#examples) then launch and
stop the bridge automatically.

## License

Licensed under the [MIT License](LICENSE).
