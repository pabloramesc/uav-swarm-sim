# uav-swarm-sim
An UAV swarm network simulator based on Python and NS-3 (C++)

## Installation

1. Clone the main repository
```bash
git clone https://github.com/pabloramesc/uav-swarm-sim.git
cd uav-swarm-sim
```

2. Optional: create a virtual environment for Python 3.12 (recommended version)

3. Install requirements

```bash
pip install -r requirements.txt
```

4. Update submodules (dqn-lab and ns-3)
```bash
git submodule update --init --recursive
```

5. Install dqn-lab as package
```bash
pip install -e libs/dqn-lab
pip install -r libs/dqn-lab/requirements.txt
```

6. Build NS-3