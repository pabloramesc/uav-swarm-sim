from setuptools import setup, find_packages

setup(
    name="multiagent_sim",
    version="0.1.0",
    author="pabloramesc",
    url="https://github.com/pabloramesc/uav-swarm-sim",
    packages=find_packages(include=["multiagent_sim", "multiagent_sim.*"]),
    python_requires=">=3.12",
)
