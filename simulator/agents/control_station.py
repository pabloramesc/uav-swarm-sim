from simulator.agents.agent import Agent


class ControlStation(Agent):
    def __init__(self, id: int) -> None:
        super().__init__(id, type="gcs")

    def update(self, dt: float = 0.01) -> None:
        super().update(dt)