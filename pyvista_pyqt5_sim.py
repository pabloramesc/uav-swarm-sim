import numpy as np
import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from multiagent_sim.math.geo import enu2geo, geo2enu
from pyvista_terrain_sim import TerrainVisualizer


class Simulation:
    def __init__(self, visualizer: TerrainVisualizer):
        self.viz = visualizer
        self.disp = 10.0

        self.users_geo = self.viz.get_random_points(num_points=20)
        self.drones_geo = self.viz.get_random_points(num_points=16)

        self.viz.initiate(self.users_geo, self.drones_geo)

        # Setup QTimer to call step periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.start(100)  # every 100 ms

    def step(self):
        # users_dm = np.random.normal(0.0, (self.disp, self.disp, 0.0), size=(20, 3))
        # drones_dm = np.random.normal(0.0, (self.disp, self.disp, 0.0), size=(16, 3))

        # users_enu = geo2enu(self.users_geo, self.viz.home) + users_dm
        # drones_enu = geo2enu(self.drones_geo, self.viz.home) + drones_dm

        # self.users_geo = enu2geo(users_enu, self.viz.home)
        # self.drones_geo = enu2geo(drones_enu, self.viz.home)

        self.viz.update(self.users_geo, self.drones_geo)


if __name__ == "__main__":
    # Ensure a QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viz = TerrainVisualizer(
        dem_path="data/elevation/barcelona_dem.tif",
        img_path="images/fused_image.png"
        )
    sim = Simulation(viz)

    try:
        app.exec_()  # Run the event loop
    except KeyboardInterrupt:
        print("Simulation terminated.")
