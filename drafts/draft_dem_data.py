import os
from simulator.environment import ElevationMap

dem_path = os.path.join("data", "barcelona_dem.tif")
elev = ElevationMap(dem_path)

# elev.plot()
# elev.plot_3d()
elev.save_elevation_image()
elev.save_satellite_image()
