"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os

import matplotlib.pyplot as plt
import rasterio

# Path del archivo DEM
dir_path = "data"
file_name = "barcelona_dem.tif"
file_path = os.path.join(dir_path, file_name)

# Cargar el archivo DEM
with rasterio.open(file_path) as dem:
    bounds = dem.bounds
    resolution = dem.res
    elevation = dem.read(1)

# Visualizar el relieve
plt.imshow(
    elevation,
    cmap="terrain"
)
plt.colorbar(label="Elevación (m)")
plt.title("Relieve 3D de Barcelona")
plt.xlabel("Longitud (deg)")
plt.ylabel("Latitud (deg)")
plt.show()
