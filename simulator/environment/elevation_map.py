"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio


class ElevationMap:
    def __init__(self, dem_path: str):
        self.dem_path = dem_path
        self.bounds = None
        self.resolution = None
        self.elevation_data: np.ndarray = None

        # Cargar el archivo DEM
        self.load_dem(self.dem_path)
        
    @property
    def max_elevation(self) -> float:
        return self.elevation_data.max()
        
    def load_dem(self, dem_path: str):
        with rasterio.open(dem_path) as dem:
            self.bounds = dem.bounds
            self.resolution = dem.res
            self.elevation_data = dem.read(1)

    def get_elevation(self, lat: float, lon: float) -> float:
        # Verificar si las coordenadas están dentro de los límites
        if not (
            self.bounds.left <= lon <= self.bounds.right
            and self.bounds.bottom <= lat <= self.bounds.top
        ):
            return 0.0

        # Calcular los índices del array
        col = int((lon - self.bounds.left) / self.resolution[0])  # Índice de columna
        row = int((self.bounds.top - lat) / self.resolution[1])  # Índice de fila

        # Leer el valor de elevación
        return self.elevation_data[row, col]

    def plot(self):
        """Visualiza el mapa de elevación."""
        plt.imshow(
            self.elevation_data,
            cmap="terrain",
            extent=(
                self.bounds.left,
                self.bounds.right,
                self.bounds.bottom,
                self.bounds.top,
            ),
        )
        plt.colorbar(label="Elevación (m)")
        plt.title("Relieve 3D")
        plt.xlabel("Longitud (deg)")
        plt.ylabel("Latitud (deg)")
        plt.show()

    def plot_3d(self):
        # Crear una malla de coordenadas (X, Y) basada en los límites y la resolución
        x = np.linspace(
            self.bounds.left, self.bounds.right, self.elevation_data.shape[1]
        )
        y = np.linspace(
            self.bounds.bottom, self.bounds.top, self.elevation_data.shape[0]
        )
        X, Y = np.meshgrid(x, y)

        # Crear la figura y el gráfico 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Graficar la superficie
        surf = ax.plot_surface(
            X,
            Y,
            self.elevation_data,
            cmap="terrain",
            edgecolor="none",
            linewidth=0,
            antialiased=True,
        )

        # Agregar una barra de color
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevación (m)")

        # Etiquetas y título
        ax.set_title("Relieve 3D")
        ax.set_xlabel("Longitud (deg)")
        ax.set_ylabel("Latitud (deg)")
        ax.set_zlabel("Elevación (m)")

        # Mostrar el gráfico
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta al archivo DEM
    dir_path = "data"
    file_name = "barcelona_dem.tif"
    file_path = os.path.join(dir_path, file_name)

    # Crear el objeto ElevationMap
    elevation_map = ElevationMap(file_path)

    # Consultar la altitud en una coordenada específica
    latitude = 41.3851  # Latitud de Barcelona
    longitude = 2.1734  # Longitud de Barcelona
    altitude = elevation_map.get_elevation(latitude, longitude)
    print(f"Altitud en latitud {latitude}, longitud {longitude}: {altitude} m")

    # Visualizar el mapa de elevación
    elevation_map.plot()

    # Visualizar el mapa de elevación en 3D
    elevation_map.plot_3d()