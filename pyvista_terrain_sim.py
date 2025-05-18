import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import rasterio

from multiagent_sim.math.geo import geo2enu, enu2geo


class TerrainVisualizer:
    def __init__(self, dem_path: str, img_path: str = None):
        self.dem_path = dem_path
        self.img_path = img_path

        self._load_terrain()

        if self.img_path:
            self._load_texture()

        self.plotter = pvqt.BackgroundPlotter()
        if self.img_path:
            self.plotter.add_mesh(self.terrain, texture=self.texture)
        else:
            self.plotter.add_mesh(
                self.terrain,
                scalars=self.terrain["elev"],
                cmap="terrain",
                show_scalar_bar=False,
            )

    def _load_terrain(self):
        with rasterio.open(self.dem_path) as dem:
            self.bounds = dem.bounds
            self.resolution = dem.res
            self.transform = dem.transform
            elevation = dem.read(1)

        self.home = (self.bounds.left, self.bounds.bottom, 0.0)

        nrows, ncols = elevation.shape
        lon = np.linspace(self.bounds.left, self.bounds.right, ncols)
        lat = np.linspace(self.bounds.top, self.bounds.bottom, nrows)

        x, y = np.meshgrid(lon, lat)
        z = np.zeros_like(x)

        mesh = pv.StructuredGrid(x, y, z)
        mesh["elev"] = elevation.ravel(order="F")

        # self.terrain = mesh.warp_by_scalar(scalars="elev", factor=0.000015)
        self.terrain = mesh.warp_by_scalar(scalars="elev", factor=0.00005)

    def _load_texture(self):
        with rasterio.open(self.img_path) as img:
            rgb = img.read([1, 2, 3])  # Read RGB bands
            rgb = np.moveaxis(rgb, 0, -1)  # Rearrange to (rows, cols, 3)

            # Normalize if needed
            if rgb.dtype != np.uint8:
                rgb = (255 * (rgb / rgb.max())).astype(np.uint8)

        # Convert terrain mesh to PolyData to support texturing
        poly = self.terrain.extract_surface()

        # Compute texture coordinates based on mesh x-y positions
        pts = poly.points
        x = pts[:, 0]
        y = pts[:, 1]
        minx, maxx = self.bounds.left, self.bounds.right
        miny, maxy = self.bounds.bottom, self.bounds.top
        u = (x - minx) / (maxx - minx)
        v = (y - miny) / (maxy - miny)
        # Assign UVs
        uv = np.column_stack((u, v))
        poly.active_t_coords = uv

        # Create PyVista texture from image
        self.texture = pv.numpy_to_texture(rgb)
        # Use the textured PolyData as terrain
        self.terrain = poly

    def get_random_points(self, num_points=1):
        """
        Sample a random point (x, y, z) on the terrain surface.
        """
        pts = self.terrain.points
        idx = np.random.choice(pts.shape[0], size=num_points)
        return pts[idx]

    def initiate(self, user_positions, drone_positions):
        self.user_points = pv.PolyData(user_positions)
        self.drone_points = pv.PolyData(drone_positions)

        self.user_actor = self.plotter.add_points(
            self.user_points,
            color="blue",
            point_size=10,
            render_points_as_spheres=True,
            reset_camera=False,
        )
        self.drone_actor = self.plotter.add_points(
            self.drone_points,
            color="red",
            point_size=12,
            render_points_as_spheres=True,
            reset_camera=False,
        )
        self.show()

    def update(self, user_positions, drone_positions):
        self.user_points.points = user_positions
        self.drone_points.points = drone_positions

        # self.plotter.update()
        self.plotter.render()
        self.plotter.app.processEvents()

    def show(self):
        self.plotter.show()


if __name__ == "__main__":
    import time

    viz = TerrainVisualizer(
        dem_path="data/elevation/barcelona_dem.tif",
        img_path="images/fused_image.png",
    )

    users_geo = viz.get_random_points(num_points=10)
    drones_geo = viz.get_random_points(num_points=10)

    viz.initiate(users_geo, drones_geo)

    disp = 10.0
    for t in range(100):
        users_dm = np.random.normal(0.0, (disp, disp, 0.0), size=(10, 3))
        drones_dm = np.random.normal(0.0, (disp, disp, 0.0), size=(10, 3))

        users_enu = geo2enu(users_geo, viz.home) + users_dm
        drones_enu = geo2enu(drones_geo, viz.home) + drones_dm

        users_geo = enu2geo(users_enu, viz.home)
        drones_geo = enu2geo(drones_enu, viz.home)

        viz.update(users_geo, drones_geo)

        time.sleep(0.1)
