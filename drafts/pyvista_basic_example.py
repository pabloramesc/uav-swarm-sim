# Import necessary libraries
import pyvista as pv
import rioxarray as riox
import numpy as np
 
# Read the data from a DEM file
data = riox.open_rasterio("data/elevation/barcelona_dem.tif")
data = data[0]
 
# Save the raster data as an array
values = np.asarray(data)
 
# Create a mesh grid
x, y = np.meshgrid(data['x'], data['y'])
 
# Set the z values and create a StructuredGrid
z = np.zeros_like(x)
mesh = pv.StructuredGrid(x, y, z)
 
# Assign Elevation Values
mesh["Elevation"] = values.ravel(order='F')
 
# Warp the mesh by scalar
topo = mesh.warp_by_scalar(scalars="Elevation", factor=0.000015)
 
# Plot the elevation map
p = pv.Plotter()
p.add_mesh(mesh=topo, scalars=topo["Elevation"], cmap='terrain')
p.show_grid(color='black')
p.set_background(color='white')
p.show(cpos="xy")