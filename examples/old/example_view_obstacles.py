from simulator.environment.obstacles.obstacles import CircularObstacle, RectangularObstacle, PolygonalObstacle
from simulator.environment.obstacles.boundaries import CircularBoundary, RectangularBoundary, PolygonalBoundary
from simulator.environment.obstacles.visualization import plot_limited_region

# Create a circular obstacle
circ_obs = CircularObstacle([0, 0], 10.0, quad_segs=16)
plot_limited_region(circ_obs)

# Create a rectangular obstacle
rect_obs = RectangularObstacle([-10, -10], [+10, +10])
plot_limited_region(rect_obs)

# Create a polygonal obstacle
poly_obs = PolygonalObstacle([[-10, 0], [-10, -5], [10, -5], [5, 10]])
plot_limited_region(poly_obs)

"""VISUALIZE BOUNDARIES"""
# Create a circular boundary
circ_bound = CircularBoundary([0, 0], 10.0, quad_segs=16)
plot_limited_region(circ_bound)

# Create a rectangular boundary
rect_bound = RectangularBoundary([-10, -10], [+10, +10])
plot_limited_region(rect_bound)

# Create a polygonal boundary
poly_bound = PolygonalBoundary([[-10, 0], [-10, -5], [10, -5], [5, 10]])
plot_limited_region(poly_bound)