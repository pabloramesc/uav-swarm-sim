from simulator.environment.avoid_regions import *

"""TEST AVOID REGIONS"""
# Create a circular avoid region
circ_obs = CircularRegion([0, 0], 10.0, quad_segs=16)
plot_limited_region(circ_obs)

# Create a rectangular avoid region
rect_obs = RectangularRegion([-10, -10], [+10, +10])
plot_limited_region(rect_obs)

# Create a polygonal avoid region
poly_obs = PolygonalRegion([[-10, 0], [-10, -5], [10, -5], [5, 10]])
plot_limited_region(poly_obs)

"""TEST OBSTACLES"""
# Create a circular obstacle
circ_obs = CircularObstacle([0, 0], 10.0, quad_segs=16)
plot_limited_region(circ_obs)

# Create a rectangular obstacle
rect_obs = RectangularObstacle([-10, -10], [+10, +10])
plot_limited_region(rect_obs)

# Create a polygonal obstacle
poly_obs = PolygonalObstacle([[-10, 0], [-10, -5], [10, -5], [5, 10]])
plot_limited_region(poly_obs)

"""TEST BOUNDARIES"""
# Create a circular boundary
circ_bound = CircularBoundary([0, 0], 10.0, quad_segs=16)
plot_limited_region(circ_bound)

# Create a rectangular boundary
rect_bound = RectangularBoundary([-10, -10], [+10, +10])
plot_limited_region(rect_bound)

# Create a polygonal boundary
poly_bound = PolygonalBoundary([[-10, 0], [-10, -5], [10, -5], [5, 10]])
plot_limited_region(poly_bound)
