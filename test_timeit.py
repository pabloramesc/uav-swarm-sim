from shapely import Point, shortest_line
from shapely.ops import nearest_points
from timeit import timeit
import numpy as np

# Crear un polígono complejo (círculo aproximado)
circle = Point(0, 0).buffer(10, resolution=32)
test_point = Point(15, 5)

# Función usando shortest_line
def using_shortest_line():
    line = shortest_line(circle.boundary, test_point)
    closest = np.array(line.coords[0])
    return closest

# Función usando nearest_points
def using_nearest_points():
    nearest = nearest_points(circle.boundary, test_point)[0]
    closest = np.array(nearest.xy)
    return closest

# Medir tiempos
time_shortest_line = timeit(using_shortest_line, number=10000)
time_nearest_points = timeit(using_nearest_points, number=10000)

print(f"Shortest line time: {time_shortest_line} s")
print(f"Nearest line time: {time_nearest_points} s")
