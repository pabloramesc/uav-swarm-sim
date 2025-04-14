import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Line3DCollection


fig = plt.figure()
ax2d: Axes = fig.add_subplot(121)
ax3d: Axes3D = fig.add_subplot(122, projection="3d")

lines2d: Line2D = ax2d.plot([], [])[0]
lines3d: Line3D = ax3d.plot([], [], [])[0]

x = [0.0, 1.0, None, -1.0, 1.0, None]
y = [0.0, 1.0, None, 0.0, -1.0, None]
z = [0.0, 1.0, None, 0.0, 1.0, None]

lines2d.set_data(x, y)

ax2d.set_xlim(-1, +1)
ax2d.set_ylim(-1, +1)

xyz = np.array([x, y, z], dtype=np.float32).T
num_lines = xyz.shape[0] // 3
# for i in range(num_lines):
#     seg_x, seg_y, seg_z = xyz[i * 3 : i * 3 + 2].T
#     line = Line3D(seg_x, seg_y, seg_z)
#     ax3d.add_line(line)

segments = [xyz[i * 3 : i * 3 + 2] for i in range(num_lines)]
lines = Line3DCollection(segments)
ax3d.add_collection3d(lines)

plt.pause(1.0)

lines.remove()

plt.pause(1.0)

ax3d.set_xlim(-1, +1)
ax3d.set_ylim(-1, +1)
ax3d.set_zlim(-1, +1)

plt.show()
