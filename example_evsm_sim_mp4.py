"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from simulator.gui.multidrone_viewer import MultiDroneViewer
from simulator.multidrone_evsm_simulator import MultiDroneEVSMSimulator

# Configuración de la simulación
dt = 0.1
num_drones = 60
xlim = np.array([-200.0, +200.0])
ylim = np.array([-100.0, +100.0])

# Inicializar simulador y entorno
sim = MultiDroneEVSMSimulator(num_drones, dt)
sim.set_rectangular_boundary((xlim[0], ylim[0]), (xlim[1], ylim[1]))
sim.add_circular_obstacle((25.0, 25.0), 25.0)
sim.add_rectangular_obstacle((-125.0, -50.0), (-100.0, +50.0))
sim.add_rectangular_obstacle((100.0, -50.0), (150.0, 0.0))
sim.set_grid_positions(origin=[-50.0, -50.0], space=5.0)
sim.initialize()

# Inicializar visualizador
gui = MultiDroneViewer(sim, xlim * 1.1, ylim * 1.1)

# Configuración para exportar a MP4
output_filename = "evsm_simulation.mp4"
fps = 30
duration = 120.0  # Duración del video en segundos
num_frames = int(fps * duration)

# Función para actualizar cada frame
def update_frame(frame: int) -> None:
    """
    Actualiza el estado de la simulación y el visualizador para un frame.

    Parameters
    ----------
    frame : int
        Índice del frame actual.
    """
    sim.update()  # Avanzar la simulación
    gui.update(force_render=True, verbose=True)  # Actualizar el visualizador

# Crear la animación
anim = FuncAnimation(gui.fig, update_frame, frames=num_frames, interval=1000 / fps)

# Guardar la animación como archivo MP4
writer = FFMpegWriter(fps=fps, metadata={"title": "EVSM Simulation"})
anim.save(output_filename, writer=writer)

print(f"Simulación exportada a {output_filename}")
