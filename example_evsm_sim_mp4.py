"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from matplotlib.animation import FFMpegWriter, FuncAnimation

from simulator.gui.multidrone_viewer import MultiDroneViewer
from simulator.multidrone_simulator_evsm import MultiDroneSimulatorEVSM
from simulator.swarming import EVSMConfig
from simulator.utils.mobility_helper import grid_positions

# Configuración de la simulación
dt = 0.1
num_drones = 60
config = EVSMConfig(
    separation_distance=50.0,
    obstacle_distance=10.0,
    max_acceleration=10.0,
    target_velocity=15.0,
    target_altitude=10.0,
)

# Inicializar simulador y entorno
sim = MultiDroneSimulatorEVSM(num_drones, dt, config=config)
sim.environment.set_rectangular_boundary([-200.0, -100.0], [+200.0, +100.0])
sim.environment.add_circular_obstacle([25.0, 25.0], 25.0)
sim.environment.add_rectangular_obstacle([-125.0, -50.0], [-100.0, +50.0])
sim.environment.add_rectangular_obstacle([100.0, -50.0], [150.0, 0.0])
p0 = grid_positions(num_drones, origin=[-50.0, -50.0], space=5.0, altitude=0.0)
sim.initialize(positions=p0)

# Inicializar visualizador
gui = MultiDroneViewer(sim, fig_size=(12, 6))

# Configuración para exportar a MP4
output_filename = "evsm_simulation.mp4"
fps = 10
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
