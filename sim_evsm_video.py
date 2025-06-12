import os
import imageio.v2 as imageio

from multiagent_sim.core.evsm_simulator import EVSMSimulator, EVSMConfig
from multiagent_sim.gui.evsm_viewer import EVSMViewer

dt = 0.01
num_drones = 25
num_users = 10
size = 1e3
grid_spacing = 10.0

config = EVSMConfig(
    separation_distance=350.0,
    obstacle_distance=20.0,
    max_acceleration=10.0,
    target_altitude=0.0,
    initial_natural_length=grid_spacing,
    natural_length_rate=5.0,
)
sim = EVSMSimulator(
    num_drones=num_drones,
    num_users=num_users,
    num_gcs=1,
    dt=dt,
    use_network=True,
    evsm_config=config,
)

sim.environment.set_rectangular_boundary([0, 0], [size, size])
sim.environment.add_circular_obstacle(center=[600, 600], radius=100)
sim.environment.add_rectangular_obstacle(bottom_left=[200, 600], top_right=[300, 800])
sim.environment.add_rectangular_obstacle(bottom_left=[600, 200], top_right=[800, 300])
sim.environment.add_rectangular_obstacle(bottom_left=[1e3, 1e3], top_right=[1e3, 1e3])

sim.initialize(home=[200, 200], spacing=grid_spacing, altitude=config.target_altitude)

gui = EVSMViewer(sim, show_legend=True, fig_size=(10, 8))

video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, "evsm_sim_network.mp4")
frames = []
last_capture_time = -1.0

while sim.sim_time <= 120.0:
    sim.update()
    fps = gui.update()
    if sim.sim_time - last_capture_time >= 1.0:
        frame = gui.capture_frame()
        frames.append(frame)
        last_capture_time = sim.sim_time
    print(f"Sim time: {sim.sim_time:.2f} s, FPS: {gui.fps:.2f}", end="\r")

print(f"\nTotal frames: {len(frames)}")
imageio.mimsave(video_path, frames, fps=10, format='ffmpeg')
print(f"\nVideo guardado en: {video_path}")