from .base import Dynamics

class RigidBodyDynamics(Dynamics):
    state_shape = (12,) # px, py, pz, psi, theta, phi, vx, vy, vz, r, p, q
    
    # TODO: Implement 12-DOF rigid body dynamics