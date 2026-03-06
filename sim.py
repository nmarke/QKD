import numpy as np
import matplotlib.pyplot as plt
import constants as const
import loss_calc as loss
from pydantic import BaseModel

# TODO Logintudinal shrot mode for vbration
class sim_vars():
    aircraft_max_velo: float = 105 # m/s
    aircraft_stall_velo: float = 50 # m/s
    animate = True

vars = sim_vars

def R_z(theta: float) -> np.ndarray:
    """
    Docstring for R_z
    
    Returns:
        3x3 numpy array array

    R_z(theta) = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    """
    c, s = np.cos(theta), np.sin(theta)

    return np.array([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1]]
        )

def R_y(theta: float) -> np.ndarray:
    """
    Docstring for R_y
    
    Returns:
        3x3 numpy array array

    R_y(theta) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    """
    c, s = np.cos(theta), np.sin(theta)

    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]]
        )

def R_x(theta: float) -> np.ndarray:
    """
    Docstring for R_x
    
    Returns:
        3x3 numpy array array

    R_x(theta) = [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]
    """
    c, s = np.cos(theta), np.sin(theta)

    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]]
        )

class sim_object(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    
    _DIM = 3
    # position  frame
    x: np.ndarray = np.zeros(_DIM, dtype=float)

    # linear velocity m/s,  frame
    v: np.ndarray = np.zeros(_DIM, dtype=float)

    # linear acceleration m/s2,  frame
    a: np.ndarray = np.zeros(_DIM, dtype=float)

    # heading (1, 0, 0)  default
    h: np.ndarray = np.array([1.0, 0.0, 0.0])

    # angular velocty r/s
    w: np.ndarray = np.zeros(_DIM, dtype=float)

    # angular acceleration r/s2
    wd: np.ndarray = np.zeros(_DIM, dtype=float)

    def hd_velo(self) -> float:
        # Call the methods with ()
        return float(np.dot(self.h, self.v))
    
    def update(self, t_step: float):
        # Update Angular Rate
        self.w += self.wd * t_step

        # Update Heading
        rot = R_z(self.w[2] * t_step) @ R_y(self.w[1] * t_step) @ R_x(self.w[0] * t_step)
        
        # Apply rotation
        self.h = rot @ self.h
        
        # Normalize to prevent the vector from "growing" or "shrinking" over time
        self.h = self.h / np.linalg.norm(self.h)

        # Update Linear Velocity
        self.v += self.a * t_step

        # Update Position
        self.x += self.v * t_step

# ======== setup sim ======== #
# define mission variables

# aircraft
ac_x = 1000.0
ac_y = -1000.0
ac_z = 1000.0
ac_v = 85 # m/s
ac_h = np.array([0.0, 1.0, 0.0], dtype=float) # point left
# alpha = np.pi/4
# ac_h = R_x(alpha) @ ac_h.T # angle up

# turret
t_x = 0
t_y = 0
t_z = 0

# create objects
origin: sim_object = sim_object()
turret: sim_object = sim_object(
    x=np.array([t_x,t_y,t_z], dtype=float)
    )
aircraft: sim_object = sim_object(
    x=np.array([ac_x,ac_y,ac_z], dtype=float),
    h = ac_h,
    v=ac_h*ac_v
)

ac = aircraft
t = turret
o = origin

# time vector
iter = 1000 # number of time steps
elapsed = 50 # number of seconds
time = np.linspace(0,elapsed,iter)

# ======== run sim ======== #
def run_sim():
    print(f"Running Simulation...")

    ac_pos_history = []
    
    # Calculate fixed time step
    dt = time[1] - time[0]

    # angular rate
    w = []

    # beam
    b = []

    # velocity
    v = []
    fv = []

    for step in time:
        # Update with dt, not absolute time
        kp = 0.60 # proportional gain
        los = ac.x - t.x # line of sight vector
        r_squared = np.dot(los, los)
        los_norm = np.linalg.norm(los)
        u = np.cross(t.h, los) / los_norm # error vector

        t.w = kp * u # apply error to acceleration

        t.update(dt)
        ac.update(dt)

        w.append(np.cross(los, ac.v) / r_squared) # angular rate
        v.append(ac.v.copy())
        fv.append(np.dot(ac.h, ac.v).copy())
        ac_pos_history.append(ac.x.copy()) # position
        b.append(t.h.copy()*los_norm) # beam path

    # Convert history to a numpy array for easy slicing [N, 3]
    history = np.array(ac_pos_history)
    true_w_arr = np.array(w)
    b_history = np.array(b)
    v_history = np.array(v)
    fv_history = np.array(fv)
    
    # Find the absolute maximum rate required for each axis
    max_req_w_rad = np.max(np.abs(true_w_arr), axis=0)

    print("\n--- REQUIRED SERVO SPEEDS ---")
    print(f"Max X (Roll) rate: {max_req_w_rad[0]:.3f} rad/s")
    print(f"Max Y (Tilt) rate: {max_req_w_rad[1]:.3f} rad/s")
    print(f"Max Z (Pan)  rate: {max_req_w_rad[2]:.3f} rad/s")

    print("\n--- AIRCRAFT SPECS, s frame ---")
    print(f"Max X velocity: {np.max(v_history[:,0])} m/s")
    print(f"Max Y velocity: {np.max(v_history[:,1])} m/s")
    print(f"Max Z velocity: {np.max(v_history[:,2])} m/s")
    print(f"Max foreward velocity: {np.max(fv_history)} m/s")

    # ======== Plotting ======== #
    if vars.animate: # AI
        import matplotlib.animation as animation

        # ======== Animation Setup ======== #
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Calculate static bounds 
        # (Crucial: If you don't do this, the camera will auto-zoom every frame and give you a headache)
        max_range = np.array([history[:,0].max()-history[:,0].min(), 
                            history[:,1].max()-history[:,1].min(), 
                            history[:,2].max()-history[:,2].min()]).max() / 2.0
        mid_x = (history[:,0].max()+history[:,0].min()) * 0.5
        mid_y = (history[:,1].max()+history[:,1].min()) * 0.5
        mid_z = (history[:,2].max()+history[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (Altitude)')
        ax.set_title('Turret Tracking Animation')

        # 2. Initialize the empty visual elements
        # We create empty lines/points here, and fill them with data in the loop
        ax.scatter([t_x], [t_y], [t_z], color='red', s=100, label='Turret')
        ac_trail, = ax.plot([], [], [], color='blue', alpha=0.3, label='Flight Path')
        ac_pt, = ax.plot([], [], [], marker='o', color='blue', label='Aircraft')
        beam, = ax.plot([], [], [], color='red', linestyle='-', linewidth=2, label='Turret Aim')

        ax.legend()

        # 3. The Update Function (The "Flipbook" page turner)
        def update_graph(num):
            # Update the aircraft trail (draws from start up to current frame 'num')
            ac_trail.set_data(history[:num, 0], history[:num, 1])
            ac_trail.set_3d_properties(history[:num, 2])
            
            # Update the current aircraft dot
            ac_pt.set_data([history[num, 0]], [history[num, 1]])
            ac_pt.set_3d_properties([history[num, 2]])
            
            # Update the Turret Beam
            beam_length = 2000
            current_heading = b_history[num]
            beam_end = origin.x + (current_heading * beam_length)
            
            # Matplotlib 3D lines require setting X/Y first, then Z separately
            beam.set_data([origin.x[0], beam_end[0]], [origin.x[1], beam_end[1]])
            beam.set_3d_properties([origin.x[2], beam_end[2]])
            
            return ac_trail, ac_pt, beam

        # 4. Run the animation
        # interval=20 means 20 milliseconds per frame (roughly 50 FPS)
        ani = animation.FuncAnimation(fig, update_graph, frames=len(history), 
                                    interval=20, blit=False)

        plt.show()
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the aircraft trajectory
        ax.plot(history[:, 0], history[:, 1], history[:, 2], label='Aircraft Path', color='blue')

        # plot the beam trajectory
        ax.plot(b_history[:, 0], b_history[:, 1], b_history[:, 2], label='Beam Path', color='purple')

        # Draw a line from the turret, extending 2000 meters along its final heading
        beam_length = 2000
        final_heading = b_history[-1] # Grab the last frame's heading
        beam_end = origin.x + (final_heading * beam_length)
        
        ax.plot([origin.x[0], beam_end[0]], 
                [origin.x[1], beam_end[1]], 
                [origin.x[2], beam_end[2]], 
                color='red', linestyle='--', label='Turret Aim')

        # Plot the turret/origin point
        ax.scatter([0], [0], [0], color='red', s=100, label='Turret (Origin)')

        # Formatting
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (Altitude)')
        ax.set_title('3D Aircraft Trajectory Relative to Turret')
        ax.legend()
        
        # Optional: Keep the scale 1:1 so it's not distorted
        max_range = np.array([history[:,0].max()-history[:,0].min(), 
                            history[:,1].max()-history[:,1].min(), 
                            history[:,2].max()-history[:,2].min()]).max() / 2.0
        mid_x = (history[:,0].max()+history[:,0].min()) * 0.5
        mid_y = (history[:,1].max()+history[:,1].min()) * 0.5
        mid_z = (history[:,2].max()+history[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()

if __name__ == '__main__':
    run_sim()