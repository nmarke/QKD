import numpy as np
import matplotlib.pyplot as plt
import loss_calc as loss
from pydantic import BaseModel

class env_consts:
    tx_a = 0.102     # tx aperture 10.2 cm
    rx_a = 0.696     # rx aperture 40.64 cm (16 in)
    link_range = 35e3  # 35 km
    jitter = 5e-6       # 5 urad
    Cn2_l = 10e-17        # Light turbulance
    Cn2_m = 10e-14        # Medium turbulance
    Cn2_h = 10e-11        # Heavy turbulance
    Cn2 = Cn2_l

    # aircraft
    ac_x = np.array([1.0, 0.0, 0.0])
    v = 50.0 # m/s
    ac_h = np.array([1.0, 0.0, 0.0], dtype=float) # point left
    ac_v = ac_h * v
    
    #aircraft turret
    act_h = np.array([-1.0, 0.0, 0.0]) # point straight behind

    # ground turret
    t_x = np.array([0.0, 0.0, 0.0])
    t_h = np.array([1.0, 0.0, 0.0]) # point straigh ahead

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

class gausian_beam(BaseModel):
    """
    beam properties
    """
    wavelength_m: float = 1550e-9 # beam wavelength, meters
    M2: int = 1 # beam quality factor
    A_t_m: float = 10e-3 # telescope aperture in meters

    @property
    def k(self) -> float:
        """wavenumber in rad/m"""
        return 2 * np.pi / self.wavelength_m
    
    @property
    def theta(self) -> float:
        """
        Divergence half-angle in radians
        Rayleigh criterion for circular aperture - approximation
        valid when aperture is the beam-defining element
        """
        return 1.22 * self.wavelength_m / env_consts.tx_a

    @property
    def W0(self) -> float:
        """
        Initial beam waist in meters
        Derived from divergence angle and beam quality factor
        """
        return self.M2 * self.wavelength_m / (np.pi * self.theta)
    
    def rho0(self, z: float, Cn2: float) -> float:
        """Atmospheric spatial coherence radius"""
        return (0.55 * Cn2 * self.k**2 * z) ** (-3/5)

    def beam_size(self, z: float, Cn2: float) -> float:
        """
        Long-term beam size under turbulence (m)
        Scriminich et al. 2022, equation (7) - full expression
        Valid across all distances, not just the far-field limit
        """
        rho = self.rho0(z, Cn2)
        diffraction_term = (self.wavelength_m * z) / (np.pi * self.W0**2)
        turbulence_factor = 1 + (2 * self.W0**2) / (rho**2)
        return self.W0 * np.sqrt(1 + turbulence_factor * diffraction_term**2)
    
    def beam_size_asymptotic(self, z: float, Cn2: float) -> float:
        """
        Long-term beam size - strong turbulence / long distance limit (m)
        Scriminich et al. 2022, equation (10)
        Only valid when z >> Rayleigh range
        """
        return (self.wavelength_m * np.sqrt(2) / np.pi) * \
               ((0.55 * Cn2 * self.k**2) ** (3/5)) * \
               (z ** (8/5))
    
    def beam_wander_fluctuations(self, z: float, Cn2: float) -> float:
        """
        Alessia Scriminich etal 2022 QuantumSci.Technol. 7 045029

        "variance of beam-wander fluctuations at the receiver aperture plane."
        """
        r2c = 2.42*Cn2*z**3*self.W0**(-1/3)

        return r2c

    def beam_size_st(self, z: float, Cn2: float) -> float:
        """
        Alessia Scriminich etal 2022 QuantumSci.Technol. 7 045029

        instantaneous short-term (ST) beam size WST(z) at a distance z
        
        """
        Wz = self.beam_size(z, Cn2)
        r2c = self.beam_wander_fluctuations(z, Cn2)
        Wst = np.sqrt(Wz**2 - r2c)

        return Wst

def norm_plane(x: sim_object) -> tuple[np.ndarray, np.ndarray]:
    """
    defiene a plane normal to the heading
    """
    h = x.h # define heading

    up = np.array([0, 0, 1]) # up vector

    if abs(np.dot(h, up)) >= 0.99: # check to see if almost vertical
        up = np.array([1, 0, 0])

    u = np.cross(h, up) # find first vector
    u = u / np.linalg.norm(u)

    v = np.cross(h, u) # find second vector
    v = v / np.linalg.norm(v)

    return u, v

def fire_laser(beam: gausian_beam, # Gausian beam object
               tx: sim_object, # turrent one
               rx: sim_object,
               num_samples: float,
               Cn2: float):
    """
    perform monte carlo sim of laser between targets
    """
    # define distance between turrets
    z = np.linalg.norm(tx.x - rx.x)

    # define rx plane (ground) using heading and position
    u, v = norm_plane(rx)
    
    # define beam center position
    beam_center_3d = tx.x + tx.h * z
    offset = beam_center_3d - rx.x # rx will be at 0,0,0 for most simulations

    # define center of projected beam from tx onto rx plane
    offset_u = np.dot(offset, u)
    offset_v = np.dot(offset, v)

    # calculate beam wander
    mean_random = beam.beam_wander_fluctuations(z, Cn2)
    sigma_wander = np.sqrt(mean_random / 2)

    # apply beam wander effects
    offset_u += np.random.normal(0, sigma_wander)
    offset_v += np.random.normal(0, sigma_wander)

    # define size of tx beam at rx plane
    Wz = beam.beam_size_st(z, Cn2)

    # define beam distribution
    sigma_beam = Wz / 2

    samples_u = np.random.normal(offset_u, sigma_beam, num_samples)
    samples_v = np.random.normal(offset_v, sigma_beam, num_samples)

    # distance of each sample from rx center
    r = np.sqrt(samples_u**2 + samples_v**2)
    hits = np.sum(r <= env_consts.rx_a)
    efficiency = hits / num_samples

    return efficiency, Wz, samples_u, samples_v, offset_u, offset_v

def plot_fire_laser(samples_u, samples_v, offset_u, offset_v, Wz, efficiency):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(samples_u, samples_v, s=0.5, alpha=0.3, color='blue', label='samples')

    # rx aperture circle
    rx_circle = plt.Circle((0, 0), env_consts.rx_a,
                            color='green', fill=False, linewidth=2, label='rx aperture')
    ax.add_patch(rx_circle)

    # beam footprint at rx plane - radius is Wz (1/e² beam radius)
    beam_circle = plt.Circle((offset_u, offset_v), Wz,
                             color='red', fill=False, linewidth=2, label='beam waist at rx')
    ax.add_patch(beam_circle)

    ax.plot(0, 0, 'g+', markersize=10)
    ax.plot(offset_u, offset_v, 'r+', markersize=10)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlabel('u (m)')
    ax.set_ylabel('v (m)')
    ax.set_title(f'Monte Carlo Beam Simulation — efficiency: {efficiency:.4f}')
    plt.show()
    
def waist_size_vs_efficiency_instance(beam, air_turret, ground_turret):
    """
    Plot the waist size and efficiency per instance
    """
    e = 1
    wz = beam.W0

    e_history = []
    wz_history = []
    x = np.linspace(0, int(env_consts.link_range), int(env_consts.link_range/100))
    for i in x:
        i = int(i)
        e, wz, samples_u, samples_v, offset_u, offset_v = fire_laser(
            beam,
            air_turret,
            ground_turret,
            1000,
            env_consts.Cn2
        )

        e_history.append(e)
        wz_history.append(wz)
        air_turret.x = np.array([i, 0, 0])
        if e<0.5: 
            x = x[:len(e_history)]
            break

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Beam waist (m)")

    ax1.plot(x, wz_history, 'bo', label="beam waist")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Efficiency (%)")
    ax2.plot(x, e_history, 'go', label="efficiency", )

    plt.suptitle('efficiency and beam waist as a function of distance')

    plt.show()

    plot_fire_laser(samples_u, samples_v, offset_u, offset_v, wz, e)

    # most recent sim #
    print(f"position of air turret: {air_turret.x}")
    print(f"Link established at distance: {np.linalg.norm(air_turret.x - ground_turret.x)} meters")
    print(f"Beam width at transmiter was: {beam.W0*2} meters")
    print(f"Beam width at receiver was: {wz} meters")
    print(f"Collection efficiency was: {e*100}%")

def maist_sze_vs_effcency_time(beam, air_turret, ground_turret):
    """
    Plot the waist size and efficiency per instance
    """
    ac = air_turret
    t = ground_turret

    e = 1
    wz = beam.W0

    e_history = []
    wz_history = []
    x = []

    # time vector
    iter = 1000 # number of time steps
    elapsed = 60*5 # number of seconds
    time = np.linspace(0,elapsed,iter)
    dt = elapsed/1000

    print(dt)

    for _ in time:
        e, wz, samples_u, samples_v, offset_u, offset_v = fire_laser(
            beam,
            ac,
            t,
            1000,
            env_consts.Cn2
        )

        x.append(ac.x[0])
        e_history.append(e)
        wz_history.append(wz)

        t.update(dt)
        ac.update(dt)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Beam waist (m)")

    ax1.plot(x, wz_history, 'bo', label="beam waist")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Efficiency (%)")
    ax2.plot(x, e_history, 'go', label="efficiency", )

    plt.suptitle('efficiency and beam waist as a function of distance')

    plt.show()

    plot_fire_laser(samples_u, samples_v, offset_u, offset_v, wz, e)

    # most recent sim #
    print(f"position of air turret: {air_turret.x}")
    print(f"Link established at distance: {np.linalg.norm(air_turret.x - ground_turret.x)} meters")
    print(f"Beam width at transmiter was: {beam.W0*2} meters")
    print(f"Beam width at receiver was: {wz} meters")
    print(f"Collection efficiency was: {e*100}%")

def run_sim():
    beam: gausian_beam = gausian_beam()
    ground_turret: sim_object = sim_object(
        x=env_consts.t_x,
        h=env_consts.t_h
    )
    air_turret: sim_object = sim_object(
        x=env_consts.ac_x,
        v=env_consts.ac_v,
        h=env_consts.act_h
    )

    maist_sze_vs_effcency_time(beam, air_turret, ground_turret)
    # waist_size_vs_efficiency_instance(beam, air_turret, ground_turret)

if __name__ == '__main__':
    run_sim()
