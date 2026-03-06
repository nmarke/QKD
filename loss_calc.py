import numpy as np
import constants as const
import matplotlib.pyplot as plt

DEBUG = 0

def calculate_visibility(Lw, N):
    """
    Calculates visibility V (in km) based on cloud microphysics.
    Formula: V = 1.002 / ((Lw * N)**0.6473)
    
    Lw: Liquid Water Content (g/m^3)
    N:  Number Concentration (cm^-3)
    """
    term = Lw * N
    visibility_km = 1.002 / (term ** 0.6473) if term > 0 else 50
    if DEBUG: print(f"visibility:: {visibility_km}")
    return visibility_km

def _atmospheric_attenuation(distance_m, wavelength_m):
    """
    Calculates atmospheric transmission for a horizontal/terrestrial link 
    using the Kim Model and Beer-Lambert Law.
    
    distance_m:    Link range in meters
    visibility_km: Atmospheric visibility in km (e.g., 20=clear, 0.2=heavy fog)
    wavelength_m:  Operating wavelength in meters
    
    Returns:       Transmittance (0.0 to 1.0)
    """
    visibility_km = calculate_visibility(const.Lw, const.N)
    
    lam_nm = wavelength_m * 1e9
    
    # account for the particle size distribution of fog/haze, Kims formula  
    if visibility_km > 50:
        q = 1.6
    elif 6 < visibility_km <= 50:
        q = 1.3
    else:
        q = 0.585 * (visibility_km ** (1/3))

    #bCalculate Attenuation Coefficient, mie integral approximation
    sigma = (3.91 / visibility_km) * ((lam_nm / 550) ** -q)

    # Calculate Transmittance using Beer-Lambert Law
    dist_km = distance_m / 1000.0
    transmission = np.exp(-sigma * dist_km)

    return transmission

def monte_carlo_efficiency_with_turbulence(A_T, A_R, L, sigma, resolution, num_samples, Cn2):
    """
    Calculates Efficiency combining:
    1. Geometric Loss (Beam Spread)
    2. Pointing Error (Jitter)
    3. Turbulence (Scintillation/Fading)
    """
    wavelength = const.lmda
    k = 2 * np.pi / wavelength
    
    # --- 1. SETUP BEAM GEOMETRY ---
    theta_divergence = 1.22 * wavelength / A_T 
    w_at_receiver = L * theta_divergence 

    print(f"beam width at receiver:: {w_at_receiver}")
    
    # Grid setup
    x = np.linspace(-A_R, A_R, resolution)
    y = np.linspace(-A_R, A_R, resolution)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)
    aperture_mask = (R_grid <= A_R/2)
    dA = (2*A_R / resolution)**2
    P_total_beam = (np.pi * w_at_receiver**2) / 2

    sigma_r = L * sigma

    # --- 2. SETUP TURBULENCE ---
    rytov_variance = 1.23 * Cn2 * (k**(7/6)) * (L**(11/6))
    
    scint_index = min(rytov_variance, 5.0) # Cap to prevent math explosion in extreme cases
    var_log = np.log(1 + scint_index) # Variance of the underlying normal distribution
    sigma_log = np.sqrt(var_log)
    mu_log = -var_log / 2

    efficiency_samples = []

    # --- 3. SIMULATION LOOP ---
    for _ in range(num_samples):
        # A. Pointing Error (Geometric Shift)
        dx = np.random.normal(0, sigma_r)
        dy = np.random.normal(0, sigma_r)

        # B. Turbulence (Intensity Fluctuation)
        if Cn2 > 0:
            turb_factor = np.random.lognormal(mu_log, sigma_log)
        else:
            turb_factor = 1.0

        # C. Calculate Power
        r2 = (X - dx)**2 + (Y - dy)**2
        I_geo = np.exp(-2 * r2 / w_at_receiver**2)
        
        # Apply Turbulence Factor to the Intensity Field
        I_total = I_geo * turb_factor
        
        collected_power = np.sum(I_total * aperture_mask * dA)
        eta = collected_power / P_total_beam
        
        efficiency_samples.append(eta)

    return np.mean(efficiency_samples)

def total_link_efficiency(A_tx, A_rx, Distance, jitter_angle, Cn2):
    
    # atm loss
    atm_transmission = _atmospheric_attenuation(
        distance_m=Distance,
        wavelength_m=const.lmda
    )

    # monte carlo
    geo_efficiency = monte_carlo_efficiency_with_turbulence(
        A_T=A_tx, 
        A_R=A_rx, 
        L=Distance, 
        sigma=jitter_angle, 
        resolution=100, 
        num_samples=500,
        Cn2=Cn2
    )

    # total
    total_eff = atm_transmission * geo_efficiency
    
    return total_eff, atm_transmission, geo_efficiency

# main #
if __name__ == "__main__":
    # TODO Logintudinal shrot mode for vbration
    # Parameters
    tx_diam = 0.102     # 10.2 cm
    rx_diam = 0.696     # 40.64 cm (16 in)
    link_range = 20000  # 20 km
    jitter = 5e-6       # 5 urad
    Cn2_l = 10e-17        # Light turbulance
    Cn2_m = 10e-14        # Medium turbulance
    Cn2_h = 10e-11        # Heavy turbulance

    total, atm, geo = 0, 0, 0

    # plot #
    min_dist = 500 # 0.5km
    max_dist = 50000 # 50km
    step = 500

    x_distances = np.arange(min_dist, max_dist, step)
    y_loss_5 = []
    y_loss_10 = []
    y_loss_15 = []
    y_loss_20 = []
    y_loss_25 = []
    y_loss = [y_loss_5, y_loss_10, y_loss_15, y_loss_20, y_loss_25]

    for i in range(5):
        for x in x_distances:
            total, atm, geo = total_link_efficiency(tx_diam, rx_diam, x, jitter, Cn2_h)
            y_loss[i].append(10*np.log10(total))
        jitter += 5e-6
        print(f"--- Simulation Results ---")
        print(f"Atmospheric Transmission: {atm:.4f} ({(atm)*100:.2f}%)")
        print(f"Geometric Efficiency:     {geo:.4f} ({(geo)*100:.2f}%)")
        print(f"TOTAL Link Efficiency:    {total:.6f}")
        print(f"Total Loss (dB):          {10*np.log10(total):.2f} dB")

    
    # Create the plot
    plt.plot(x_distances, y_loss_5, color='red', label='5 urad')
    plt.plot(x_distances, y_loss_10, color='yellow', label='10 urad')
    plt.plot(x_distances, y_loss_15, color='orange', label='15 urad')
    plt.plot(x_distances, y_loss_20, color='green', label='20 urad')
    plt.plot(x_distances, y_loss_25, color='blue', label='25 urad')
    plt.legend()

    # Add labels and title (optional)
    plt.xlabel("distance (m)")
    plt.ylabel("atenuation (db)")
    plt.title("loss due to jitter, light cirrus clouds, heavy turbulance")

    # Display the plot
    plt.show()