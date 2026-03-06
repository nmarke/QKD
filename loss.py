import numpy as np
import constants as const

def _tx_gain(theta: float) -> float:
    """
    returns the transmitter gain as a function of the full 
    transmitting divergence angle, theta

    note, theta is probably a functon of tx diameter
    """
    return 16 / theta**2

def _rx_gain(rxd: float) -> float:
    """
    returns the receiver gain as a function of telescope 
    diameter (mm) and operating wavelength (nm)
    """
    return (rxd*np.pi/const.lmda)**2

def _tx_pointing_loss(theta: float) -> float:
    """
    returns the transmitter pointing loss as a
    function of pointing error, theta (radians)

    pointing error could be due to jitter, etc
    """
    return np.exp(-1*_tx_gain*(theta**2))

def _rx_poining_loss(theta: float) -> float:
    """
    returns the receiver pointing loss as a
    function of receiver pointing error, theta (radians)

    pointing error could be due to jitter, etc
    """
    return np.exp(-_rx_gain*(theta**2))

def _slant_distance(Re: float, hs: float, he: float, theta) -> float:
    R = Re + he
    H = hs - he

    return R * (np.sqrt(
        ((R + H) / R)**2 - (np.cos(theta))**2
    ) - np.sin(theta))

def _fs_path_loss() -> float:
    """
    free space path loss due to slant distance

    always returns one because slant does not play a factor
    in signal loss for fso within the atmosphere as there is
    never a point during which the beam ravels through a
    vacuum
    """
    return 1.0

def _mie_scattering(he: float, theta: float) -> float:
    """
    models effects of mie scattering effect

    Mie scattering occurs when the diameter of atmospheric particles 
    is equal to or greater than the wavelength of the optical beam

    rho denoes extincion ratio
    """
    a = -0.000545*const.lmda**2 + 0.002*const.lmda - 0.0038
    b = 0.00628*const.lmda**2 - 0.0232*const.lmda + 0.00439
    c = -0.028*const.lmda**2 + 0.101*const.lmda = 0.18
    d = -0.228*const.lmda**3 + 0.922*const.lmda**2 - 1.26*const.lmda + 0.719

    rho = a*he**3 + b*he**2 + c*he + d

    return np.exp(-rho/np.sin(theta))

def _geometrical_scattering(phi: float) -> float:
    """
    models effecrs of geometric scattering

    Geometrical scattering is used to model the attenuation due 
    to atmosphere that is close to the surface of the Earth and 
    is caused by fog or dense clouds.

    V is visibility in km
    N is cloud number concentration
    da is distance beam travels through troposphere
    theta is attenuation coefficient
    """

    v = 1.002/(const.Lw*N)**0.6473

    theta = (3.91/v)*(const.lmda/550)**(-1*phi)

    da = _slant_distance(
        1,
        1,
        1,
        1
    )

    return np.exp(-1*theta*da)

def _atmospheric_loss():
    """
    atmospheric attenuation loss due to geomeric and mie
    """
    Im = _mie_scattering(
        1,
        1,
        1
    )

    Ig = _geometrical_scattering(
        1,
        1
    )

    return Im * Ig

def loss(Atx: float, Arx: float, L: float):

    theta_max = 1.22*const.const.lmda/Arx        #Calculates maximum angle of dispersion from transmitter
    w0 = L*theta_max                       #Calculates width of beam

    P_total = np.pi*w0**2/2

