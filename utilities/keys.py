"""

Keys
=======
These classes define the different keys used for all the simulators. As simulators are developed, their keys will be
added to this file via classes.

Author: Alvaro Marcos Canedo
"""


class AircraftKeys:
    """

    This class defines the names of the variables used in aircraft simulator so it is easier to change the name of one
    variable

    """

    # Angles
    euler_angles = "euler_angles"
    phi = "phi"
    theta = "theta"
    psi = "psi"

    alpha = "alpha"
    beta = "beta"

    # Speeds
    # Linear speeds
    speed = "speed"
    u = "u"
    v = "v"
    w = "w"

    # Angular speed
    speed_ang = "speed_ang"
    p = "p"
    q = "q"
    r = "r"

    # Control surfaces
    de = "de"
    da = "da"
    dr = "dr"
    df = "df"
    brake = "brake"
    t_lever = "n"

    # Mass properties
    Ix = "Ix"
    Iy = "Iy"
    Iz = "Iz"
    Ixz = "Ixz"
    Iyz = "Iyz"
    Ixy = "Ixy"
    cg = "cg"
    mass = "m"

    # Position
    position = "position"
    lat = "latitude"
    lon = "longitude"
    earth_position = "earth_position"

    # Others
    fuel = "fuel"
    anemometer = "anemometer"
    inertial = "inertial"
    aerodynamic = "aerodynamic"
    T_lever = "Throttle_lever"



__all__ = ["AircraftKeys"]
