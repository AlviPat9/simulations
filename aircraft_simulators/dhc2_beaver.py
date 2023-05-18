"""
DHC2 - Beaver
================

Flight Simulator environment for DHC2 - beaver aircraft.

Author: Alvaro Marcos Canedo
"""

from Ares.simulators.aircraft import Aircraft
from Ares.integrators.forward_euler import ForwardEuler
from Ares.utilities.keys import AircraftKeys as Ak

import numpy as np
from scipy.signal import dlti
import json


class DHC2Beaver(Aircraft):
    """

    Definition of the model for the DHC2 - Beaver.

    """

    def __init__(self, initial_state: np.ndarray, step_size=0.3, final_time=1000):
        """

        Constructor method

        @param initial_state: Initial state of the model.
        @type initial_state: np.ndarray
        @param step_size: Step size for the calculation.
        @type step_size: float, optional
        @param final_time: Final time for the simulation.
        @type final_time: float, optional

        """
        # Call superclass
        super().__init__()

        # Definition of geometry of the aircraft
        self.c = 1.5875  # Mean aerodynamic Chord (MAC) [m]
        self.b = 14.63  # Wing span [m]
        self.S = 23.23  # Wing area [m^2]

        # Landing gear data definition
        self.damp = 150  # Damping coefficient for the landing gear [Ns/m]
        self.k = 2000  # Stiffness coefficient for the landing gear [N/m]

        # Fuel consumption
        self.fc = 76 / 3600  # [kg/s]

        # Path to the aerodynamic coefficients
        path = r'C:\ProgramData\Calculos\python\Ares\aircraft_simulators\dhc2_beaver_aero.json'

        # Load aerodynamic data of the aircraft
        with open(path, 'r') as f:
            self.aero = json.load(f)

        # Call integrator
        self.integration = ForwardEuler(system=self.equations(), step_size=step_size, final_time=final_time,
                                        initial_state=initial_state)

        # Definition of the mass model
        self.mass = self.mass_model()

        # Definition of sensors for the aircraft
        self.sensors = self._sensors(step_size)

        # Definition of actuators for the aircraft
        self.actuators = self._actuators(step_size)

    def _sensors(self, step_size: float) -> dict:
        """

        Definition of the transfer functions of the actuators.

        @param step_size: Step size for the integration model.
        @type step_size: float

        @return: Transfer functions associated to the sensors of the aircraft.
        @rtype: dict

        """

        # 2nd order Pade approximation

        # Time delay of 0.1s
        anemometer = dlti([0.1 ** 2 / 12.0, -0.05, 1], [0.1 ** 2 / 12.0, 0.05, 1], dt=step_size)

        # Time delay of 0.06s
        inertial = dlti([0.0003, -0.03, 1], [0.0003, 0.03, 1], dt=step_size)

        return {Ak.anemometer: anemometer, Ak.inertial: inertial}

    def _actuators(self, step_size: float) -> dict:
        """

        Definition of the actuators model for the aircraft.

        @param step_size: Step size for the integration model.
        @type step_size: float

        @return: Transfer functions associated to the different actuators of the aircraft.
        @rtype: dict

        """

        # Low-pass filter model

        # Time constant of 0.1s
        aerodynamic = dlti([1], [0.1, 1], dt=step_size)

        # Time constant of 5s
        throttle_lever = dlti([1], [5, 1], dt=step_size)

        return {Ak.aerodynamic: aerodynamic, Ak.T_lever: throttle_lever}

    def aerodynamic_model(self, n: float, angles: dict, angular_speed: dict, airspeed: float, delta: dict,
                          accelerations: dict, fuel: float) -> np.ndarray:
        """
        Aerodynamic model of the DHC2 - Beaver.

        @param n: RPM of the engine.
        @type n: float
        @param angles: Necessary angles for determining the movement of the aircraft.
        @type angles: dict
        @param angular_speed: Angular speed of the aircraft.
        @type angular_speed: dict
        @param airspeed: Airspeed of the aircraft.
        @type airspeed: float
        @param delta: Control surface deflection.
        @type delta: dict
        @param accelerations: Accelerations needed for the aerodynamic model of the DHC.
        @type accelerations: dict
        @param fuel: Fuel available in the Tank of the aircraft.
        @type fuel: float

        @return: Aerodynamic model of the aircraft. For the DHC2, the aerodynamic model includes the power plant.
        @rtype: np.ndarray

        """

        power_plant = self.power_plant_model(n, airspeed, fuel)

        Cx = self.aero['CX_0'] + self.aero['CX_apt'] * power_plant + self.aero['CX_apt2_a'] * angles[
            Ak.alpha] * power_plant ** 2 + \
             self.aero['CX_a'] * angles[Ak.alpha] + self.aero['CX_a2'] * angles[Ak.alpha] ** 2 + \
             self.aero['CX_a3'] * angles[Ak.alpha] ** 3 + self.aero['CX_q'] * angular_speed[
                 Ak.q] * self.c / airspeed + \
             self.aero['CX_dr'] * delta[Ak.dr] + self.aero['CX_df'] * delta[Ak.df] + self.aero['CX_df_a'] * delta[
                 Ak.df] * \
             angles[Ak.alpha]

        Cy = self.aero['CY_0'] + self.aero['CY_b'] * angles[Ak.beta] + self.aero['CY_p'] * angular_speed[
            Ak.p] * self.b / (2 * airspeed) + \
             self.aero['CY_r'] * angular_speed[Ak.r] * self.b / (2 * airspeed) + self.aero['CY_da'] * delta[Ak.da] + \
             self.aero['CY_dr'] * delta[Ak.dr] + self.aero['CY_dr_a'] * delta[Ak.dr] * angles[Ak.alpha] + \
             self.aero['CY_bp'] * accelerations['bp'] * self.b / (airspeed * 2)

        Cz = self.aero['CZ_0'] + self.aero['CZ_apt'] * power_plant + self.aero['CZ_a'] * angles[Ak.alpha] + \
             self.aero['CZ_a3'] * angles[Ak.alpha] ** 3 + self.aero['CZ_q'] + angular_speed[
                 Ak.q] * self.c / airspeed + \
             self.aero['CZ_de'] * delta[Ak.de] + self.aero['CZ_de_b2'] * delta[Ak.de] * angles[Ak.beta] ** 2 + \
             self.aero['CZ_df'] * delta[Ak.df] + self.aero['CZ_df_a'] * delta[Ak.df] * angles[Ak.alpha]

        Cl = self.aero['Cl_0'] + self.aero['Cl_b'] * angles[Ak.beta] + \
             self.aero['Cl_p'] * angular_speed[Ak.p] * self.b / (2 * airspeed) + \
             self.aero['Cl_r'] * self.b / (2 * airspeed) + self.aero['Cl_da'] * delta[Ak.da] + \
             self.aero['Cl_dr'] * delta[Ak.dr] + self.aero['Cl_a2_apt'] * power_plant * angles[Ak.alpha] ** 2 + \
             self.aero[
                 'Cl_da_a'] * angles[Ak.alpha] * \
             delta[Ak.da]

        Cm = self.aero['Cm_0'] + self.aero['Cm_apt'] * power_plant + self.aero['Cm_a'] * angles[Ak.alpha] + \
             self.aero['Cm_a2'] * angles[Ak.alpha] ** 2 + self.aero['Cm_q'] * angular_speed[
                 Ak.q] * self.c / airspeed + \
             self.aero['Cm_de'] * delta[Ak.de] + self.aero['Cm_b2'] * angles[Ak.beta] ** 2 + \
             self.aero['Cm_r'] * angular_speed[Ak.r] * self.b / (2 * airspeed) + self.aero['Cm_df'] * delta[Ak.df]

        Cn = self.aero['Cn_0'] + self.aero['Cn_b'] * angles[Ak.beta] + + self.aero['Cn_b3'] * angles[Ak.beta] ** 3 + \
             self.aero['Cn_p'] * angular_speed[Ak.p] * self.b / (2 * airspeed) + self.aero['Cn_da'] * delta[Ak.da] + \
             self.aero['Cn_dr'] * delta[Ak.dr] + self.aero['Cn_apt3'] * power_plant ** 3 + self.aero[
                 'Cn_q'] * self.c / airspeed
        #
        # Cx = np.sin(angles[Ak.alpha]) * Cz_b + np.cos(angles[Ak.alpha]) * Cx_b
        # Cz = - np.sin(angles[Ak.alpha]) * Cz_b - np.cos(angles[Ak.alpha]) * Cx_b

        return 0.5 * self.atmosphere.rho * airspeed ** 2 * self.S * np.array([[Cx, Cy, Cz],
                                                                              [self.b * Cl, self.c * Cm, self.b * Cn]])

    def power_plant_model(self, n: float, airspeed: float, fuel: float) -> float:
        """

        Definition of the engine model for the DHC2 - Beaver.

        @param n: rpm of the engine.
        @type n: float
        @param airspeed: Airspeed of the aircraft.
        @type airspeed: float
        @param fuel: Fuel available in the tank.
        @type: float

        @return: Engine value modeled to be included in aerodynamic model.
        @rtype: float

        """

        # Definition of constants
        a = 0.08696
        b = 191.18
        Pz = 20  # Manifold pressure -> At the moment defines as constant.

        if fuel > 0.0:
            # Calculation of the engine power -> Last number is the conversion between horsepower and wats
            P = (-326.5 + (0.00412 * (Pz + 7.4) * (n + 2010) + (408.0 - 0.0965 * n) *
                           (1.0 - self.atmosphere.rho / self.atmosphere.rho_sl))) * 0.74570

            # Normalization to be included in aerodynamic model
            apt = a + b * P / (0.5 * self.atmosphere.rho * airspeed ** 3)
        else:
            apt = 0.0

        return apt

    def mass_model(self) -> dict:
        """

        Mass model for the DHC2 - Beaver

        @return: Mass model of the DHC2 - Beaver.
        @rtype: dict

        """

        # Definition of the mass model for the aircraft
        return {Ak.Ix: 5368.39, Ak.Iy: 6928.93, Ak.Iz: 11158.75, Ak.Ixz: 117.64, Ak.Ixy: 0.0, Ak.Iyz: 0.0,
                Ak.mass: 2288.231, Ak.cg: np.array([0.5996, 0.0, -0.8851])}

    def gravity(self, euler_angles: dict, mass: float) -> np.ndarray:
        """

        Gravity model for the DHC2 - Beaver

        @param euler_angles: Euler angles of the movement.
        @type euler_angles: dict
        @param mass: Mass of the aircraft.
        @type mass: float

        @return: Gravity applied to the aircraft.
        @rtype: np.ndarray
        """

        return mass * self.g * np.array([np.sin(euler_angles[Ak.theta]),
                                         np.sin(euler_angles[Ak.phi]) * np.cos(euler_angles[Ak.theta]),
                                         np.cos(euler_angles[Ak.phi]) * np.cos(euler_angles[Ak.theta])])

    def forces(self, n: float, angles: dict, angular_speed: dict, airspeed: float, delta: dict, accelerations: dict,
               euler_angles: dict, mass: float, y: float, brake_pedal: float, fuel: float) -> tuple:
        """

        Forces wrapper for the DHC2 - Beaver.

        @param n: RPM of the engine.
        @type n: float
        @param angles: Necessary angles for determining the movement of the aircraft.
        @type angles: dict
        @param angular_speed: Angular speed of the aircraft.
        @type angular_speed: dict
        @param airspeed: Airspeed of the aircraft.
        @type airspeed: float
        @param delta: Control surface deflection.
        @type delta: dict
        @param accelerations: Accelerations needed for the aerodynamic model of the DHC.
        @type accelerations: dict
        @param euler_angles: Euler angles of the movement.
        @type euler_angles: dict
        @param mass: Mass of the aircraft.
        @type mass: float
        @param y: Displacement of the landing gear.
        @type y: float
        @param brake_pedal: Pedal brake input from the pilot.
        @type brake_pedal: float
        @param fuel: Fuel Available in the tank of the aircraft.
        @type fuel: float

        @return: Total forces (and torques) applied to the aircraft.
        @rtype: tuple
        """

        # Aerodynamic forces -> For the DHC2 - Beaver the engine is included in the aerodynamic forces
        aero = self.aerodynamic_model(n, angles, angular_speed, airspeed, delta, accelerations, fuel)

        # Gravity forces
        gravity = self.gravity(euler_angles, mass)

        # Landing gear forces -> At the moment not included
        lg_force = self.landing_gear(y, aero[2] + mass * self.g, brake_pedal)

        aero[0][0] -= lg_force

        return aero[0] + gravity, aero[1]

    def landing_gear(self, y: float, normal_force: float, brake_pedal: float) -> float:
        """

        Landing gear model for the DHC2 - Beaver.

        @param y: Displacement of the landing gear
        @type y: float
        @param normal_force: Normal forces that apply to the landing gear.
        @type normal_force: float
        @param brake_pedal: Pedal brake input from the pilot.
        @type brake_pedal: float

        @return: Force opposite to direction of the movement
        @rtype: float
        """

        if y >= 0:

            # Calculate Friction Force
            friction_coefficient = 0.3
            F_friction = friction_coefficient * normal_force

            # Calculate Braking forces
            braking_coefficient = 0.2
            F_brake = brake_pedal * normal_force * braking_coefficient

            return F_friction + F_brake

        else:
            return 0.0

    def pid(self, *args):
        pass

    def controller(self, *args):
        pass

    def calculate(self):
        """

        Method to launch the calculation of the equations for the DHC2 - Beaver.

        @return:
        """
        # Inputs of the aircraft model
        delta = {Ak.de: 0.0, Ak.dr: 0.0, Ak.da: 0.0, Ak.df: 0.0, Ak.brake: 0.0, Ak.t_lever: 1800.0}

        # beta derivative
        acceleration = 0.0

        # Get state of the aircraft
        aircraft_state = self.integration.get_state()

        while self.integration.time < 1000:
            self.atmosphere.calculate(aircraft_state[-1])
            aircraft_state = np.concatenate(aircraft_state,
                                            self.integration.integrate_step(self.integration.get_state(), delta,
                                                                            acceleration))

        pass

    def step(self, delta) -> dict:
        """

        Step Method for the Simulator to fit the flight environment for the reinforcement learning model.

        @param delta: Actions done to the control surfaces of the aircraft.
        @type delta: dict

        @return: Information of the current state of the aircraft.
        @rtype: dict
        """
        # Set acceleration to 0 -> not calculated
        acceleration = 0.0

        # Get aircraft state
        aircraft_state = self.integration.get_state()

        self.atmosphere.calculate(aircraft_state[11])

        self.integration.integrate_step(self.integration.get_state(), delta, acceleration)

        intermediate_dict = self.state_to_dict(self.integration.get_state())

        output_vars = [Ak.speed, Ak.position, Ak.euler_angles, Ak.fuel]

        return {key: intermediate_dict[key] for key in intermediate_dict if key in output_vars}

    def reset(self) -> np.ndarray:
        """

        Reset method for the simulation.

        @return: Initial state of the simulation.
        @rtype: np.ndarray
        """

        return self.integration.reset()

    def equations(self, *args) -> np.ndarray:
        """

        Equations of the aircraft model developed for the DHC2 - Beaver.

        @param args:

        @return: Actual state of the aircraft based on its derivatives.
        @rtype: np.ndarray
        """

        # Angular speeds
        p = args[1][0]
        q = args[1][1]
        r = args[1][2]

        # Euler angles
        phi = args[1][3]
        theta = args[1][4]
        psi = args[1][5]

        # Position
        x = args[1][6]
        y = args[1][7]
        z = args[1][8]

        # Speed
        u = args[1][9]
        v = args[1][10]
        w = args[1][11]

        # Landing gear data
        y_lg = args[1][12]
        v_lg = args[1][13]

        # Fuel available
        fuel = args[1][14]

        # Latitude and longitude
        lat = args[1][15]
        lon = args[1][16]

        # Delta
        delta = args[2]

        # Accelerations
        accelerations = args[3]

        # Engine
        n = delta[Ak.t_lever]

        # Calculate angle of attack and angle of sideslip
        angles = {Ak.alpha: np.Aktan2(w, u), Ak.beta: np.Aktan2(v, (u ** 2 + w ** 2) ** 0.5)}

        # Forces wrapper
        forces, torques = self.forces(n=n, angles=angles, angular_speed={Ak.p: p, Ak.q: q, Ak.r: r},
                                      airspeed=np.linalg.norm([u, v, w]), delta=delta,
                                      euler_angles={Ak.phi: phi, Ak.theta: theta, Ak.psi: psi},
                                      accelerations=accelerations,
                                      mass=self.mass[Ak.mass], y=y_lg, brake_pedal=delta[Ak.brake], fuel=fuel)

        # Derivatives of angular speed in body-frame
        pp = (self.mass[Ak.Iz] * torques[0] + self.mass[Ak.Ixz] * torques[2] - q * r * self.mass[
            Ak.Ixz] ** 2 - q * r *
              self.mass[
                  Ak.Iz] ** 2
              + self.mass[Ak.Ixz] * self.mass[Ak.Iy] * p * q + self.mass[Ak.Ixz] * self.mass[Ak.Iz] * p * q +
              self.mass[Ak.Iz] * self.mass[Ak.Iy] * q * r) / (
                     self.mass[Ak.Ix] * self.mass[Ak.Iz] - self.mass[Ak.Ixz] ** 2)
        qp = (torques[1] - self.mass[Ak.Ixz] * p ** 2 + self.mass[Ak.Ixz] * r ** 2 - self.mass[Ak.Ix] * p * r +
              self.mass[Ak.Iz] * p * r) / self.mass[Ak.Iy]

        rp = (self.mass[Ak.Ixz] * torques[0] + self.mass[Ak.Ix] * torques[2] + p * q * self.mass[Ak.Ix] ** 2 -
              self.mass[Ak.Ix] * self.mass[Ak.Iy] * p * q - p * q * self.mass[Ak.Ixz] ** 2 -
              self.mass[Ak.Ix] * self.mass[Ak.Ixz] * q * r + self.mass[Ak.Ixz] * self.mass[Ak.Iy] * q * r -
              self.mass[
                  Ak.Ixz] * self.mass[Ak.Iz] * q * r) / (
                     self.mass[Ak.Ix] * self.mass[Ak.Iz] * self.mass[Ak.Ixz] ** 2)

        # Derivatives of euler angles
        phip = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        thetap = q * np.cos(phi) - r * np.sin(phi)
        psip = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)

        # Airspeed derivative
        up = forces[0] / self.mass[Ak.mass] - q * w + r * v
        vp = forces[1] / self.mass[Ak.mass] - r * u + p * w
        wp = forces[2] / self.mass[Ak.mass] - p * v + q * u

        # Get rotation matrix
        rotation = self.rotation_matrix(np.array([phi, theta, psi]))

        # Position Derivatives
        xp, yp, zp = rotation.dot(np.array([x, y, z]))

        # Landing gear dynamics
        yp_lg = v_lg
        vp_lg = (forces[2] - self.damp * v_lg - self.k * y_lg) / self.mass[Ak.mass]

        # Fuel consumption
        fuelp = - self.fc * n / 2400

        # Latitude and longitude derivatives
        latp, lonp = self.earth_position(lat, lon, x, y, z)

        return np.array([pp, qp, rp, phip, thetap, psip, up, vp, wp, xp, yp, zp, yp_lg, vp_lg, fuelp, latp, lonp])

    def state_to_dict(self, state):
        """

        Method to convert the np.ndarray object to a dict with variable names.

        @param state: Current state of the aircraft.
        @type state: np.ndarray

        @return: Dictionary of the current state of the aircraft.
        @rtype: dict
        """

        return {Ak.speed_ang: state[0:3], Ak.euler_angles: state[3:6], Ak.speed: state[7:10],
                Ak.earth_position: state[10:13], Ak.fuel: state[14], Ak.position: state[15:]}

    @staticmethod
    def rotation_matrix(angles):
        """

        Rotation matrix.

        @param angles: angles to compute the rotation (X - Y - Z).
        @type angles: np.ndarray

        @return: Rotation matrix for the angles defined in the input.
        @rtype: np.ndarray

        """

        return np.array([[np.cos(angles[1]) * np.cos(angles[2]),
                          np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2]) - np.cos(angles[0]) * np.sin(
                              angles[2]),
                          np.cos(angles[0]) * np.sin(angles[1]) * np.cos(angles[2]) + np.sin(angles[0]) * np.sin(
                              angles[2])],
                         [np.cos(angles[1]) * np.sin(angles[2]),
                          np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) + np.cos(angles[0]) * np.cos(
                              angles[2]),
                          np.cos(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) - np.sin(angles[0]) * np.cos(
                              angles[2])],
                         [-np.sin(angles[1]), np.sin(angles[0]) * np.cos(angles[1]),
                          np.cos(angles[0]) * np.cos(angles[1])]
                         ])


__all__ = ["DHC2Beaver"]
