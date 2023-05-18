"""

Aircraft Simulation
=======================

This is the base class for the aircraft simulator.

Author: Alvaro Marcos Canedo

"""

from abc import ABC, abstractmethod

from Ares.aircraft_simulators.atmospheric_model import AtmosphericModel

from numpy import sin, cos, sqrt


class Aircraft(ABC):
    """

    This is the base class for the aircraft simulator

    """

    def __init__(self):
        """

        Constructor method.

        """

        self.model = None

        # Definition of constants
        self.g = 9.81
        self.R = 6378137.0
        self.f = 1 / 298.257222101
        self._b = self.R * (1 - self.f)
        self.e2 = 1 - (1 - self.f ** 2)

        # Instantiate atmospheric model
        self.atmosphere = AtmosphericModel()

    def earth_position(self, lat: float, lon: float, x: float, y: float, z: float) -> tuple:
        """

        Method to calculate the earth position of the aircraft.

        @param lat: Current latitude of the aircraft.
        @type lat: float
        @param lon: Current longitude of the aircraft.
        @type lon: float
        @param x: x Position of the aircraft.
        @type x: float
        @param y: y Position of the aircraft.
        @type y: float
        @param z: z Position of the aircraft.
        @type z: float

        @return: Latitude and longitude rates.
        @rtype: tuple
        """

        # Intermediate calculus
        v = self.R / sqrt((1 - self.e2 * sin(lat) ** 2))
        r = v * (1.0 - self.e2) / (1.0 - self.e2 * sin(lat) ** 2)

        # Calculation of latitude and longitude rates
        latp = x / (r + z)
        lonp = y / (cos(lat) * (self.R + z))

        return latp, lonp

    @abstractmethod
    def _sensors(self, *args):
        """

        Definition of sensors of the aircraft.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def _actuators(self, *args):
        """

        Definition of actuators of the aircraft.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def aerodynamic_model(self, *args):
        """

        Definition of the aerodynamic model oof the aircraft.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def power_plant_model(self, *args):
        """

        Definition of the power plant model oof the aircraft.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def equations(self, *args):
        """

        Definition of the equations of the model.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def pid(self, *args):
        """

        Definition of the PID-loop of the model.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def controller(self, *args):
        """

        Definition of the controller of the model.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def mass_model(self, *args):
        """

        Definition of the mass model of the aircraft.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def landing_gear(self, *args):
        """

        Definition of the landing gear model for the aircraft.


        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def gravity(self, *args):
        """

        Definition of the gravity model applicable to the aircraft.

        @param args: Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def forces(self, *args):
        """

        Definition of all the forces applicable to the aircraft.

        @param args: Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def calculate(self, *args):
        """

        Method to launch the calculation of the equations of the aircraft.

        @param args: Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def state_to_dict(self, *args):
        """

        Method to convert aircraft state to a dictionary

        @param args: Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')


__all__ = ["Aircraft"]
