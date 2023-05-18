"""

Atmospheric model
=================

Atmospheric model for the flight simulator.

"""

from numpy import exp


class AtmosphericModel(object):
    """

    Atmospheric model for the flight simulator

    """

    def __init__(self):
        """

        Constructor method.

        """

        # Definition of sea-level atmospheric conditions
        self.rho_sl = 1.225  # [kg/m^3]
        self.T_sl = 288.15  # [K]
        self.P_sl = 101325.0  # [Pa]
        self._gamma = 1.4  # [-]
        self._R = 287.053  # [J/(kgK)
        self.rho = None
        self.T = None
        self.P = None
        self.a = None

    def calculate(self, h):
        """

        Method to calculate the atmospheric conditions at certain altitude.

        @param h: altitude [m] to compute atmospheric conditions.
        @type h: float

        """

        if h < 11000.0:
            # Atmospheric conditions at Troposphere
            self.T = self.T_sl - 6.5 * h / 1000
            self.P = self.P_sl * (self.T_sl / (self.T_sl - 6.5 * h / 1000)) ** (34.1632 / -6.5)

        elif h < 20000.0:
            # Atmospheric conditions at low Stratosphere
            self.T = 216.65
            self.P = 22632.06 * exp(-34.1632 * (h - 11000) / (1000 * self.T))

        elif h < 32000.0:
            # Atmospheric conditions at mid-Stratosphere
            self.T = 196.65 + h / 1000
            self.P = 5474.889 * (216.65 / (216.65 + (h - 20000) / 1000)) ** 34.1632

        elif h < 47000.0:
            # Atmospheric conditions at high Stratosphere
            self.T = 139.95 + 2.8 * h / 1000
            self.P = 868.0187 * (228.65 / (228.65 + 2.8 * (h - 32000) / 1000)) ** (34.1632 / 2.8)

        elif h < 51000.0:
            # Atmospheric conditions at low Mesosphere
            self.T = 270.65
            self.P = 110.9063 * exp(-34.1632 * (h - 47000) / (1000 * 270.65))

        elif h < 71000.0:
            # Atmospheric conditions at mid-Mesosphere
            self.T = 413.45 - 2.8 * h / 1000
            self.P = 66.93887 * (270.65 / (270.65 - 2.8 * (h - 51000) / 1000)) ** (34.1632 / -2.8)

        elif h < 84852.0:
            # Atmospheric conditions at high Mesosphere
            self.T = 365.65 - 2.0 * h / 1000
            self.P = 3.95642 * (214.65 / (214.65-2 * (h - 71000) / 1000)) ** (34.1632 / -2.0)

        self.rho = self.P / (self._R * self.T)
        self.a = (self._gamma * self._R * self.T) ** 0.5


__all__ = ["AtmosphericModel"]
