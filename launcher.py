from simulations.aircraft_simulators.dhc2_beaver import DHC2Beaver

from simulations.utilities.keys import AircraftKeys as Ak

import numpy as np

initial_state = np.array([0, 0, 0,
                          0, 0.1919, 0,
                          34.2165, -0.720949, 7.32877,
                          0, 0, 609.6,
                          -1, -1,
                          500,
                          0, 0])


aircraft = DHC2Beaver(initial_state, step_size=0.1)

delta = {Ak.de: -0.093, Ak.da: 0.009624, Ak.dr: -0.0495, Ak.df: 0, Ak.brake: 0, Ak.t_lever: 1800}

# integrate
for i in range(100):
    aircraft.step(delta)