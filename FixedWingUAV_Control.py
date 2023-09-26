#!/usr/bin/env python
""" Non-Linear Fixed-Wing UAV Control
        (Based on work from Dr. John Schierman)
        This code will allow one to model a rigid aircraft operating in steady winds.
        The aircraft will be guided via nonlinear feedback laws to follow profiles:
            - Commanded velocities
            - Commanded rates of climb/descent
            - Commanded headings
        The attitude dynamics of the vehicle will be approximated.
"""
__author__ = "Alex Springer"
__version__ = "1.0.0"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Production"

import numpy as np
from scipy import integrate
import pathlib
import wgs84

# Constants
r2d = 180.0 / np.pi
d2r = 1 / r2d
mph2fps = 1.46667
fps2mph = 1 / mph2fps
m2feet = 3.28084

# Earth parameters
Re_bar = 6371000 * m2feet
const_density = 2.3769e-3
const_gravit = 32.17


class FixedWingVehicle:
    """Implements the base vehicle.

    Attributes
    ----------
    state : numpy array
        State of the aircraft.
    params : :class:`.AircraftParams`
        Parameters of the aircraft.
    launched : bool
        Flag indicating if the vehicle has been launched yet.
    """

    class AircraftParams:
        def __init__(self, params=None):
            self.weight_max = np.nan
            self.weight_min = np.nan
            self.speed_max = np.nan
            self.speed_min = np.nan
            self.Kf = np.nan
            self.omega_T = np.nan
            self.omega_L = np.nan
            self.omega_mu = np.nan
            self.T_max = np.nan
            self.K_Lmax = np.nan
            self.mu_max = np.nan
            self.C_Do = np.nan
            self.C_Lalpha = np.nan
            self.alpha_o = np.nan
            self.wing_area = np.nan
            self.aspect_ratio = np.nan
            self.wing_eff = np.nan
            for param in params.keys():
                self.__setattr__(param, params.get(param))

    class GuidanceTransferFunctions:
        def __init__(self, transfer_functions):
            for tf in transfer_functions.keys():
                self.__setattr__(tf, transfer_functions.get(tf))

    def __init__(self, params, guidanceTFs, launched=False):
        """Initalize a Fixed Wing Vehicle object.
        
        Parameters
        ----------
        params : :dict:`Dict of all applicable aircraft parameters`
            Parameters of the aircraft.
        """
        self.params = self.AircraftParams(params)
        self.guidanceTFs = self.GuidanceTransferFunctions(guidanceTFs)
        self.launched = launched


if __name__ == '__main__':
    print('Hello world!')

    # Define the aircraft (example aircraft is C-130)
    new_aircraft_parameters = {'weight_max': 327000,
                               'weight_min': 157000,
                               'speed_max': 600 * mph2fps,
                               'speed_min': 200 * mph2fps,
                               'Kf': 4e-6,
                               'omega_T': 2,
                               'omega_L': 2.5,
                               'omega_mu': 1,
                               'T_max': 72000,
                               'K_Lmax': 2.6,
                               'mu_max': 30 * d2r,
                               'C_Do': 0.0183,
                               'C_Lalpha': 0.0920 / d2r,
                               'alpha_o': -0.05 * d2r,
                               'wing_area': 1745,
                               'aspect_ratio': 10.1,
                               'wing_eff': 0.613}

    PI_guidance_TFs = {'K_Tp': 0.08,
                    'K_Ti': 0.002,
                    'K_Lp': 0.5,
                    'K_Li': 0.01,
                    'K_mu_p': 0.075}
    
    my_C130 = FixedWingVehicle(new_aircraft_parameters, PI_guidance_TFs)