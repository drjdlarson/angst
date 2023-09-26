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
import utils

# Constants
r2d = 180.0 / np.pi
d2r = 1 / r2d
mph2fps = 1.46667
fps2mph = 1 / mph2fps
m2feet = 3.28084

# Earth parameters
Re_bar = 6371000 * m2feet
const_density = 2.3769e-3
const_gravity_m = 32.17


class InitialConditions:
    def __init__(self, ICs):
        for ic in ICs.keys():
            self.__setattr__(ic, ICs.get(ic))


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


class FixedWingVehicle:
    """Implements the base vehicle.
        The base vehicle is a fixed-wing aircraft defined using the
            :class:`FixedWingVehicle.AircraftParams'

    Attributes
    ----------
    state : numpy array
        State of the aircraft.
    params : :class:`.AircraftParams`
        Parameters of the aircraft.
    launched : bool
        Flag indicating if the vehicle has been launched yet.
    """
    units = 'Imperial'
    angles = 'Radians'

    def __init__(self, params, guidanceTFs, launched=False, ICs=None):
        """Initalize a Fixed Wing Vehicle object.
        
        Parameters
        ----------
        params : :dict:`Dict of all applicable aircraft parameters`
            Parameters of the aircraft.
        guidanceTFs : :dict:`Dict of applicable PI guidance transfer function constants`
            Parameters of the PI Guidance Transfer Functions.
        launched : :bool:`Launched boolean`
            Boolean to set the status of the aircraft as launched.
        ICs : :dict:`Initial conditions`
            Initial conditions of aircraft state.
        """
        self.AircraftParams = AircraftParams(params)
        self.GuidanceTransferFunctions = GuidanceTransferFunctions(guidanceTFs)
        self.launched = launched
        self.UserCommand = [np.nan, np.nan, np.nan]  # Velocity, h_dot (rate of climb), psi (heading)
        self.UserCommandHistory = self.UserCommand
        if ICs is not None:
            self.setInitialConditions(ICs)


    def setInitialConditions(self, ICs):
        self.InitialConditions = InitialConditions(ICs)


    def userCommandTrajectory(self, velocity, rate_of_climb, heading):
        # Set the new user command
        self.UserCommand = [velocity, rate_of_climb, heading]
        # Add the new user command to history
        self.UserCommandHistory = np.vstack((self.UserCommandHistory, self.UserCommand))


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

    init_cond = {'v_BN_W': 400 * mph2fps,
                 'h_o': 0,
                 'gamma': 0,
                 'sigma': 0,
                 'lat': 33.2098 * d2r,
                 'lon': -87.5692 * d2r,
                 'v_WN_N': [25 * mph2fps, 25 * mph2fps, 0],
                 'weight': 300000}
    
    my_C130.setInitialConditions(init_cond)
    my_C130.userCommandTrajectory(450 * mph2fps, 5 * d2r, 15 * d2r)

    print(my_C130.UserCommandHistory)