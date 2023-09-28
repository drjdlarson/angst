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
from scipy import solve_ivp
import pandas as pd
import pathlib
import wgs84
import utils
import control

# Constants
r2d = 180.0 / np.pi
d2r = 1 / r2d
mph2fps = 1.46667
fps2mph = 1 / mph2fps
m2feet = 3.28084

# Earth parameters
Re_bar = 6371000 * m2feet
const_density = 2.3769e-3
const_gravity = 32.17


class InitialConditions:
    # This is most likely pointless and should become OBE.
    def __init__(self, ICs):
        for ic in ICs.keys():
            self.__setattr__(ic, ICs.get(ic))


class AircraftParams:
    """ Class AircraftParams
    This class allows the user to define the physical model of an aircraft to be used
        as an object of the FixedWingVehicle class.
    The default aircraft parameters, set to np.nan upon unitilization unless included in the params dictionary, are:
        weight_max, weight_min, speed_max, speed_min, Kf, omega_T, omega_L, omega_mu, T_max, K_Lmax, mu_max,
        C_Do, C_Lalpha, alpha_o, wing_area, aspect_ratio, wing_eff
    Other parameters may be defined as required either as key/value pairs in the params input to the initialization,
        or by using the .setAircraftParameters() function, passing the new parameter(s) as a key/value pair(s) in a dict.
    """
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
        self.setAircraftParameters(params)

    def setAircraftParameters(self, params):
        """ Add or update Parameters of AircraftParams object

        Parameters
        ----------
        params : :dict:`Dict of aircraft parameter(s) to be added/changed`
            Example: {'weight_max': 120, 'max_thrust': 100, 'min_thrust': -20}
                > In this example, weight_max, an existing parameter, is updated to 120.
                > max_thrust and min_thrust, parameters which don't yet exist, are added.
        """
        for param in params.keys():
            self.__setattr__(param, params.get(param))


class FixedWingVehicle:
    """Implements the base vehicle.
    """
    units = 'Imperial'
    angles = 'Radians'

    def __init__(self, params, ICs=None):
        """Initalize a Fixed Wing Vehicle object.

        Parameters
        ----------
        params : :dict:`Dict of all applicable aircraft parameters`
            Parameters of the aircraft.
        ICs : :dict:`Initial conditions`
            Initial conditions of aircraft state. If not provided, the object will be initialized
                with empty arrays for all state variables.
            Allowable keys are: m, v_BN_W, gamma, sigma, lat, lon, h, airspeed, alpha, drag, and time
                (note that time for the initial conditions is the start time of the vehicle's state, and
                differs from the dt parameter of the FixedWingVehicle.updateState function) 
        """
        self.AircraftParams = AircraftParams(params)
        self.wind = self.state.v_WN_N

        # Initialize vehicle state variables
        self.m = []
        self.weight = []
        self.v_BN_W = []
        self.v_WN_N = []  # Added for future time-dependent winds
        self.gamma = []
        self.sigma = []
        self.lat = []
        self.lon = []
        self.h = []
        self.airspeed = []
        self.alpha = []
        self.drag = []
        self.time = []


    def updateState(self, m, v_BN_W, gamma, sigma, lat, lon, h, airspeed, alpha, drag, dt):
        self.m.append(m)
        self.v_WN_N.append(self.wind)
        self.v_BN_W.append(v_BN_W)
        self.gamma.append(gamma)
        self.sigma.append(sigma)
        self.lat.append(lat)
        self.lon.append(lon)
        self.h.append(h)
        self.airspeed.append(airspeed)
        self.alpha.append(alpha)
        self.drag.append(drag)
        self.time.append(self.time[-1]+dt)


class FW_NLPerf_GuidanceSystem:
    """ Fixed-Wing Nonlinear Performance Guidance System
    TODO:
        After algorithm implementation, change the following:
        m -> mass
        v_BN_W_c -> V_veh_c
        v_BN_W -> V_veh
        sigma_c -> heading_c (or psi_c)
        sigma -> heading (or psi)

    Guidance System inputs:
        m           mass of the aircraft
        v_BN_W_c    Commanded inertial velocity
        v_BN_W      Current inertial velocity (output from EOM)
        gamma_c     Commanded flight path angle
        gamma       Current flight path angle (output from EOM)
        airspeed    Current airspeed (output from EOM)
        sigma_c     Commanded heading angle clockwise from North
        sigma       Current heading angle clockwise from North (output from EOM)

    Guidance System outputs:
        thrust      magnitude of thrust vector in line with aircraft body
        lift        magnitude of lift vector in line with aircraft z-axis
        alpha_c     angle of attack commanded by guidance system (unused in EOM)
        mu          wind-axes bank angle (phi_w in textbook)
        h_c         commanded height of aircraft (? - unused in EOM)

    Guidance System assumptions:
        a. Air mass (wind) uniformly translating w.r.t. Earth-fixed inertial frame
        b. Aero forces/moments on vehicle depend only on airspeed and orientation to air mass
        c. Presence of winds give rise to differences in inertial velocity and airspeed
    """

    def __init__(self, vehicle, K_Tp, K_Ti, K_Lp, K_Li, K_mu_p, dt=0.01, initialCommand=None):
        self.Vehicle = vehicle
        self.K_Tp = K_Tp
        self.K_Ti = K_Ti
        self.K_Lp = K_Lp
        self.K_Li = K_Li
        self.K_mu_p = K_mu_p
        self.dt = dt  # Default is 0.01 seconds
        self.UserCommand = [np.nan, np.nan, np.nan]  # Velocity, h_dot (rate of climb), psi (heading)
        self.v_BN_W_c = np.nan
        self.gamma_c = np.nan
        self.sigma_c = np.nan
        if initialCommand is not None:
            if len(initialCommand) != 3:
                print(f'Cannot set {initialCommand} as a trajectory. Voiding input.')
            else:
                self.setCommandTrajectory(initialCommand[0], initialCommand[1], initialCommand[2])
        self.units = self.Vehicle.units  # Adopt units from vehicle at init
        self.angles = self.Vehicle.angles  # Adopt angle units from vehicle at init
        self.V_err = 0
        self.xT = 0

    def setCommandTrajectory(self, velocity, rate_of_climb, heading):
        # Set the new user command
        self.UserCommand = [velocity, rate_of_climb, heading]
        # Add the new user command to guidance system input history
        if len(self.Vehicle.time) == 0:
            UserCommandTime = '0'
        else:
            UserCommandTime = str(self.Vehicle.time[-1])
        self.UserCommandHistory = {UserCommandTime: self.UserCommand}

        # Convert the trajectory command to guidance system inputs
        self.v_BN_W_c = velocity  # Commanded velocity
        self.gamma_c = np.arcsin(rate_of_climb/velocity)  # Commanded flight path angle
        self.sigma_c = heading  # Commanded heading

    def getGuidanceCommands(self, m, v_BN_W, gamma, airspeed, sigma, dt=None):
        """ Get the Guidance System outputs based on current state and commanded trajectory.
        Note: Be sure to check the current vehicle units via:
            > [FW_NLPerf_GuidanceSystem].Vehicle.units
            > [FW_NLPerf_GuidanceSystem].Vehicle.angles
            **At the initialization of the guidance system, the units of the vehicle were inherited.
                However, it is recommended to check the current guidance system units as well:
                > [FW_NLPerf_GuidanceSystem].units
                > [FW_NLPerf_GuidanceSystem].angles

        Parameters
        ----------
        Inputs:

        m : float
            Current mass of the aircraft.
        v_BN_W : float
            Current inertial velocity of the aircraft.
        gamma : float
            Current flight path angle of the aircraft.
        airspeed : float
            Current airspeed of the aircraft.
        sigma : float
            Current heading angle of the aircraft.

        Outputs:


        """
        if np.nan in [self.v_BN_W_c, self.gamma_c, self.sigma_c]:
            print('Unable to get Guidance commands because no User Trajectory Command has been set.')
            return

        if dt is None:
            dt = self.dt
        thrust = self._thrustCommand(m, v_BN_W, dt)
        print(f'Commanding thrust: {thrust} lbf')

    def _thrustCommand(self, m, v_BN_W, dt):
        V_err_old = self.V_err
        xT_old = self.xT
        self.V_err = self.v_BN_W_c - v_BN_W  # Calculate inertial velocity error

        # Evaluate ODE x_T_dot = m*V_err to receive x_T for new velocity error
        sol = solve_ivp(self.__xT_dot_ode, [V_err_old, self.V_err], [xT_old])
        self.xT = sol.xT[-1]

        # Use xT in calculation of Thrust command
        Tc = self.K_Ti*self.xT + self.K_Tp*self.m[-1]*self.V_err
        return Tc

    def __xT_dot_ode(self, Ve, xT): return self.Vehicle.m[-1] * Ve


if __name__ == '__main__':
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
    
    # Define the aircraft's initial conditions
    init_cond = {'v_BN_W': 400 * mph2fps,
                 'h': 0,
                 'gamma': 0,
                 'sigma': 0,
                 'lat': 33.2098 * d2r,
                 'lon': -87.5692 * d2r,
                 'v_WN_N': [25 * mph2fps, 25 * mph2fps, 0],
                 'weight': 300000}

    # Build the aircraft object
    my_C130 = FixedWingVehicle(new_aircraft_parameters, init_cond)

    # Build the guidance system using the aircraft object and control system transfer function constants
    C130_Guidance = FW_NLPerf_GuidanceSystem(my_C130, 0.08, 0.002, 0.5, 0.01, 0.075)

    # Give the aircraft a command
    C130_Guidance.setCommandTrajectory(450 * mph2fps, 5 * d2r, 15 * d2r)

    # Test command system
    # C130_Guidance.getGuidanceCommands(300000/const_gravity, C130_Guidance.Vehicle.InitialConditions.v_BN_W, C130_Guidance.Vehicle.InitialConditions.gamma,
    #                                   C130_Guidance.Vehicle.InitialConditions.v_BN_W, C130_Guidance.Vehicle.InitialConditions.sigma)

    print(my_C130.state.weight)
    # print(C130_Guidance.TF.K_Li)
    # print(C130_Guidance.sigma_c)

    C130_Guidance.Vehicle.updateState(300000*const_gravity, 405*mph2fps, 0, 0, 33.21 * d2r, -87.6 * d2r, 2,
                                      405*mph2fps, 3*d2r, 0.2, 0.01)
    print(my_C130.state)