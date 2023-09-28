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
__version__ = "0.0.1"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Production"

import numpy as np
from scipy.integrate import solve_ivp
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
        self.mdot = 10  # Fuel burn rate
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
    dt = 0.01  # Default time step for the vehicle

    def __init__(self, params, m, v_BN_W, gamma, sigma, lat, lon, h, alpha, drag, time=0):
        """Initalize a Fixed Wing Vehicle object.

        Parameters
        ----------
        params : :dict:`Dict of all applicable aircraft parameters`
            Parameters of the aircraft.
        m : :float:`Initial mass of vehicle`
        v_BN_W : :float:`Initial inertial velocity of aircraft`
        gamma : :float:`Initial flight path angle`
        sigma : :float:`Initial wind-axes heading angle`
        lat : :float:`Initial latitude`
        lon : :float:`Initial longitude`
        h : :float:`Initial height above ground` - TODO
        alpha : :float:`Initial vehicle angle of attack`
        drag : :float:`Initial vehicle drag force`
        time : :float:`Initial vehicle time (default is 0)`
        """

        # Set the aircraft physical parameters
        self.AircraftParams = AircraftParams(params)

        # Initialize vehicle state variables
        self.m = [m]
        self.weight = [m * const_gravity]
        self.v_BN_W = [v_BN_W]
        self.v_WN_N = [0, 0, 0]  # Added for future time-dependent winds, default winds are none
        self.gamma = [gamma]
        self.sigma = [sigma]
        self.lat = [lat]
        self.lon = [lon]
        self.h = [h]
        # self.airspeed = [airspeed]
        self.alpha = [alpha]
        self.drag = [drag]
        self.time = [time]

    def setWinds(self, v_WN_N):
        """
            Right now, this function only sets the static wind speed.
            In the future, I want to have the ability to create time-dependent winds,
            and to save wind state history.
        """
        self.v_WN_N = v_WN_N


    def updateState(self, m=0, v_BN_W=0, gamma=0, sigma=0, lat=0, lon=0, h=0, airspeed=0, alpha=0, drag=0, time=0):
        self.v_WN_N.append(self.v_WN_N)  # Constant wind; future versions can have shifting wind
        self.v_BN_W.append(v_BN_W)
        self.gamma.append(gamma)
        self.sigma.append(sigma)
        self.lat.append(lat)
        self.lon.append(lon)
        self.h.append(h)
        # self.airspeed.append(airspeed)
        self.alpha.append(alpha)
        self.drag.append(drag)
        if time == 0:
            time = self.time[-1] + self.dt
        self.time.append(time)
        if m == 0:
            m = self.m[-1] - self.AircraftParams.mdot * (self.time[-1] - self.time[-2])
        self.m.append(m)
        self.weight.append(m * const_gravity)


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

        # Set tuning parameters
        self.K_Tp = K_Tp
        self.K_Ti = K_Ti
        self.K_Lp = K_Lp
        self.K_Li = K_Li
        self.K_mu_p = K_mu_p
        self.dt = dt  # Default is 0.01 seconds

        # Initialize user commands and set, if applicable
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

    def getGuidanceCommands(self, dt=None):
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
        thrust = self._thrustCommand()
        lift, alpha_c, h_c = self._liftGuidanceSystem()
        mu = self._headingGuidanceSystem()
        print(f'Commanding thrust: {thrust} lbf')
        print(f'Commanding lift: {lift} lbf, by setting angle of attack to {alpha_c} and altitude to {h_c}')
        print(f'Commanding wind-axes bank angle: {mu}')

    def _thrustCommand(self):
        V_err_old = self.V_err
        xT_old = self.xT
        self.V_err = self.v_BN_W_c - self.Vehicle.v_BN_W[-1]  # Calculate inertial velocity error

        # Evaluate ODE x_T_dot = m*V_err via RK45 to receive x_T for new velocity error
        sol = solve_ivp(self.__xT_dot_ode, [V_err_old, self.V_err], [xT_old], method='RK45')
        self.xT = sol.y[-1][-1]

        # Use xT in calculation of Thrust command
        Tc = self.K_Ti*self.xT + self.K_Tp*self.Vehicle.m[-1]*self.V_err
        if hasattr(self.Vehicle.AircraftParams, 'max_thrust'):
            if Tc > self.Vehicle.AircraftParams.max_thrust:
                Tc = self.Vehicle.AircraftParams.max_thrust
        return Tc

    def __xT_dot_ode(self, Ve, xT): return self.Vehicle.m[-1] * Ve

    def _liftGuidanceSystem(self):
        return np.nan, np.nan, np.nan

    def _headingGuidanceSystem(self):
        return np.nan


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
                 'weight': 300000,
                 'time': 0}

    # Build the aircraft object
    my_C130 = FixedWingVehicle(new_aircraft_parameters, m=300000/const_gravity, v_BN_W=400*mph2fps, gamma=0, sigma=0, lat=33.2098*d2r,
                               lon=-87.5692*d2r, h=0, airspeed=0, alpha=0, drag=0)

    # Build the guidance system using the aircraft object and control system transfer function constants
    C130_Guidance = FW_NLPerf_GuidanceSystem(my_C130, 0.08, 0.002, 0.5, 0.01, 0.075)

    # Give the aircraft a command
    C130_Guidance.setCommandTrajectory(450 * mph2fps, 5 * d2r, 15 * d2r)

    # Test command system
    # C130_Guidance.getGuidanceCommands(300000/const_gravity, C130_Guidance.Vehicle.InitialConditions.v_BN_W, C130_Guidance.Vehicle.InitialConditions.gamma,
    #                                   C130_Guidance.Vehicle.InitialConditions.v_BN_W, C130_Guidance.Vehicle.InitialConditions.sigma)

    print(my_C130.v_BN_W, my_C130.weight)
    # print(C130_Guidance.TF.K_Li)
    # print(C130_Guidance.sigma_c)

    C130_Guidance.Vehicle.updateState(v_BN_W=405*mph2fps)
    print(my_C130.v_BN_W, my_C130.weight)