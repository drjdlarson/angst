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
const_gravity = 32.17


class FixedWingVehicle:
    """Implements the base vehicle.
    This class allows the user to define the physical model of an aircraft to be used
        as an object of the FixedWingVehicle class.
    The default aircraft parameters, set to np.nan upon unitilization unless specified, are:
        weight_max, weight_min, speed_max, speed_min, Kf, omega_T, omega_L, omega_mu, T_max, K_Lmax, mu_max,
        C_Do, C_Lalpha, alpha_o, wing_area, aspect_ratio, wing_eff
    Other parameters may be defined as required by using the .setAircraftParameters() function,
        passing the new parameter(s) as a key/value pair(s) in a dict.
    """
    units = 'Imperial'
    angles = 'Radians'

    def __init__(self, VehicleParameters):
        """Initalize a Fixed Wing Vehicle object.

        Parameters
        ----------
        VehicleParameters : :dict:`Physical parameters of the aircraft`
            Required keys are:
                weight_max
                weight_min
                speed_max
                speed_min
                Kf
                omega_T
                omega_L
                T_max
                K_Lmax
                mu_max
                C_Do
                C_Lalpha
                alpha_o
                wing_area
                aspect_ratio
                wing_eff
            Optional keys are:
                mdot
        """

        self.weight_max = VehicleParameters['weight_max']  # lbs
        self.weight_min = VehicleParameters['weight_min']  # lbs
        self.speed_max = VehicleParameters['speed_max']  # fps
        self.speed_min = VehicleParameters['speed_min']  # fps
        self.Kf = VehicleParameters['Kf']  # Fuel burning gain value ?
        self.omega_T = VehicleParameters['omega_T']  # rad/s - time constant for engine/airframe response
        self.omega_L = VehicleParameters['omega_L']  # rad/s  - time constant for engine/airframe response
        self.omega_mu = VehicleParameters['omega_mu']  # rad/s  - time constant for engine/airframe response
        self.T_max = VehicleParameters['T_max']  # lbs
        self.K_Lmax = VehicleParameters['K_Lmax']  # lbs/fps^2
        self.mu_max = VehicleParameters['mu_max']  # rad
        self.C_Do = VehicleParameters['C_Do']  # unitless
        self.C_Lalpha = VehicleParameters['C_Lalpha']  # rad^(-1)
        self.alpha_o = VehicleParameters['alpha_o']  # rad
        self.wing_area = VehicleParameters['wing_area']  # ft^2
        self.aspect_ratio = VehicleParameters['aspect_ratio']  # unitless
        self.wing_eff = VehicleParameters['wing_eff']  # ?
        self.mdot = np.nan  # Fuel burn rate, lbs/s
        if 'mdot' in VehicleParameters.keys():
            self.mdot = VehicleParameters['mdot']

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

    def __init__(self, vehicle, TF_constants, InitialConditions, time = 0, dt=0.01):
        """ Initialize a fixed-wing nonlinear performance guidance system.
        
        Parameters
        ----------
        vehicle : object of class FixedWingVehicle to be commanded
            Must have the following parameters set:
                weight_max, weight_min, speed_max, speed_min, Kf, omega_T,
                omega_L, omega_mu, T_max, K_Lmax, mu_max, C_Do, C_Lalpha,
                alpha_o, wing_area, aspect_ratio, wing_eff
        TF_constants : :dict:`Dictionary of PI Guidance transfer function coefficients`
            Required keys: K_Tp, K_Ti, K_Lp, K_Li, K_mu_p
        InitialConditions : :dict:`Dictionary of Initial Conditions`
            Required keys: v_BN_W, h, gamma, sigma, lat, lon, v_WN_N, weight
        time : :float:`Time of vehicle GNC initialization`
            Default value is 0. This can be used for vehicles spawned at varying times.
        dt : :float:`Time delta to be used for integration and next step calculations`
            Can also be specified at any later time for non-uniform time steps
        """
        self.Vehicle = vehicle

        # Set tuning parameters
        self.K_Tp = TF_constants['K_Tp']
        self.K_Ti = TF_constants['K_Ti']
        self.K_Lp = TF_constants['K_Lp']
        self.K_Li = TF_constants['K_Li']
        self.K_mu_p = TF_constants['K_mu_p']
        self.dt = dt  # Default is 0.01 seconds

        # Set Initial Conditions
        self.time = [time]
        self.v_BN_W = [InitialConditions['v_BN_W']]
        self.h = [InitialConditions['h']]
        self.gamma = [InitialConditions['gamma']]
        self.sigma = [InitialConditions['sigma']]
        self.lat = [InitialConditions['lat']]
        self.lon = [InitialConditions['lon']]
        self.v_WN_N = [InitialConditions['v_WN_N']]
        self.weight = [InitialConditions['weight']]
        self.mass = [self.weight[0]/const_gravity]
        self.airspeed = [utils.wind_vector(self.v_BN_W[0], self.gamma[0], self.sigma[0])]
        self.alpha = [0]
        self.drag = [0]
        self.mu = [0]

        # Set vehicle GNC initiation time
        self.time = [time]

        # Initialize user commands and internal variables
        self.command = self.userCommand(np.nan, np.nan, np.nan)
        self.units = self.Vehicle.units  # Adopt units from vehicle at init
        self.angles = self.Vehicle.angles  # Adopt angle units from vehicle at init
        self.V_err = 0
        self.xT = 0
        self.hdot_err = 0
        self.Tc = 0
        self.Thrust = [0]
        self.xL = 0
        self.Lc = 0
        self.Lift = [0]
        self.sigma_err = 0

    class userCommand:
        def __init__(self, v_BN_W, gamma, sigma):
            self.v_BN_W = v_BN_W
            self.gamma = gamma
            self.sigma = sigma

    def setCommandTrajectory(self, velocity, rate_of_climb, heading):
        # Set the new user command
        self.command.v_BN_W = velocity  # Commanded velocity
        self.command.gamma = np.arcsin(rate_of_climb/velocity)  # Commanded flight path angle
        self.command.sigma = heading  # Commanded heading

    def stepTime(self, dt=None):
        if dt is None:
            dt = self.dt
        self.getGuidanceCommands(dt)
        self.getEquationsOfMotions_Ideal(dt)
        self.time.append(self.time[-1]+dt)

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
        if np.nan in [self.command.v_BN_W, self.command.gamma, self.command.sigma]:
            print('Unable to get Guidance commands because no User Trajectory Command has been set.')
            return

        if dt is None:
            dt = self.dt
        thrust_c, thrust = self._thrustGuidanceSystem(dt)
        lift, alpha_c, h_c = self._liftGuidanceSystem(dt)
        mu = self._headingGuidanceSystem(dt)
        print(f'Commanding thrust: {thrust_c} lbf, resulting in {thrust} lbf thrust')
        print(f'Commanding lift: {lift} lbf, by setting angle of attack to {alpha_c} and altitude to {h_c}')
        print(f'Commanding wind-axes bank angle: {mu}')

    def getEquationsOfMotion_Ideal(self, dt=None):
        if dt is None:
            dt = self.dt

        # Calculate fuel burn based on thrust
        sol = solve_ivp(self.__m_dot_ode, [self.time[-1], self.time[-1] + dt], [self.mass[-1]], method='RK45')
        self.mass.append(sol.y[-1][-1])

        # Calculate alpha and drag
        a = ((2*self.Lift[-1]) / (const_density * self.Vehicle.wing_area * self.Vehicle.C_Lalpha * self.airspeed[-1]**2)) + self.Vehicle.alpha_o
        d = 0.5 * const_density * self.Vehicle.wing_area * self.Vehicle.C_Do * self.airspeed[-1]**2
        self.alpha.append(a)
        self.drag.append(d)

        # Calculate v_BN_W, gamma, sigma
        y0 = [self.v_BN_W[-1], self.gamma[-1], self.sigma[-1]]
        sol = solve_ivp(self.__eom_ode, [self.time[-1], self.time[-1] + dt], y0, method='RK45')
        self.v_BN_W.append(sol.y[0][-1])
        self.gamma.append(sol.y[1][-1])
        self.sigma.append(sol.y[2][-1])

        # Calculate airspeed
        self.airspeed.append(utils.wind_vector(self.v_BN_W[-1], self.gamma[-1], self.sigma[-1]))

        # Convert to ECEF
        y0 = [self.lat[-1], self.lon[-1], self.h[-1]]
        sol = solve_ivp(self.__ecef_ode, [self.time[-1], self.time[-1] + dt], y0, method='RK45')
        self.lat.append(sol.y[0][-1])
        self.lon.append(sol.y[1][-1])
        self.h.append(sol.y[2][-1])

    def _thrustGuidanceSystem(self, dt):
        xT_old = self.xT
        self.V_err = self.command.v_BN_W - self.v_BN_W[-1]  # Calculate inertial velocity error

        # Evaluate ODE x_T_dot = m*V_err via RK45 to receive x_T for new velocity error
        sol = solve_ivp(self.__xT_dot_ode, [self.time[-1], self.time[-1] + dt], [xT_old], method='RK45')
        self.xT = sol.y[-1][-1]

        # Use xT in calculation of Thrust command
        self.Tc = self.K_Ti*self.xT + self.K_Tp*self.mass[-1]*self.V_err

        # Saturation of thrust command
        if self.Tc > self.Vehicle.T_max:
            print(f'Commanded thrust {self.Tc} exceeds max thrust {self.Vehicle.T_max}')
            self.Tc = self.Vehicle.T_max

        sol = solve_ivp(self.__T_dot_ode, [self.time[-1], self.time[-1] + dt], [self.Thrust[-1]], method='RK45')
        self.Thrust.append(sol.y[-1][-1])

        # Saturation of vehicle thrust
        if self.Thrust[-1] > self.Vehicle.T_max:
            self.Thrust[-1] = self.Vehicle.T_max

        return self.Tc, self.Thrust[-1]

    def _liftGuidanceSystem(self, dt):
        # Step 1: Calculate max lift (L_max)
        # Inputs: v_BN_W (Current aircraft inertial velocity)
        # Outputs: L_max (maximum lift)
        L_max = self.v_BN_W[-1]**2 * self.Vehicle.K_Lmax

        # Calculate commanded lift (L_c)
        xL_old = self.xL
        self.hdot_err = self.command.v_BN_W*(np.sin(self.command.gamma) - np.sin(self.gamma[-1]))
        # Evaluate ODE x_L_dot = m*h_dot_err via RK45 to receive x_L for Lift Command calculation
        sol = solve_ivp(self.__xL_dot_ode, [self.time[-1], self.time[-1] + dt], [xL_old], method='RK45')
        self.xL = sol.y[-1][-1]
        self.Lc = self.K_Li*self.xL + self.K_Lp*self.mass[-1]*self.hdot_err

        # Saturation (upper/lower limits on commanded lift)
        if self.Lc > L_max:
            print(f'Command lift {self.Lc} is greater than max lift {L_max}, setting to {L_max}')
            self.Lc = L_max

        # Calculate lift
        sol = solve_ivp(self.__L_dot_ode, [self.time[-1], self.time[-1] + dt], [self.Lift[-1]], method='RK45')
        self.Lift.append(sol.y[-1][-1])

        if self.Lift[-1] > L_max:
            self.Lift[-1] = L_max

        # Calculate commanded angle of attack (alpha_c)
        alpha_c = 2 * self.Lc / (const_density * self.Vehicle.wing_area * self.Vehicle.C_Lalpha * self.airspeed[-1]**2) + self.Vehicle.alpha_o

        # Calculate altitude command (h_c)
        h_c = np.sin(self.command.gamma) * self.command.v_BN_W * (self.time[-1] + dt) + self.h[0]

        return self.Lift[-1], alpha_c, h_c

    def _headingGuidanceSystem(self, dt):
        # NOTE mu in code equals Phi_W in book (wind-axes bank angle)
        # NOTE sigma in code equals Psi_W in book (heading)
        self.sigma_err = self.command.sigma - self.sigma[-1]
        mu = self.K_mu_p*(self.command.v_BN_W / const_gravity) * self.sigma_err

        if np.abs(mu) > self.Vehicle.mu_max:
            print(f'Command bank angle {mu} exceeds max allowable bank angle |{self.Vehicle.mu_max}|')
            mu = np.sign(mu) * self.Vehicle.mu_max

        self.mu.append(mu)

        return mu
    
    def updateState(self, m=0, v_BN_W=0, gamma=0, sigma=0, lat=0, lon=0, h=0, airspeed=0, alpha=0, drag=0, time=0):
        """ Potentially OBE? """
        self.v_WN_N.append(self.v_WN_N)  # Constant wind; future versions can have shifting wind
        self.v_BN_W.append(v_BN_W)
        self.gamma.append(gamma)
        self.sigma.append(sigma)
        self.lat.append(lat)
        self.lon.append(lon)
        self.h.append(h)
        self.alpha.append(alpha)
        self.drag.append(drag)
        if time == 0:
            # If the user did not specify a time step, assume it's self.dt (inherited from vehicle class)
            time = self.time[-1] + self.dt
        self.time.append(time)
        if m == 0:
            m = self.m[-1] - self.Vehicle.mdot * (self.time[-1] - self.time[-2])
        self.mass.append(m)
        self.weight.append(m * const_gravity)
        self.airspeed.append(utils.wind_vector(v_BN_W, gamma, sigma))

    def __xT_dot_ode(self, t, xT=0): return self.mass[-1] * self.V_err

    def __T_dot_ode(self, t, T): return -1*self.Vehicle.omega_T*T + self.Vehicle.omega_T*self.Tc

    def __xL_dot_ode(self, t, xL): return self.mass[-1] * self.hdot_err

    def __L_dot_ode(self, t, L): return -1*self.Vehicle.omega_L*L + self.Vehicle.omega_L*self.Lc

    def __m_dot_ode(self, t, m): return -1*self.Vehicle.Kf * self.Thrust[-1]

    def __eom_ode(self, t, y0):
        # y0 = [v_BN_W, gamma, sigma]
        v_BN_W_dot = ((self.Thrust[-1] - self.drag[-1]) / self.mass[-1]) - const_gravity * self.gamma[-1]
        gamma_dot = (1/self.v_BN_W[-1]) * ((self.Lift[-1] * np.cos(self.mu[-1]/self.mass[-1]) - const_gravity * np.cos(self.gamma[-1])))
        sigma_dot = (1/(self.v_BN_W[-1] * np.cos(self.gamma[-1]))) * (self.Lift[-1] * np.sin(self.mu[-1] / self.mass[-1]))
        return [v_BN_W_dot, gamma_dot, sigma_dot]

    def __ecef_ode(self, t, y0):
        # y0 = [lat, lon, h]
        lat_dot = self.v_BN_W[-1] * np.cos(self.gamma[-1]) * np.cos(self.sigma[-1]) / (Re_bar + self.h[-1])
        lon_dot = self.v_BN_W[-1] * np.cos(self.gamma[-1]) * np.sin(self.sigma[-1]) / ((Re_bar + self.h[-1]) * np.cos(self.lat[-1]))
        h_dot = self.v_BN_W[-1] * np.sin(self.gamma[-1])
        return [lat_dot, lon_dot, h_dot]

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

    # Build the aircraft object
    my_C130 = FixedWingVehicle(new_aircraft_parameters)
    
    # Define the aircraft's initial conditions
    init_cond = {'v_BN_W': 400 * mph2fps,
                 'h': 0,
                 'gamma': 0,
                 'sigma': 0,
                 'lat': 33.2098 * d2r,
                 'lon': -87.5692 * d2r,
                 'v_WN_N': [25 * mph2fps, 25 * mph2fps, 0],
                 'weight': 300000}

    # PI Guidance Transfer Functions
    TF_constants = {'K_Tp': 0.08, 'K_Ti': 0.002, 'K_Lp': 0.5, 'K_Li': 0.01, 'K_mu_p': 0.075}

    # Build the guidance system using the aircraft object and control system transfer function constants
    C130_Guidance = FW_NLPerf_GuidanceSystem(my_C130, TF_constants, init_cond)

    # Give the aircraft a command
    # velocity = 450 mph
    # rate_of_climb = 5 degrees
    # heading = 15 degrees (NNE)
    C130_Guidance.setCommandTrajectory(450 * mph2fps, 5 * d2r, 15 * d2r)

    # Define a maximum thrust value
    # my_C130.AircraftParams.setAircraftParameters({'max_thrust': 8600})

    # Test command system
    C130_Guidance.getGuidanceCommands()

    # Test EOMs
    C130_Guidance.getEquationsOfMotion_Ideal()

    # print(my_C130.v_BN_W, my_C130.weight)
    # print(C130_Guidance.TF.K_Li)
    # print(C130_Guidance.sigma_c)

    # C130_Guidance.Vehicle.updateState(v_BN_W=405*mph2fps)
    # print(my_C130.v_BN_W, my_C130.weight)