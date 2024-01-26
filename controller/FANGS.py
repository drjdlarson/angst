#!/usr/bin/env python
""" (F)ixed Wing (A)ircraft (N)onlinear (G)uidance (S)ystem
        The algorithms followed for the nonlinear controller are described in the case study for a
        Nonlinear Aircraft-Performance Simulation by Dr. John Schierman in his Modern Flight Dynamics textbook.
        This project is a nonlinear controller for a fixed-wing aircraft.
        The aircraft will be guided via nonlinear feedback laws to follow a specified flight profile:
            - Commanded velocities
            - Commanded rates of climb/descent
            - Commanded headings

    At each time step, the guidance system will be updated with commands. The user must then either:
        a. Import state data from measurements
        b. Import state data from a state estimator

    Version notes: v2.0.0
        This version will add changes to the API to allow the user to input an "absolute" command:
            - Flyover waypoint
            - Groundspeed
            - Altitude
        This adds the capability for the user to designate mission-related commands.
"""
__author__ = "Alex Springer"
__version__ = "2.0.0"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Development"

import numpy as np
from scipy.integrate import solve_ivp
import controller.utils as utils

class GuidanceSystem:
    """ Fixed-Wing Nonlinear Guidance System

    The FW_NL_GuidanceSystem algorithm - Generates guidance commands for the aircraft
        a. Thrust Guidance System
        b. Lift Guidance System
        c. Heading Guidance System

    Guidance System inputs:
        m           mass of the aircraft
        v_BN_W_c    commanded inertial velocity
        v_BN_W      current inertial velocity (output from EOM)
        gamma_c     commanded flight path angle
        gamma       current flight path angle (output from EOM)
        airspeed    current airspeed (output from EOM)
        sigma_c     commanded heading angle clockwise from North
        sigma       current heading angle clockwise from North (output from EOM)

        -- New with version 2.0:
        waypoint_c  commanded fly-over waypoint
        altitude_c  commanded aircraft altitude
        v_GS_c      commanded aircraft groundspeed

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

    def __init__(self, vehicle, TF_constants, InitialConditions, time = 0, dt=0.01, verbose=True):
        """ Initialize a fixed-wing nonlinear performance guidance system.

        Note: The aircraft will initialize with a set of steady-state commands designed to
              keep the aircraft flying at the initialized state. If you need to update the aircraft
              commands, use either setCommandTrajectory() or setFlyoverCommand()
        
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
        self.verbose = verbose
        if self.verbose: print(f'\n---\nInitializing GNC for aircraft {self.Vehicle.aircraftID} at time {time}')

        # Set tuning parameters
        self.K_Tp = TF_constants['K_Tp']
        self.K_Ti = TF_constants['K_Ti']
        self.K_Lp = TF_constants['K_Lp']
        self.K_Li = TF_constants['K_Li']
        self.K_mu_p = TF_constants['K_mu_p']
        self.dt = dt  # Default is 0.01 seconds
        self.Vehicle.dt = self.dt  # Update the Vehicle's default time step

        # Set Initial Conditions
        self.time = [time]
        self.v_BN_W = [InitialConditions['v_BN_W']]  # Body-frame velocity of aircraft
        self.h = [InitialConditions['h']]
        self.gamma = [InitialConditions['gamma']]
        self.sigma = [InitialConditions['sigma']]
        self.lat = [InitialConditions['lat']]
        self.lon = [InitialConditions['lon']]
        self.v_WN_N = [InitialConditions['v_WN_N']]  # NED-frame wind speed
        self.weight = [InitialConditions['weight']]
        self.mass = [self.weight[0]/utils.const_gravity]
        self.airspeed = [utils.wind_vector(self.v_BN_W[0], self.gamma[0], self.sigma[0])]

        # Set vehicle GNC initiation time
        self.time = [time]

        # Initialize user commands and internal variables
        self.command = self.userCommand(self.v_BN_W[0], self.gamma[0], self.sigma[0], self.h[0])
        self.v_BN_W_c_hist = [0]
        self.gamma_c_hist = [0]
        self.sigma_c_hist = [0]
        self.units = self.Vehicle.units  # Adopt units from vehicle at init
        self.angles = self.Vehicle.angles  # Adopt angle units from vehicle at init
        self.V_err = 0
        self.hdot_err = 0
        self.xL = self.weight[0] / self.K_Li  # Lc = K_Li*xL + K_Lp*m*hdot_err
        self.Lc = [self.weight[0]]  # Command it to maintain lift at equilibrium
        self.Lift = [self.weight[0]]  # Assuming equilibrium
        self.h_c = [self.h[0]]
        self.sigma_err = 0
        self.crashed = False

        # Calculate initial alpha, drag, and mu
        self.alpha = [self._calculateAlpha()]
        self.alpha_c = [self.alpha[0]]
        self.drag = [self._calculateDrag()]
        self.mu = [self._calculateMu()]
        self.xT = self.drag[0] / self.K_Ti  # Tc = K_Ti*xT + K_Tp*m*V_err
        self.Tc = [self.drag[0]]
        self.Thrust = [self.drag[0]]

    class userCommand:
        def __init__(self, v_BN_W, gamma, sigma, h):
            self.time = 0
            self.guidance_command_time = 0
            self.v_BN_W = v_BN_W
            self.gamma = gamma
            self.sigma = sigma
            self.v_BN_W_history = [v_BN_W]
            self.gamma_history = [gamma]
            self.sigma_history = [sigma]
            self.airspeed = utils.wind_vector(v_BN_W, gamma, sigma)
            self.airspeed_history = [self.airspeed]
            self.h_ref = h

            # Add hooks for flyover command
            self.groundspeed = np.nan
            self.altitude = np.nan
            self.waypoint = (np.nan, np.nan)
            self.groundspeed_history = [np.nan]
            self.altitude_history = [np.nan]
            self.waypoint_history = [(np.nan, np.nan)]

            # Flag what kind of command is operating:
            #   Currently accepted states: "trajectory", "flyover"
            self._command_type = "trajectory"
            self._change_type = True

        def save_history(self):
            self.v_BN_W_history.append(self.v_BN_W)
            self.gamma_history.append(self.gamma)
            self.sigma_history.append(self.sigma)
            self.airspeed_history.append(self.airspeed)
            self.groundspeed_history.append(self.groundspeed)
            self.altitude_history.append(self.altitude)
            self.waypoint_history.append(self.waypoint)

    def setCommandTrajectory(self, velocity, flight_path_angle, heading):
        """ Set a user-defined commanded aircraft trajectory
        
        Note: This command will come into effect on the next iteration of the guidance system.

        Parameters
        ----------
        velocity : :float:`(feet per second) The commanded forward velocity of the aircraft.`
            Use this command to set the forward airspeed of the aircraft.
        flight_path_angle : :float:`(radians) The commanded flight path angle of the aircraft.`
            The flight path angle is the angle at which the aircraft is either climbing (+) or descending (-)
        heading : :float:`(radians) The commanded heading of the aircraft.`
            The heading of the aircraft is defined as clockwise from North.
        change_type : :boolean:`(True/False) Change the type of user command`
            This is used internally to avoid resetting the type of command input by the user.
        """
        # Set the new user command
        self.command.v_BN_W = velocity  # Commanded velocity
        self.command.gamma = flight_path_angle  # Commanded flight path angle
        self.command.sigma = heading  # Commanded heading
        self.command.airspeed = utils.wind_vector(self.command.v_BN_W, self.command.gamma, self.command.sigma)

        # Update errors
        self.V_err = self.command.v_BN_W - self.v_BN_W[-1]  # Calculate inertial velocity error
        self.hdot_err = self.command.v_BN_W*(np.sin(self.command.gamma) - np.sin(self.gamma[-1]))
        self.sigma_err = self.command.sigma - self.sigma[-1]

        # Update time since last command
        self.command.time = self.time[-1]

        # Update the command reference heigh
        self.command.h_ref = self.h[-1]

        # Update the system operating command type
        if self.command._change_type:
            self.command._command_type = "trajectory"


    def setFlyoverCommand(self, groundspeed, altitude, waypoint):
        """ Set a user-defined commanded fly-over point
        
        Note: This command will come into effect on the next iteration of the guidance system.

        WARNING: The altitude is measured from mean sea level, so be sure to account for terrain!

        Parameters
        ----------
        groundspeed : :float:`(feet per second) The commanded forward velocity of the aircraft relative to the ground.`
            Use this command to set the forward airspeed of the aircraft relative to the ground.
        altitude : :float:`(feet) The commanded altitude of the aircraft.`
            The altitude is measured from mean sea level, so be sure to account for terrain!
        waypoint : :tuple:`(radians) The commanded fly-over target point.`
            Input as (latitude, longitude) in degree decimal format.
        change_type : :boolean:`(True/False) Change the type of user command`
            This is used internally to avoid resetting the type of command input by the user.
        """
        # Set the new user command
        self.command.groundspeed = groundspeed  # Commanded groundspeed
        self.command.altitude = altitude  # Commanded flight path angle
        self.command.waypoint = waypoint  # Commanded heading
        # self.command.airspeed = utils.wind_vector(self.command.v_BN_W, self.command.gamma, self.command.sigma)

        # Update errors
        # self.V_err = self.command.v_BN_W - self.v_BN_W[-1]  # Calculate inertial velocity error
        # self.hdot_err = self.command.v_BN_W*(np.sin(self.command.gamma) - np.sin(self.gamma[-1]))
        # self.sigma_err = self.command.sigma - self.sigma[-1]

        # Update time since last command
        self.command.guidance_command_time = self.time[-1]

        # Update the command reference height
        self.command.h_ref = self.h[-1]

        # Update the system operating command type
        if self.command._change_type:
            self.command._command_type = "flyover"

    def getGuidanceCommands(self, dt=None):
        """ Get the Guidance System outputs based on current state and commanded trajectory.
        Note: Be sure to check the current vehicle units via:
            > [FW_NL_GuidanceSystem].Vehicle.units
            > [FW_NL_GuidanceSystem].Vehicle.angles
            **At the initialization of the guidance system, the units of the vehicle were inherited.
                However, it is recommended to check the current guidance system units as well:
                > [FW_NL_GuidanceSystem].units
                > [FW_NL_GuidanceSystem].angles

        Parameters
        ----------
        dt : :float:`Optional. Time step value.
        """
        if np.nan in [self.command.v_BN_W, self.command.gamma, self.command.sigma]:
            print('Unable to get Guidance commands because no User Trajectory Command has been set.')
            return

        if dt is None:
            dt = self.dt

        if self.command._command_type == "flyover":
            self._setTrajectory()

        self._thrustGuidanceSystem(dt)
        self._liftGuidanceSystem(dt)
        self._headingGuidanceSystem(dt)

    def updateSystemState(self, mass=None, v_BN_W=None, gamma=None, sigma=None, lat=None, lon=None, h=None, airspeed=None, alpha=None, drag=None, dt=None):
        """ User-supplied state update before asking for next guidance system command.
        If any states are left un-supplied, they will be estimated using an ideal equations of motion algorithm.
    
        Parameters
        ----------
        m : :float:`estimated aircraft mass following fuel burn`
        v_BN_W : :float:`estimated aircraft inertial velocity response`
        gamma : :float:`estimated flight path angle response`
        sigma : :float:`estimated heading angle clockwise from North response`
        lat : :float:`estimated aircraft latitude response`
        lon : :float:`estimated aircraft longitude response`
        h : :float:`estimated aircraft altitude response`
        airspeed : :float:`estimated aircraft airspeed response`
        alpha : :float:`estimated aircraft angle of attack response`
        drag : :float:`estimated aircraft drag force response`
        dt : :float:`Optional. Time step value.
        """
        if dt is None:
            dt = self.dt
        sys_states = (mass, v_BN_W, gamma, sigma, lat, lon, h, airspeed, alpha, drag)
        if self.crashed:
            self.mass.append(self.mass[-1])
            self.v_BN_W.append(0)
            self.gamma.append(0)
            self.sigma.append(0)
            self.lat.append(self.lat[-1])
            self.lon.append(self.lon[-1])
            self.h.append(self.h[-1])
            self.airspeed.append(0)
            self.alpha.append(0)
            self.drag.append(0)
            self.time.append(self.time[-1] + dt)
            self.command.save_history()
        else:
            if None in sys_states:
                ideal_eom = self._getEquationsOfMotion_Ideal()
                sys_states = [ideal_eom[i] if sys_states[i] is None else sys_states[i] for i in range(len(sys_states))]
            self.mass.append(sys_states[0])
            self.v_BN_W.append(sys_states[1])
            self.gamma.append(sys_states[2])
            self.sigma.append(sys_states[3])
            self.lat.append(sys_states[4])
            self.lon.append(sys_states[5])
            self.h.append(sys_states[6])
            self.airspeed.append(sys_states[7])
            self.alpha.append(sys_states[8])
            self.drag.append(sys_states[9])
            self.time.append(self.time[-1] + dt)
            self.command.save_history()

    def _setTrajectory(self):
        """ A proportional controller to calculate the required trajectory commands to meet the
            user-designated flyover point (if set).
        """
        # Needed are the velocity, flight-path angle, and heading
        lat = self.lat[-1]
        lon = self.lon[-1]
        required_lat = self.command.waypoint[0]
        required_lon = self.command.waypoint[1]

        # velocity - Fly as fast as possible until within 1500 feet
        if utils.get_distance(lat, lon, required_lat, required_lon) > 1500:
            velocity = self.Vehicle.speed_max
        else:
            velocity = self.command.groundspeed()

        # flight path angle
        # Need to calculate some angle between where the aircraft currently is and the altitude it needs to be at
        maximum_glideslope = 25  # degrees
        alpha = np.arctan2(self.command.altitude - self.h[-1], 250)  # hard-code 250 feet as adjacent in TOA
        # saturate at 30 degrees glide slope
        if abs(alpha) > maximum_glideslope:
            alpha = np.sign(alpha) * maximum_glideslope
        flight_path_angle = alpha

        # heading
        heading = utils.get_bearing(lat, lon, required_lat, required_lon)

        self.command._change_type = False  # Don't change to a trajectory controller
        self.setCommandTrajectory(velocity, flight_path_angle, heading)
        self.command._change_type = True  # Do no harm
        return

    def _getEquationsOfMotion_Ideal(self, dt=None):
        """ An ideal equations of motion solver for a rigid body fixed-wing aircraft.
        This will be the default state solver for any updated system states the user does not supply at any time step.

        Parameters
        ----------
        dt : :float:`Optional. Time step value.
        """
        if dt is None:
            dt = self.dt

        # Calculate fuel burn based on thrust
        sol = solve_ivp(self.__m_dot_ode, [self.time[-1], self.time[-1] + dt], [self.mass[-1]], method='RK45')
        mass = sol.y[-1][-1]

        # Calculate alpha and drag
        alpha = self._calculateAlpha()
        drag = self._calculateDrag()

        # Calculate v_BN_W, gamma, sigma
        y0 = [self.v_BN_W[-1], self.gamma[-1], self.sigma[-1]]
        sol = solve_ivp(self.__eom_ode, [self.time[-1], self.time[-1] + dt], y0, method='RK45')
        v_BN_W = sol.y[0][-1]
        gamma = sol.y[1][-1]
        sigma = sol.y[2][-1]

        # Calculate airspeed
        airspeed = utils.wind_vector(self.v_BN_W[-1], self.gamma[-1], self.sigma[-1])

        # Convert to ECEF
        y0 = [self.lat[-1], self.lon[-1], self.h[-1]]
        sol = solve_ivp(self.__ecef_ode, [self.time[-1], self.time[-1] + dt], y0, method='RK45')
        lat = sol.y[0][-1]
        lon = sol.y[1][-1]
        h = sol.y[2][-1]

        return mass, v_BN_W, gamma, sigma, lat, lon, h, airspeed, alpha, drag

    def _thrustGuidanceSystem(self, dt):
        xT_old = self.xT
        self.V_err = self.command.v_BN_W - self.v_BN_W[-1]  # Calculate inertial velocity error

        # Evaluate ODE x_T_dot = m*V_err via RK45 to receive x_T for new velocity error
        sol = solve_ivp(self.__xT_dot_ode, [self.time[-1], self.time[-1] + dt], [xT_old], method='RK45')
        self.xT = sol.y[-1][-1]

        # Use xT in calculation of Thrust command
        self.Tc.append(self.K_Ti*self.xT + self.K_Tp*self.mass[-1]*self.V_err)

        # Saturation of thrust command
        if self.Tc[-1] > self.Vehicle.T_max:
            # print(f'({self.Vehicle.aircraftID}) Commanded thrust {self.Tc} exceeds max thrust {self.Vehicle.T_max}')
            self.Tc[-1] = self.Vehicle.T_max
        elif self.Tc[-1] < 0:
            self.Tc[-1] = 0

        sol = solve_ivp(self.__T_dot_ode, [self.time[-1], self.time[-1] + dt], [self.Thrust[-1]], method='RK45')
        self.Thrust.append(sol.y[-1][-1])

        # Saturation of vehicle thrust
        if self.Thrust[-1] > self.Vehicle.T_max:
            self.Thrust[-1] = self.Vehicle.T_max
        elif self.Thrust[-1] < 0:
            self.Thrust[-1] = 0

        return self.Tc[1], self.Thrust[-1]

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
        self.Lc.append(self.K_Li*self.xL + self.K_Lp*self.mass[-1]*self.hdot_err)

        # Saturation (upper/lower limits on commanded lift)
        if self.Lc[-1] > L_max:
            # print(f'({self.Vehicle.aircraftID}) Command lift {self.Lc} is greater than max lift {L_max}, setting to {L_max}')
            self.Lc[-1] = L_max

        # Calculate lift
        sol = solve_ivp(self.__L_dot_ode, [self.time[-1], self.time[-1] + dt], [self.Lift[-1]], method='RK45')
        self.Lift.append(sol.y[-1][-1])

        if self.Lift[-1] > L_max:
            self.Lift[-1] = L_max

        # Calculate commanded angle of attack (alpha_c)
        alpha_c = 2 * self.Lc[-1] / (utils.const_density * self.Vehicle.wing_area * self.Vehicle.C_Lalpha * self.airspeed[-1]**2) + self.Vehicle.alpha_o
        self.alpha_c.append(alpha_c)

        # Calculate altitude command (h_c)
        h_c = np.sin(self.command.gamma) * self.command.v_BN_W * (self.time[-1] + dt) + self.command.h_ref
        self.h_c.append(h_c)

        return self.Lift[-1], alpha_c, h_c

    def _headingGuidanceSystem(self, dt):
        # NOTE mu in code equals Phi_W in book (wind-axes bank angle)
        # NOTE sigma in code equals Psi_W in book (heading)
        self.sigma_err = self.command.sigma - self.sigma[-1]
        mu = self._calculateMu()

        if np.abs(mu) > self.Vehicle.mu_max:
            # print(f'({self.Vehicle.aircraftID}) Command bank angle {mu} exceeds max allowable bank angle |{self.Vehicle.mu_max}|')
            mu = np.sign(mu) * self.Vehicle.mu_max

        self.mu.append(mu)

        return mu

    def _calculateAlpha(self):
        return ((2*self.Lift[-1]) / (utils.const_density * self.Vehicle.wing_area * self.Vehicle.C_Lalpha * self.airspeed[-1]**2)) + self.Vehicle.alpha_o

    def _calculateDrag(self):
        return 0.5 * utils.const_density * self.Vehicle.wing_area * self.Vehicle.C_Do * self.airspeed[-1]**2 + (2 * self.Lift[-1]**2) / (utils.const_density * self.Vehicle.wing_area * np.pi * self.Vehicle.aspect_ratio * self.Vehicle.wing_eff * self.airspeed[-1]**2)

    def _calculateMu(self):
        return self.K_mu_p*(self.command.v_BN_W / utils.const_gravity) * self.sigma_err

    def __xT_dot_ode(self, t, xT=0): return self.mass[-1] * self.V_err

    def __T_dot_ode(self, t, T): return -1*self.Vehicle.omega_T*T + self.Vehicle.omega_T*self.Tc[-1]

    def __xL_dot_ode(self, t, xL): return self.mass[-1] * self.hdot_err

    def __L_dot_ode(self, t, L): return -1*self.Vehicle.omega_L*L + self.Vehicle.omega_L*self.Lc[-1]

    def __m_dot_ode(self, t, m): return -1*self.Vehicle.Kf * self.Thrust[-1]

    def __eom_ode(self, t, y0):
        # y0 = [v_BN_W, gamma, sigma]
        v_BN_W_dot = ((self.Thrust[-1] - self.drag[-1]) / self.mass[-1]) - utils.const_gravity * np.sin(self.gamma[-1])
        gamma_dot = (1/self.v_BN_W[-1]) * ((self.Lift[-1] * np.cos(self.mu[-1])/self.mass[-1]) - utils.const_gravity * np.cos(self.gamma[-1]))
        sigma_dot = (1/(self.v_BN_W[-1] * np.cos(self.gamma[-1]))) * (self.Lift[-1] * np.sin(self.mu[-1]) / self.mass[-1])
        return [v_BN_W_dot, gamma_dot, sigma_dot]

    def __ecef_ode(self, t, y0):
        # y0 = [lat, lon, h]
        lat_dot = self.v_BN_W[-1] * np.cos(self.gamma[-1]) * np.cos(self.sigma[-1]) / (utils.Re_bar + self.h[-1])
        lon_dot = self.v_BN_W[-1] * np.cos(self.gamma[-1]) * np.sin(self.sigma[-1]) / ((utils.Re_bar + self.h[-1]) * np.cos(self.lat[-1]))
        h_dot = self.v_BN_W[-1] * np.sin(self.gamma[-1])
        return [lat_dot, lon_dot, h_dot]
