#!/usr/bin/env python
""" Ideal Equations of Motion for Fixed-Wing Aircraft
    Designed to be used as an ideal state estimator for the FANGS guidance algorithm.
    Approximates the attitude dynamics and responses of the vehicle:
        a. Aerodynamic Response
        b. Fuel Burn Response
        c. Motion Response
        d. Airspeed Response
        e. Conversion to Earth-Centered, Earth-Fixed (ECEF) Response
    
    EOM inputs:
        thrust      thrust command
        lift        lift command
        mu          wind-axes bank angle command

    EOM outputs:
        m           mass following fuel burn
        v_BN_W      inertial velocity
        gamma       flight path angle
        sigma       heading angle clockwise from North
        lat         latitude
        lon         longitude
        h           altitude
        airspeed    airspeed
        alpha       angle of attack response
        drag        drag force
    
    FANGS Guidance System assumptions:
        a. Air mass (wind) uniformly translating w.r.t. Earth-fixed inertial frame
        b. Aero forces/moments on vehicle depend only on airspeed and orientation to air mass
        c. Presence of winds give rise to differences in inertial velocity and airspeed
"""

import numpy as np
from scipy.integrate import solve_ivp
import controller.utils as utils

def ideal_EOM_RBFW(Vehicle, thrust_c, lift_c, alpha_c, mu_c, h_c, v_BN_W, gamma, sigma, mass, airspeed, lat, lon, h, time, dt=None):
    """ An ideal equations of motion solver for a rigid body fixed-wing aircraft.

    Equations of Motion (EOM) System - Approximates aircraft responses to guidance system
        a. Aerodynamic Response
        b. Fuel Burn Response
        c. Motion Response
        d. Airspeed Response
        e. Conversion to Earth-Centered, Earth-Fixed (ECEF) Response

    Parameters
    ----------
    Inputs:
        Vehicle : :object:`object of class vehicle.FixedWingVehicle.FixedWingVehicle`
        thrust_c : :float:`current guidance system thrust command`
        lift_c : :float:`current guidance system lift command`
        alpha_c : :float:`current guidance system angle of attack command`
        mu_c : :float:`current guidance system wind-axes bank angle command`
        h_c : :float:`current guidance system altitude command`
        v_BN_W : :float:`most recent inertial velocity of the aircraft`
        gamma : :float:`most recent flight path angle of the aircraft`
        sigma : :float:`most recent heading angle of the aircraft`
        mass : :float:`most recent aircraft mass estimate`
        airspeed : :float:`most recent aircraft airspeed estimate`
        lat : :float:`most recent aircraft latitude estimate`
        lon : :float:`most recent aircraft longitude estimate`
        h : float:`most recent aircraft altitude estimate` 
        time : :float:`current aircraft time value`
        dt : :float:`optional time step value; recommended to use guidance system time step.
            default is Vehicle.dt, which may differ from user guidance system time step.`

    Outputs:
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
    """
    if dt is None:
        dt = Vehicle.dt

    # Calculate fuel burn based on thrust
    sol = solve_ivp(__m_dot_ode, [time, time + dt], [mass], args=(Vehicle.Kf, thrust_c), method='RK45')
    mass = sol.y[-1][-1]

    # Calculate alpha and drag
    alpha = _calculateAlpha(lift_c, airspeed, Vehicle.wing_area, Vehicle.C_Lalpha, Vehicle.alpha_o)
    drag = _calculateDrag(lift_c, airspeed, Vehicle.wing_area, Vehicle.aspect_ratio, Vehicle.wing_eff, Vehicle.C_Do)

    # Calculate v_BN_W, gamma, sigma
    y0 = [v_BN_W, gamma, sigma]
    sol = solve_ivp(__eom_ode, [time, time + dt], y0, args=(thrust_c, lift_c, drag, mass, mu_c), method='RK45')
    v_BN_W = sol.y[0][-1]
    gamma = sol.y[1][-1]
    sigma = sol.y[2][-1]

    # Calculate airspeed
    airspeed = utils.wind_vector(v_BN_W, gamma, sigma)

    # Convert to ECEF
    y0 = [lat, lon, h]
    sol = solve_ivp(__ecef_ode, [time, time + dt], y0, args=(v_BN_W, gamma, sigma), method='RK45')
    lat = sol.y[0][-1]
    lon = sol.y[1][-1]
    h = sol.y[2][-1]

    return mass, v_BN_W, gamma, sigma, lat, lon, h, airspeed, alpha, drag

def _calculateAlpha(lift, airspeed, wing_area, C_Lalpha, alpha_o):
    return ((2*lift) / (utils.const_density * wing_area * C_Lalpha * airspeed**2)) + alpha_o

def _calculateDrag(lift, airspeed, wing_area, aspect_ratio, wing_eff, C_Do):
    return 0.5 * utils.const_density * wing_area * C_Do * airspeed**2 + (2 * lift**2) / (utils.const_density * wing_area * np.pi * aspect_ratio * wing_eff * airspeed**2)

def __m_dot_ode(t, m, Kf, thrust): return -1*Kf * thrust

def __eom_ode(t, state, thrust, lift, drag, mass, mu):
    v_BN_W, gamma, sigma = state  # Unpack the state
    v_BN_W_dot = ((thrust - drag) / mass) - utils.const_gravity * np.sin(gamma)
    gamma_dot = (1/v_BN_W) * ((lift * np.cos(mu)/mass) - utils.const_gravity * np.cos(gamma))
    sigma_dot = (1/(v_BN_W * np.cos(gamma))) * (lift * np.sin(mu) / mass)
    return [v_BN_W_dot, gamma_dot, sigma_dot]

def __ecef_ode(t, state, v_BN_W, gamma, sigma):
    lat, lon, h = state  # Unpack the state
    lat_dot = v_BN_W * np.cos(gamma) * np.cos(sigma) / (utils.Re_bar + h)
    lon_dot = v_BN_W * np.cos(gamma) * np.sin(sigma) / ((utils.Re_bar + h) * np.cos(lat))
    h_dot = v_BN_W * np.sin(gamma)
    return [lat_dot, lon_dot, h_dot]