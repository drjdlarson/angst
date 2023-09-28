# -*- coding: utf-8 -*-
"""
Created on Thurs September 28 2023

@author: ahspringer

This script is used as a sandbox for testing various functions in the FW_UAV_GNC project.
"""

from FixedWingUAV_Control import *

if __name__ == "__main__":
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
                               lon=-87.5692*d2r, h=0, alpha=0, drag=0)

    # Build the guidance system using the aircraft object and control system transfer function constants
    C130_Guidance = FW_NLPerf_GuidanceSystem(my_C130, 0.08, 0.002, 0.5, 0.01, 0.075)

    # Give the aircraft a command
    # velocity = 450 mph
    # rate_of_climb = 5 degrees
    # heading = 15 degrees (NNE)
    C130_Guidance.setCommandTrajectory(450 * mph2fps, 5 * d2r, 15 * d2r)

    # Define a maximum thrust value
    # my_C130.AircraftParams.setAircraftParameters({'max_thrust': 8600})

    # Test command system
    C130_Guidance.getGuidanceCommands()

    # print(my_C130.v_BN_W, my_C130.weight)
    # print(C130_Guidance.TF.K_Li)
    # print(C130_Guidance.sigma_c)

    # C130_Guidance.Vehicle.updateState(v_BN_W=405*mph2fps)
    # print(my_C130.v_BN_W, my_C130.weight)