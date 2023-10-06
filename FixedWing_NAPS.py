#!/usr/bin/env python
""" Fixed Wing (N)onlinear (A)ircraft - (P)erformance (S)imulation
        The algorithms followed for the nonlinear controller are described in the case study for a
        Nonlinear Aircraft-Performance Simulation by Dr. John Schierman in his Modern Flight Dynamics textbook.
        This project is a nonlinear controller for a fixed-wing aircraft.
        This code will allow one to model a rigid aircraft operating in steady winds.
        The aircraft will be guided via nonlinear feedback laws to follow a specified flight profile:
            - Commanded velocities
            - Commanded rates of climb/descent
            - Commanded headings
        In the current implementation, the attitude dynamics of the vehicle are approximated using ideal equations of motion.
"""
__author__ = "Alex Springer"
__version__ = "1.0.1"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Production"

from vehicle.FixedWingVehicle import FixedWingVehicle
import controller.utils as utils
from controller.FANGS import GuidanceSystem


def run_FW_UAV_GNC_Test(stopTime, loadSimulationFilePath=None, saveSimulationFilePath=None, saveFiguresFolderPath=None):
    # If a previously-run simulation .pkl file is supplied, load that instead of running a new simulation.
    if loadSimulationFilePath is not None:
        print(f'Loading saved simulation data from <{loadSimulationFilePath}>')
        acft_Guidance = utils.load_obj(loadSimulationFilePath)

    # Otherwise, run a simulation and save the results to a .pkl file if a saveSimulationFilePath is given.
    else:
        # Define the aircraft (example aircraft is C-130)
        new_aircraft_parameters = {'weight_max': 327000,
                                   'weight_min': 157000,
                                   'speed_max': 600 * utils.mph2fps,
                                   'speed_min': 200 * utils.mph2fps,
                                   'Kf': 4e-6,
                                   'omega_T': 2,
                                   'omega_L': 2.5,
                                   'omega_mu': 1,
                                   'T_max': 72000,
                                   'K_Lmax': 2.6,
                                   'mu_max': 30 * utils.d2r,
                                   'C_Do': 0.0183,
                                   'C_Lalpha': 0.0920 / utils.d2r,
                                   'alpha_o': -0.05 * utils.d2r,
                                   'wing_area': 1745,
                                   'aspect_ratio': 10.1,
                                   'wing_eff': 0.613}

        # Build the aircraft object
        with utils.Timer('build_acft_obj'):
            my_acft = FixedWingVehicle(new_aircraft_parameters)

        # Define the aircraft's initial conditions
        init_cond = {'v_BN_W': 400 * utils.mph2fps,
                    'h': 0,
                    'gamma': 0,
                    'sigma': 0,
                    'lat': 33.2098 * utils.d2r,
                    'lon': -87.5692 * utils.d2r,
                    'v_WN_N': [25 * utils.mph2fps, 25 * utils.mph2fps, 0],
                    'weight': 300000}

        # PI Guidance Transfer Functions
        TF_constants = {'K_Tp': 0.08, 'K_Ti': 0.002, 'K_Lp': 0.5, 'K_Li': 0.01, 'K_mu_p': 0.075}

        # Build the guidance system using the aircraft object and control system transfer function constants
        with utils.Timer('build_GuidanceSystem_obj'):
            acft_Guidance = GuidanceSystem(my_acft, TF_constants, init_cond)
        
        with utils.Timer('run_FW_UAV_GNC_Test'):
            while acft_Guidance.time[-1] < stopTime:
                if acft_Guidance.time[-1] >= 1 and acft_Guidance.time[-2] < 1:
                    # Give the aircraft a command
                    # velocity = 450 mph
                    # rate_of_climb = 5 degrees
                    # heading = 15 degrees (NNE)
                    acft_Guidance.setCommandTrajectory(450 * utils.mph2fps, 5 * utils.d2r, 15 * utils.d2r)
                acft_Guidance.stepTime()

        if saveSimulationFilePath is not None:
            with utils.Timer('save_obj'):
                utils.save_obj(acft_Guidance, saveSimulationFilePath)

    # Show/save the plots from the simulation
    if saveFiguresFolderPath is None:
        utils.plotSim(acft_Guidance, showPlots=True)
    else:
        utils.plotSim(acft_Guidance, saveFolder=saveFiguresFolderPath, showPlots=False)

    return


if __name__ == '__main__':
    # Run through simulation -- Note that commands are set at 1s
    # run_FW_UAV_GNC_Test(120, loadSimulationFilePath=r'.\saved_simulations\120s_C130_1s_command_run_FW_UAV_GNC_test_C130.pkl', saveFiguresFolderPath=r'.\saved_simulations\figures\120 second C130 simulation')
    run_FW_UAV_GNC_Test(15)