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
__version__ = "1.2.0"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Production"

from vehicle.FixedWingVehicle import FixedWingVehicle
from vehicle.ideal_EOM import ideal_EOM_RBFW as RBFW
import controller.utils as utils
from controller.FANGS import GuidanceSystem
import matplotlib.pyplot as plt
import tracking.track_generator as track

import sys
import pytest
import numpy as np
import numpy.random as rnd
from copy import deepcopy

import scipy.stats as stats
import random

import gncpy.filters as gfilts
import gncpy.dynamics.basic as gdyn
import gncpy.distributions as gdistrib
import carbs.swarm_estimator.tracker as tracker
import serums.models as smodels
from serums.enums import GSMTypes, SingleObjectDistance

rng_seed = 7

def run_C2(stopTime, saveSimulationFilePath=None, saveFiguresFolderPath=None):
    verbose = False

    # Define the aircraft (C2) -- Assume it's in a perfect hover for now
    C2 = {
        "lat": 36.2434 * utils.d2r,
        "lon": -112.2822 * utils.d2r,
        "alt": 7000,
        "roll": 0,
        "pitch": 0,
        "heading": 0,
        "time": [0],
        "dt": 0.05,
    }

    # Define the aircraft (target drone)
    drone_parameters = {
        "weight_max": 80,
        "weight_min": 40,
        "speed_max": 75 * utils.knts2fps,
        "speed_min": 25 * utils.knts2fps,
        "Kf": 0,
        "omega_T": 2,
        "omega_L": 2.5,
        "omega_mu": 1,
        "T_max": 30,
        "K_Lmax": 2.6,
        "mu_max": 60 * utils.d2r,
        "C_Do": 0.02,
        "C_Lalpha": 0.1 / utils.d2r,
        "alpha_o": -0.05 * utils.d2r,
        "wing_area": 8,
        "aspect_ratio": 12,
        "wing_eff": 0.8,
    }

    # Build the drone aircraft objects
    with utils.Timer("build_drone_obj(s)"):
        drone1 = FixedWingVehicle(drone_parameters, aircraftID=1001, dt=C2["dt"])
        drone2 = FixedWingVehicle(drone_parameters, aircraftID=1002, dt=C2["dt"])
        drone3 = FixedWingVehicle(drone_parameters, aircraftID=1003, dt=C2["dt"])
        drone4 = FixedWingVehicle(drone_parameters, aircraftID=1004, dt=C2["dt"])
        drone5 = FixedWingVehicle(drone_parameters, aircraftID=1005, dt=C2["dt"])
        drone6 = FixedWingVehicle(drone_parameters, aircraftID=1006, dt=C2["dt"])
        drone7 = FixedWingVehicle(drone_parameters, aircraftID=1007, dt=C2["dt"])
        drone8 = FixedWingVehicle(drone_parameters, aircraftID=1008, dt=C2["dt"])

    # Define the drone initial conditions - true for all 8 drones since C2 at hover
    drone_init_conds = {
        "v_BN_W": 50 * utils.knts2fps,
        "h": C2['alt'],  # Launched from C2 aircraft
        "gamma": 0,  # No ascent or descent
        "sigma": 0,  # Northbound
        "lat": C2['lat'],
        "lon": C2['lon'],
        "v_WN_N": [10 * utils.knts2fps, 10 * utils.knts2fps, 0],
        "weight": 80,
    }

    # Drone PI Guidance Transfer Functions
    TF_constants = {
        "K_Tp": 0.08,
        "K_Ti": 0.002,
        "K_Lp": 0.5,
        "K_Li": 0.01,
        "K_mu_p": 0.075,
    }

    # Build the guidance system using the aircraft object and control system transfer function constants
    with utils.Timer("build_initial_drone_GuidanceSystem_obj(s)"):
        drone1_gnc = GuidanceSystem(drone1, TF_constants, drone_init_conds, dt=C2["dt"], verbose=verbose)

    # Launch each drone 5 seconds after the other
    launch_delay = 5  # seconds

    # Define minimum/maximum altitude
    land_altitude = 600
    minimum_altitude = 1000
    maximum_altitude = 10000

    # Initialize the track builder - drone 1 (launched at start of program)
    C2_drone1_tracker = track.noisy_a2a(C2["lat"], C2["lon"], C2["alt"], C2["roll"], C2["pitch"], C2["heading"],
                                        noise_mean=0, noise_std=0.15, time=drone1_gnc.time[-1])
    C2_drone1_tracker.angle_units = 'Degrees'
    C2_drone1_tracker.distance_units = 'Feet'
    C2_drone1_tracker.track_target(drone1_gnc.lat[-1], drone1_gnc.lon[-1], drone1_gnc.h[-1], drone1_gnc.time[-1])

    drones = {"drone1":drone1_gnc}

    """ RUN THE SIMULATION """
    runSim = False  # Set to True to run the simulation, or False to load .pkl files from previous simulations
    verbose = False  # Set to True to get a lot of outputs

    if runSim:
        with utils.Timer(f"simulate_{stopTime}_s_drone_flight"):
            while C2["time"][-1] < stopTime:

                # Launch a new drone every 5 seconds until there are 8 drones launched
                if C2["time"][-1] >= launch_delay*1 and "drone2" not in drones.keys():
                    drone2_gnc = GuidanceSystem(drone2, TF_constants, drone_init_conds, time=C2["time"][-1], dt=C2["dt"], verbose=verbose)
                    drones["drone2"] = drone2_gnc
                elif C2["time"][-1] >= launch_delay*2 and "drone3" not in drones.keys():
                    drone3_gnc = GuidanceSystem(drone3, TF_constants, drone_init_conds, time=C2["time"][-1], dt=C2["dt"], verbose=verbose)
                    drones["drone3"] = drone3_gnc
                elif C2["time"][-1] >= launch_delay*3 and "drone4" not in drones.keys():
                    drone4_gnc = GuidanceSystem(drone4, TF_constants, drone_init_conds, time=C2["time"][-1], dt=C2["dt"], verbose=verbose)
                    drones["drone4"] = drone4_gnc
                elif C2["time"][-1] >= launch_delay*4 and "drone5" not in drones.keys():
                    drone5_gnc = GuidanceSystem(drone5, TF_constants, drone_init_conds, time=C2["time"][-1], dt=C2["dt"], verbose=verbose)
                    drones["drone5"] = drone5_gnc
                elif C2["time"][-1] >= launch_delay*5 and "drone6" not in drones.keys():
                    drone6_gnc = GuidanceSystem(drone6, TF_constants, drone_init_conds, time=C2["time"][-1], dt=C2["dt"], verbose=verbose)
                    drones["drone6"] = drone6_gnc
                elif C2["time"][-1] >= launch_delay*6 and "drone7" not in drones.keys():
                    drone7_gnc = GuidanceSystem(drone7, TF_constants, drone_init_conds, time=C2["time"][-1], dt=C2["dt"], verbose=verbose)
                    drones["drone7"] = drone7_gnc
                elif C2["time"][-1] >= launch_delay*7 and "drone8" not in drones.keys():
                    drone8_gnc = GuidanceSystem(drone8, TF_constants, drone_init_conds, time=C2["time"][-1], dt=C2["dt"], verbose=verbose)
                    drones["drone8"] = drone8_gnc

                for drone_name, drone_gnc in drones.items():
                    if not drone_gnc.crashed:
                        np.random.seed(drone_gnc.Vehicle.aircraftID)
                        cmdtime = np.random.rand()

                        # Command the drone if it is time to do so
                        if drone_gnc.time[-1] >= 20*cmdtime+drone_gnc.time[0] and drone_gnc.command.time == 0:
                            np.random.seed(drone_gnc.Vehicle.aircraftID)
                            vel = np.random.randint(drone_gnc.Vehicle.speed_min+10, drone_gnc.Vehicle.speed_max-15) * utils.knts2fps
                            np.random.seed(drone_gnc.Vehicle.aircraftID)
                            roc = np.random.randint(-5, 5) * utils.d2r
                            np.random.seed(drone_gnc.Vehicle.aircraftID)
                            hdg = np.random.randint(0, 359) * utils.d2r
                            if hdg < 185 and hdg > 175:
                                hdg = 180 + np.sign(hdg)*5  # Avoid hitting the C2 - HOVER ONLY LOGIC
                            drone_gnc.setCommandTrajectory(vel, roc, hdg)
                            if drone_gnc.verbose: print(f'---\nCommanding {drone_name}:\n > Velocity: {vel/utils.knts2fps} knt\n > Flight Path Angle: {roc/utils.d2r} deg\n > Heading: {hdg/utils.d2r} deg\n > Drone Time: {drone_gnc.time[-1]}')

                        # Attempt to rescue the drone if it is flying too high or too low
                        if drone_gnc.h[-1] <= minimum_altitude or drone_gnc.h[-1] >= maximum_altitude:
                            # Level off the aircraft
                            if drone_gnc.verbose: print(f'{drone_name} altitude = {drone_gnc.h[-1]}, commanding 0 degree rate of climb (level off).')
                            drone_gnc.setCommandTrajectory(drone_gnc.command.v_BN_W_history[-1], 0, drone_gnc.command.sigma_history[-1])

                        # Oops, the drone died
                        if drone_gnc.h[-1] <= land_altitude:
                            drone_gnc.crashed = True
                            if drone_gnc.verbose: print(f'{drone_name} has crashed. Its wreckage can be found at {drone_gnc.lat[-1]/utils.d2r} N, {drone_gnc.lon[-1]/utils.d2r} E')
                            break

                        # Generate FANGS output (guidance commands)
                        drone_gnc.getGuidanceCommands()

                        # Calculate aircraft state using ideal EOM
                        new_state = RBFW(
                            drone_gnc.Vehicle,
                            drone_gnc.Thrust[-1],
                            drone_gnc.Lift[-1],
                            drone_gnc.alpha_c,
                            drone_gnc.mu[-1],
                            drone_gnc.h_c,
                            drone_gnc.v_BN_W[-1],
                            drone_gnc.gamma[-1],
                            drone_gnc.sigma[-1],
                            drone_gnc.mass[-1],
                            drone_gnc.airspeed[-1],
                            drone_gnc.lat[-1],
                            drone_gnc.lon[-1],
                            drone_gnc.h[-1],
                            drone_gnc.time[-1],
                            drone_gnc.dt)
                        mass, v_BN_W, gamma, sigma, lat, lon, h, airspeed, alpha, drag = new_state
                        drone_gnc.updateSystemState(
                            mass=mass,
                            v_BN_W=v_BN_W,
                            gamma=gamma,
                            sigma=sigma,
                            lat=lat,
                            lon=lon,
                            h=h,
                            airspeed=airspeed,
                            alpha=alpha,
                            drag=drag)

                C2["time"].append(C2["time"][-1] + C2["dt"])
                if len(C2["time"])%500 == 0:
                    perc = round(C2["time"][-1]/stopTime*100)
                    print(f'{perc}% complete...')

    else:
        with utils.Timer(f"load_drone_objects"):
            drones = {}  # Reset drones
            drones_to_load = [1, 2, 3, 4, 5, 6, 7, 8]
            for drone in drones_to_load:
                drones[f'drone{drone}'] = utils.load_obj(f'drone{drone}.pkl')
            # print(drones.items())

        with utils.Timer(f"track_drone_targets"):
            track_builders = {}
            for drone_name, drone_gnc in drones.items():
                print(f'Building noisy track data for {drone_name}...')
                track_builders[drone_name] = track.noisy_a2a(C2["lat"], C2["lon"], C2["alt"], C2["roll"], C2["pitch"], C2["heading"],
                                                             noise_mean=0, noise_std=0.15, time=drone_gnc.time[0])
                track_builders[drone_name].angle_units = 'Degrees'
                track_builders[drone_name].distance_units = 'Feet'
                for ii in range(len(drone_gnc.time)):
                    track_builders[drone_name].track_target(drone_gnc.lat[ii], drone_gnc.lon[ii], drone_gnc.h[ii], drone_gnc.time[ii])

                print(f'{drone_name} tracked from {track_builders[drone_name].time[0]} seconds to {track_builders[drone_name].time[-1]} seconds')

                track_builders[drone_name].to_csv(f'{drone_name}_track.csv', downsample=10)
                utils.gnc_to_csv(drone_gnc, f'{drone_name}_lla.csv')

        with utils.Timer('saving_track_plot'):
            fig, (ax_brg, ax_elv, ax_rng) = plt.subplots(3)
            [ax_brg.plot(track.time, track.target.bearing) for _, track in track_builders.items()]
            [ax_rng.plot(track.time, track.target.range) for _, track in track_builders.items()]
            [ax_elv.plot(track.time, track.target.elevation) for _, track in track_builders.items()]
            ax_rng.set_title(f'target range [Feet]')
            ax_brg.set_title(f'target bearing [Degrees]')
            ax_elv.set_title(f'target elevation [Degrees]')
            ax_brg.set_xticklabels([])
            ax_brg.set_xticks([])
            ax_brg.axes.get_xaxis().set_visible(False)
            ax_brg.grid(visible='True')
            ax_elv.set_xticklabels([])
            ax_elv.set_xticks([])
            ax_elv.axes.get_xaxis().set_visible(False)
            ax_elv.grid(visible='True')
            ax_rng.grid(visible='True')
            fig.savefig('6tracks.png')

    if runSim:
        with utils.Timer('saving_drone_flight_objs'):
            for drone_name, drone_gnc in drones.items():
                utils.save_obj(drone_gnc, f'{drone_name}.pkl')

    return


if __name__ == "__main__":
    run_C2(stopTime=6*60)
