#!/usr/bin/env python
""" A single drone simulation of the FANGS implementing both trajectory and flyover commands.

    NOTE: MOVE TO TOP DIRECTORY BEFORE RUNNING THIS SIMULATION
"""
from vehicle.FixedWingVehicle import FixedWingVehicle
from vehicle.ideal_EOM import ideal_EOM_RBFW as RBFW
import controller.utils as utils
from controller.FANGS import GuidanceSystem
import matplotlib.pyplot as plt
import tracking.track_generator as track
import numpy as np


rng_seed = 7

def runsim(stopTime, saveSimulationFilePath=None, saveFiguresFolderPath=None):
    if saveSimulationFilePath is None:
        saveSimulationFilePath = '.'
    if saveFiguresFolderPath is None:
        saveFiguresFolderPath = '.'
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
        "speed_max": 115 * utils.knts2fps,
        "speed_min": 25 * utils.knts2fps,
        "Kf": 0,
        "omega_T": 2.0,
        "omega_L": 0.9,
        "omega_mu": 1.0,
        "T_max": 45,
        "K_Lmax": 0.3,
        "mu_max": 45 * utils.d2r,
        "C_Do": 0.05,
        "C_Lalpha": 0.6 / utils.d2r,
        "alpha_o": -0.1 * utils.d2r,
        "wing_area": 8,
        "aspect_ratio": 12,
        "wing_eff": 0.8,
    }

    # Build the drone aircraft objects
    with utils.Timer("build_drone_obj(s)"):
        drone1 = FixedWingVehicle(drone_parameters, aircraftID=1001, dt=C2["dt"])

    # Define the drone initial conditions - true for all 8 drones since C2 at hover
    lat, lon = utils.get_point_at_distance(C2['lat'], C2['lon'], 150, C2['heading'])
    drone_init_conds = {
        "v_BN_W": 50 * utils.knts2fps,  # Stabilizes into flight at 50kts
        "h": C2['alt'],  # Launched from C2 aircraft
        "gamma": 0*utils.d2r,  # 0 deg ascent
        "sigma": C2['heading'],  # Northbound
        "lat": lat,
        "lon": lon,
        "v_WN_N": [0 * utils.knts2fps, 15 * utils.knts2fps, 0],
        "weight": 80,
    }

    # Drone PI Guidance Transfer Functions
    TF_constants = {
        "K_Tp": 0.15,  # 0.2
        "K_Ti": 0.05,  # 0.01
        "K_Lp": 0.3,
        "K_Li": 0.03,  # 0.01
        "K_mu_p": 0.03,  # 0.075
        "K_alpha": 0.05,
        "K_velocity": 0.025
    }

    # Build the guidance system using the aircraft object and control system transfer function constants
    with utils.Timer("build_initial_drone_GuidanceSystem_obj(s)"):
        drone1_gnc = GuidanceSystem(drone1, TF_constants, drone_init_conds, dt=C2["dt"], verbose=verbose)

    # Launch each drone 5 seconds after the other
    launch_delay = 5  # seconds

    # Define minimum/maximum altitude
    land_altitude = 600
    minimum_altitude = 1000
    maximum_altitude = 20000

    # Initialize the track builder - drone 1 (launched at start of program)
    C2_drone1_tracker = track.noisy_a2a(C2["lat"], C2["lon"], C2["alt"], C2["roll"], C2["pitch"], C2["heading"],
                                        noise_mean=0, noise_std=0.15, time=drone1_gnc.time[-1])
    C2_drone1_tracker.angle_units = 'Degrees'
    C2_drone1_tracker.distance_units = 'Feet'
    C2_drone1_tracker.track_target(drone1_gnc.lat[-1], drone1_gnc.lon[-1], drone1_gnc.h[-1], drone1_gnc.time[-1])

    drones = {"drone1":drone1_gnc}

    """ RUN THE SIMULATION """
    runSim = True  # Set to True to run the simulation, or False to load .pkl files from previous simulations
    verbose = True  # Set to True to get a lot of outputs
    time_of_first_command = 2*60
    time_of_second_command = 10*60
    time_of_third_command = 30*60
    first_command = False
    second_command = False
    third_command = False

    if runSim:
        with utils.Timer(f"simulate_{stopTime}_s_drone_flight"):
            drone1_gnc.verbose = verbose
            while C2["time"][-1] < stopTime:
                for drone_name, drone_gnc in drones.items():
                    if not drone_gnc.crashed:
                        np.random.seed(drone_gnc.Vehicle.aircraftID)

                        # Command the drone if it is time to do so
                        if drone_gnc.time[-1] >= time_of_first_command+drone_gnc.time[0] and first_command is False:
                            print('>-- First Command --<')
                            vel = 95*utils.knts2fps  # 95 knots
                            fpa = 6*utils.d2r  # 6 degree ascent
                            hdg = 270*utils.d2r  # West-bound
                            drone_gnc.setCommandTrajectory(vel, fpa, hdg)
                            first_command = True
                        if drone_gnc.time[-1] >= time_of_second_command+drone_gnc.time[0] and second_command is False:
                            print('>-- Second Command --<')
                            vel = 50*utils.knts2fps  # 50 knts for surveillance
                            alt = 11000  # 11000 ft MSL
                            target_flyover = ((36.530367)*utils.d2r, (-112.057600)*utils.d2r)
                            drone_gnc.setCommandFlyover(vel, alt, target_flyover)
                            second_command = True
                        if drone_gnc.time[-1] >= time_of_third_command+drone_gnc.time[0] and third_command is False:
                            print('>-- Third Command --<')
                            vel = 50 * utils.knts2fps
                            alt = 7700  # 7700 ft MSL
                            target_flyover = ((36.449291)*utils.d2r, (-112.399009)*utils.d2r)
                            drone_gnc.setCommandFlyover(vel, alt, target_flyover)
                            third_command = True

                        # Attempt to rescue the drone if it is flying too high or too low
                        if drone_gnc.h[-1] <= minimum_altitude or drone_gnc.h[-1] >= maximum_altitude:
                            # Level off the aircraft
                            if drone_gnc.verbose: print(f'{drone_name} altitude = {drone_gnc.h[-1]}, commanding +/-10 degree rate of climb.')
                            if drone_gnc.h[-1] >= maximum_altitude:
                                change_needed = 0
                            else:
                                change_needed = 1
                            drone_gnc.setCommandTrajectory(drone_gnc.command.v_BN_W_history[-1], change_needed*10, drone_gnc.command.sigma_history[-1])

                        # Oops, the drone died
                        if drone_gnc.h[-1] <= land_altitude:
                            drone_gnc.crashed = True
                            t_crash = C2["time"][-1]
                            print(f'{drone_name} has crashed at time {t_crash}. Its wreckage can be found at {drone_gnc.lat[-1]/utils.d2r} N, {drone_gnc.lon[-1]/utils.d2r} E')
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
                if len(C2["time"])%5000 == 0:
                    perc = round(C2["time"][-1]/stopTime*100)
                    print(f'{perc}% complete...')

    else:
        with utils.Timer(f"load_drone_objects"):
            drones = {}  # Reset drones
            drones[f'drone1'] = utils.load_obj(r"C:\Users\sprin\OneDrive\Documents\Project\fangs\saved_simulations\Single_Agent_Sim\abs_command_drone1.pkl")

    with utils.Timer(f"track_drone_targets"):
        track_builders = {}
        for drone_name, drone_gnc in drones.items():
            print(f'Building track data for {drone_name}...')
            track_builders[drone_name] = track.noisy_a2a(C2["lat"], C2["lon"], C2["alt"], C2["roll"], C2["pitch"], C2["heading"],
                                                            noise_mean=0, noise_std=0.15, time=drone_gnc.time[0])
            # track_builders[drone_name] = track.ideal_a2a(C2["lat"], C2["lon"], C2["alt"], C2["roll"], C2["pitch"], C2["heading"])
            track_builders[drone_name].angle_units = 'Degrees'
            track_builders[drone_name].distance_units = 'Feet'
            for ii in range(len(drone_gnc.time)):
                # track_builders[drone_name].track_target_ideal(drone_gnc.lat[ii], drone_gnc.lon[ii], drone_gnc.h[ii], drone_gnc.time[ii])
                track_builders[drone_name].track_target(drone_gnc.lat[ii], drone_gnc.lon[ii], drone_gnc.h[ii], drone_gnc.time[ii])
                if ii == 10:
                    print()

            print(f'{drone_name} tracked from {track_builders[drone_name].time[0]} seconds to {track_builders[drone_name].time[-1]} seconds')

            # track_builders[drone_name].to_csv(f'{saveSimulationFilePath}\\{drone_name}_ideal_track.csv', downsample=10)
            # utils.gnc_to_csv(drone_gnc, f'{saveSimulationFilePath}\\{drone_name}_ideal_lla.csv')
            track_builders[drone_name].to_csv(f'{saveSimulationFilePath}\\{drone_name}_noisy_track_2.csv', downsample=10)
            utils.gnc_to_csv(drone_gnc, f'{saveSimulationFilePath}\\{drone_name}_noisy_dronedata_2.csv')

    with utils.Timer(f'saving_{len(track_builders.keys())}_track_plot'):
        for drone_name, drone_gnc in drones.items():
            utils.plotSim(drone_gnc, saveFiguresFolderPath)

    with utils.Timer(f'saving_KML_files'):
        for drone_name, drone_gnc in drones.items():
            utils.writeKMLfromObj(drone_gnc, saveFolder=saveSimulationFilePath, )

    if runSim:
        with utils.Timer('saving_drone_flight_objs'):
            for drone_name, drone_gnc in drones.items():
                utils.save_obj(drone_gnc, f'{saveSimulationFilePath}\\abs_command_{drone_name}.pkl')

    return


if __name__ == "__main__":
    with utils.Timer('OVERALL SIMULATION'):
        runsim(stopTime=45*60,
               saveSimulationFilePath=r'C:\Users\sprin\OneDrive\Documents\Project\fangs\saved_simulations\Single_Agent_Sim',
               saveFiguresFolderPath=r'C:\Users\sprin\OneDrive\Documents\Project\fangs\saved_simulations\Single_Agent_Sim')