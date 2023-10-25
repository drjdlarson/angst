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
from vehicle.ideal_EOM import ideal_EOM_RBFW as RBFW
import controller.utils as utils
from controller.FANGS import GuidanceSystem
import matplotlib.pyplot as plt
import tracking.track_generator as track


def run_C2(stopTime, saveSimulationFilePath=None, saveFiguresFolderPath=None):
    # Define the aircraft (C2) -- Assume it's in a hover
    C2 = {'lat':36.2434 * utils.d2r,
          'lon':-112.2822 * utils.d2r,
          'alt':7000,
          'roll':0,
          'pitch':0,
          'heading':0}

    # Define the aircraft (SR drone)
    drone_parameters = {'weight_max': 80,
                        'weight_min': 40,
                        'speed_max': 75 * utils.knts2fps,
                        'speed_min': 25 * utils.knts2fps,
                        'Kf': 0,
                        'omega_T': 7,
                        'omega_L': 20,
                        'omega_mu': 3,
                        'T_max': 30,
                        'K_Lmax': 30,
                        'mu_max': 45 * utils.d2r,
                        'C_Do': 0.02,
                        'C_Lalpha': 0.1 / utils.d2r,
                        'alpha_o': -0.05 * utils.d2r,
                        'wing_area': 8,
                        'aspect_ratio': 8,
                        'wing_eff': 0.8}

    # Build the aircraft object
    with utils.Timer('build_drone_obj'):
        drone = FixedWingVehicle(drone_parameters, aircraftID=1001)

    # Define the drone's initial conditions
    drone1_init_cond = {'v_BN_W': 50 * utils.knts2fps,
                        'h': 7000,  # 7000ft MSL
                        'gamma': 0,  # No ascent or descent
                        'sigma': 0,  # Northbound
                        'lat': 36.2434 * utils.d2r,
                        'lon': -112.2822 * utils.d2r,
                        'v_WN_N': [10 * utils.knts2fps, 10 * utils.knts2fps, 0],
                        'weight': 80}

    # Drone PI Guidance Transfer Functions
    TF_constants = {'K_Tp': 0.15, 'K_Ti': 0.01, 'K_Lp': 0.7, 'K_Li': 0.05, 'K_mu_p': 0.075}

    # Build the guidance system using the aircraft object and control system transfer function constants
    with utils.Timer('build_drone_GuidanceSystem_obj'):
        drone1_gnc = GuidanceSystem(drone, TF_constants, drone1_init_cond)
        bearing = [track.target_bearing((C2['lat'], C2['lon'], C2['alt']), (drone1_gnc.lat[-1], drone1_gnc.lon[-1], drone1_gnc.h[-1]), method='lla')]
    
    # Initialize the track builder
    C2_drone1_tracker = track.air_to_air_3d_ideal(C2['lat'], C2['lon'], C2['alt'],
                                                  C2['roll'], C2['pitch'], C2['heading'],
                                                  time=drone1_gnc.time[-1])

    with utils.Timer(f'simulate_{stopTime}_s_drone_flight'):
        # drone1_gnc = utils.load_obj(r'saved_simulations\drone1_6min.pkl')
        runSim = True
        
        if runSim:
            while drone1_gnc.time[-1] < stopTime:
                # "Measure" the drone from the C2 aircraft
                C2_drone1_tracker.track_target(drone1_gnc.lat[-1], drone1_gnc.lon[-1], drone1_gnc.h[-1], drone1_gnc.time[-1])

                if drone1_gnc.time[-1] >= 1 and drone1_gnc.command.time == 0:
                    # Give the aircraft a command at the ~1 second mark
                    # velocity = 75 knots
                    # rate_of_climb = 5 degrees
                    # heading = 15 degrees (NNE)
                    drone1_gnc.setCommandTrajectory(75 * utils.knts2fps, 5 * utils.d2r, 15 * utils.d2r)
                    last_command_time = [drone1_gnc.command.time]

                if drone1_gnc.time[-1] >= 2*60 and drone1_gnc.command.time == last_command_time[0]:
                    # Give the aircraft another command at the 2 minute mark
                    # velocity = 60 mph
                    # rate_of_climb = 0 degrees
                    # heading = -90 degrees (W)
                    drone1_gnc.setCommandTrajectory(60 * utils.knts2fps, 0 * utils.d2r, -90 * utils.d2r)
                    last_command_time.append(drone1_gnc.command.time)

                # Generate FANGS output (guidance commands)
                drone1_gnc.getGuidanceCommands()

                # Calculate aircraft state using ideal EOM
                new_state = RBFW(drone1_gnc.Vehicle, drone1_gnc.Thrust[-1], drone1_gnc.Lift[-1], drone1_gnc.alpha_c, drone1_gnc.mu[-1], drone1_gnc.h_c, drone1_gnc.v_BN_W[-1], drone1_gnc.gamma[-1], drone1_gnc.sigma[-1], drone1_gnc.mass[-1], drone1_gnc.airspeed[-1], drone1_gnc.lat[-1], drone1_gnc.lon[-1], drone1_gnc.h[-1], drone1_gnc.time[-1], drone1_gnc.dt)
                mass, v_BN_W, gamma, sigma, lat, lon, h, airspeed, alpha, drag = new_state
                drone1_gnc.updateSystemState(mass=mass, v_BN_W=v_BN_W, gamma=gamma, sigma=sigma, lat=lat, lon=lon, h=h, airspeed=airspeed, alpha=alpha, drag=drag)

        else:
            for ii in range(len(drone1_gnc.time)):
                if ii > 0:
                    bearing.append(track.target_bearing((C2['lat'], C2['lon'], C2['alt']), (drone1_gnc.lat[ii], drone1_gnc.lon[ii], drone1_gnc.h[ii]), method='lla'))
                C2_drone1_tracker.track_target(drone1_gnc.lat[ii], drone1_gnc.lon[ii], drone1_gnc.h[ii], drone1_gnc.time[ii])

    _, (ax_brg, ax_trk) = plt.subplots(2)
    ax_brg.plot(drone1_gnc.time, C2_drone1_tracker.range)
    ax_trk.plot(drone1_gnc.time, C2_drone1_tracker.bearing)
    plt.show()

    utils.plotSim(drone1_gnc, showPlots=True, saveFolder=r'saved_simulations\figures\drone1', filePrefix=f'stopTime')
    if runSim: utils.save_obj(drone1_gnc, r'saved_simulations\drone1_6min.pkl')

    utils.saveTrack(drone1_gnc, r'saved_simulations\tracks\drone1_6min.csv', downsample=50)

    return


if __name__ == '__main__':
    run_C2(stopTime=6*60)  # 6 minutes