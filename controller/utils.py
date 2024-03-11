#!/usr/bin/env python
""" Utility functions for FANGS.py script

    v1.4.0 Notes:
        1. Updated how plots are made. Added the following commands:
            plotGroundspeed()
            plotAirspeed()
            plotFlightPathAngle()
            plotHeading()
            plotAngleOfAttack()
            plotHeight()
            plotCoordinates()
            plotThrustDrag()
            plotLift()
            plotBankAngle()
        2. Added KML writer tool:
            writeKMLfromObj()
        3. Added miles2feet constant
"""
__author__ = "Alex Springer"
__version__ = "1.4.0"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Production"

import numpy as np
import time
import math
import pickle
import simplekml
import scipy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as et
from contextlib import contextmanager
from matplotlib.ticker import FormatStrFormatter

# Constants
r2d = 180.0 / np.pi
d2r = 1 / r2d
mph2fps = 1.46667
fps2mph = 1 / mph2fps
m2feet = 3.28084
knts2fps = 1.68781
miles2feet = 5280

# Earth parameters
Re_bar = 6371000 * m2feet
const_density = 2.3769e-3
const_gravity = 32.17


def get_point_at_distance(lat1, lon1, dist, bearing):
    """
    lat: initial latitude, in radians
    lon: initial longitude, in radians
    d: target distance from initial in meters
    bearing: (true) heading in radians

    Returns new lat/lon coordinate {dist}m from initial, in degrees
    """
    lat2 = np.arcsin(np.sin(lat1) * np.cos(dist/Re_bar) + np.cos(lat1) * np.sin(dist/Re_bar) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(dist/Re_bar) * np.cos(lat1),
        np.cos(dist/Re_bar) - np.sin(lat1) * np.sin(lat2)
    )
    return lat2, lon2


def get_bearing(lat1, long1, lat2, long2, units="Radians"):
    if units == "Degrees":
        lat1 = math.radians(lat1)
        long1 = math.radians(long1)
        lat2 = math.radians(lat2)
        long2 = math.radians(long2)
    dLon = (long2 - long1)
    x = math.cos(lat2) * math.sin(dLon)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    brng = np.arctan2(x,y)
    if units == "Degrees":
        brng = np.degrees(brng)
    return brng


def get_distance(lat1, long1, lat2, long2, units="Radians"):
    if units == "Degrees":
        lat1 = math.radians(lat1)
        long1 = math.radians(long1)
        lat2 = math.radians(lat2)
        long2 = math.radians(long2)
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = Re_bar * c
    return distance


def wind_vector(v_BN_W, gamma, sigma):
    u_inf = v_BN_W * np.cos(gamma) * np.cos(sigma)
    v_inf = v_BN_W * np.cos(gamma) * np.sin(sigma)
    w_inf = v_BN_W * np.sin(gamma)
    return la.norm([u_inf, v_inf, w_inf])


def save_obj(obj, filepath):
    with open(filepath, 'wb') as savepath:
        pickle.dump(obj, savepath)


def load_obj(filepath):
    with open(filepath, 'rb') as loadpath:
        return pickle.load(loadpath)


def gnc_to_csv(obj, filepath, downsample=1):
    lat = [x * r2d for x in obj.lat[::downsample]]
    lon = [x * r2d for x in obj.lon[::downsample]]
    df = pd.DataFrame.from_dict({'time':obj.time[::downsample],
                                 'lat':lat,
                                 'lon':lon,
                                 'altitude':obj.h[::downsample],
                                 'groundspeed':obj.v_BN_W[::downsample],
                                 'gamma':obj.gamma[::downsample],
                                 'sigma':obj.sigma[::downsample],
                                 'airspeed':obj.airspeed[::downsample],
                                 'lift':obj.Lift[::downsample],
                                 'thrust':obj.Thrust[::downsample],
                                 'angle_of_attack':obj.alpha[::downsample],
                                 'command_vel':obj.command.v_BN_W_history[::downsample],
                                 'command_gamma':obj.command.gamma_history[::downsample],
                                 'command_sigma':obj.command.sigma_history[::downsample],
                                 'command_airspeed':obj.command.airspeed_history[::downsample],
                                 'command_lift': obj.Lc[::downsample],
                                 'command_thrust':obj.Tc[::downsample],
                                 'command_angle_of_attack':obj.alpha_c[::downsample]})
    df.to_csv(filepath, index=False)


def plotSim(simulation_guidance_object, saveFolder=None, filePrefix=None, showPlots=False, plotsToMake=0):
    acft_Guidance = simulation_guidance_object

    # Plot results (groundspeed)
    if plotsToMake == 0 or 'Groundspeed' in plotsToMake:
        fig_gndspd, ax_gndspd = plotGroundspeed(acft_Guidance)

    # Plot results (airspeed)
    if plotsToMake == 0 or 'Airspeed' in plotsToMake:
        fig_airspd, ax_airspd = plotAirspeed(acft_Guidance)

    # Plot results (flight path angle)
    if plotsToMake == 0 or 'FlightPath' in plotsToMake:
        fig_fltpth, ax_fltpth = plotFlightPathAngle(acft_Guidance)

    # Plot results (heading)
    if plotsToMake == 0 or 'Heading' in plotsToMake:
        fig_hdg, ax_hdg = plotHeading(acft_Guidance)

    # Plot results (angle of attack)
    if plotsToMake == 0 or 'AoA' in plotsToMake:
        fig_aoa, ax_aoa = plotAngleOfAttack(acft_Guidance)

    # Plot results (height)
    if plotsToMake == 0 or 'Height' in plotsToMake:
        fig_hgt, ax_hgt = plotHeight(acft_Guidance)

    # Plot results (geodetic coordinates)
    if plotsToMake == 0 or 'Coordinates' in plotsToMake:
        fig_coords, ax_coords = plotCoordinates(acft_Guidance)

    # Plot results (thrust and drag)
    if plotsToMake == 0 or 'Thrust' in plotsToMake:
        fig_forces, ax_forces = plotThrustDrag(acft_Guidance)

    # Plot results (lift)
    if plotsToMake == 0 or 'Lift' in plotsToMake:
        fig_lift, ax_lift = plotLift(acft_Guidance)

    # Plot results (bank angle)
    if plotsToMake == 0 or 'Bank' in plotsToMake:
        fig_bank, ax_bank = plotBankAngle(acft_Guidance)

    # Save plots
    if saveFolder is not None:
        if filePrefix is None:
            filePrefix = ''
        with Timer('save_figures'):
            fig_gndspd.savefig(saveFolder+'\\'+filePrefix+'gndspd.png')
            fig_airspd.savefig(saveFolder+'\\'+filePrefix+'airspd.png')
            fig_fltpth.savefig(saveFolder+'\\'+filePrefix+'fltpth.png')
            fig_hdg.savefig(saveFolder+'\\'+filePrefix+'hdg.png')
            fig_aoa.savefig(saveFolder+'\\'+filePrefix+'aoa.png')
            fig_hgt.savefig(saveFolder+'\\'+filePrefix+'hgt.png')
            fig_coords.savefig(saveFolder+'\\'+filePrefix+'coords.png')
            fig_forces.savefig(saveFolder+'\\'+filePrefix+'forces.png')
            fig_lift.savefig(saveFolder+'\\'+filePrefix+'lift.png')
            fig_bank.savefig(saveFolder+'\\'+filePrefix+'bank.png')

    # Show plots
    if showPlots: plt.show()
    plt.close()


def plotGroundspeed(acft_Guidance):
    fig_gndspd, ax_gndspd = plt.subplots(1)
    v_BN_W_mph = [x for x in acft_Guidance.v_BN_W]
    v_BN_W_c_mph = [x for x in acft_Guidance.command.v_BN_W_history]
    ax_gndspd.plot(acft_Guidance.time, v_BN_W_mph, label='Response')
    ax_gndspd.plot(acft_Guidance.time, v_BN_W_c_mph, label='Commanded')
    ax_gndspd.set_title('Gndspeed Response')
    ax_gndspd.set_xlabel('Time (s)')
    ax_gndspd.set_ylabel('Groundspeed (fps)')
    ax_gndspd.legend()
    ax_gndspd.grid(visible='True')
    ax_gndspd.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    return fig_gndspd, ax_gndspd


def plotAirspeed(acft_Guidance):
    fig_airspd, ax_airspd = plt.subplots(1)
    v_airspeed_mph = [x for x in acft_Guidance.airspeed]
    v_airspeed_c_mph = [x for x in acft_Guidance.command.airspeed_history]
    ax_airspd.plot(acft_Guidance.time, v_airspeed_mph, label='Response')
    ax_airspd.plot(acft_Guidance.time, v_airspeed_c_mph, label='Commanded')
    ax_airspd.set_title('Airspeed Response')
    ax_airspd.set_xlabel('Time (s)')
    ax_airspd.set_ylabel('Airspeed (fps)')
    ax_airspd.legend()
    ax_airspd.grid(visible='True')
    ax_airspd.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    return fig_airspd, ax_airspd


def plotFlightPathAngle(acft_Guidance):
    fig_fltpth, ax_fltpth = plt.subplots(1)
    gamma_deg = [x*r2d for x in acft_Guidance.gamma]
    gamma_c_deg = [x*r2d for x in acft_Guidance.command.gamma_history]
    ax_fltpth.plot(acft_Guidance.time, gamma_deg, label='Response')
    ax_fltpth.plot(acft_Guidance.time, gamma_c_deg, label='Commanded')
    ax_fltpth.set_title('Flight Path Angle Response')
    ax_fltpth.set_xlabel('Time (s)')
    ax_fltpth.set_ylabel('Flight Path Angle (deg)')
    ax_fltpth.legend()
    ax_fltpth.grid(visible='True')
    return fig_fltpth, ax_fltpth


def plotHeading(acft_Guidance):
    fig_hdg, ax_hdg = plt.subplots(1)
    sigma_deg = [x*r2d for x in acft_Guidance.sigma]
    sigma_c_deg = [x*r2d for x in acft_Guidance.command.sigma_history]
    ax_hdg.plot(acft_Guidance.time, sigma_deg, label='Response')
    ax_hdg.plot(acft_Guidance.time, sigma_c_deg, label='Commanded')
    ax_hdg.set_title('Heading Response')
    ax_hdg.set_xlabel('Time (s)')
    ax_hdg.set_ylabel('Heading (deg)')
    ax_hdg.legend()
    ax_hdg.grid(visible='True')
    return fig_hdg, ax_hdg


def plotAngleOfAttack(acft_Guidance):
    fig_aoa, ax_aoa = plt.subplots(1)
    alpha_deg = [x*r2d for x in acft_Guidance.alpha]
    alpha_c_deg = [x*r2d for x in acft_Guidance.alpha_c]
    ax_aoa.plot(acft_Guidance.time, alpha_c_deg, label='Commanded')
    ax_aoa.plot(acft_Guidance.time, alpha_deg, label='Response')
    ax_aoa.set_title('Angle of Attack')
    ax_aoa.set_xlabel('Time (s)')
    ax_aoa.set_ylabel('Angle of Attack (deg)')
    ax_aoa.legend()
    ax_aoa.grid(visible='True')
    return fig_aoa, ax_aoa


def plotHeight(acft_Guidance):
    fig_hgt, ax_hgt = plt.subplots(1)
    ax_hgt.plot(acft_Guidance.time, acft_Guidance.h_c, label='Commanded')
    ax_hgt.plot(acft_Guidance.time, acft_Guidance.h, label='Response')
    ax_hgt.set_title('Height')
    ax_hgt.set_xlabel('Time (s)')
    ax_hgt.set_ylabel('Height (ft)')
    ax_hgt.legend()
    ax_hgt.grid(visible='True')
    return fig_hgt, ax_hgt


def plotCoordinates(acft_Guidance):
    fig_coords, ax_coords = plt.subplots(1)
    if type(acft_Guidance) is dict:
        for acft, obj in acft_Guidance.items():
            _plotCoordinates(obj, ax_coords, legendkey=acft)
    else:
        _plotCoordinates(acft_Guidance, ax_coords)
    return fig_coords, ax_coords


def _plotCoordinates(acft_Guidance, ax_coords, legendkey=None):
    lat_deg = [x*r2d for x in acft_Guidance.lat]
    lon_deg = [x*r2d for x in acft_Guidance.lon]
    target_coords = [(x*r2d, y*r2d) for (x,y) in acft_Guidance.command.waypoint_history]
    line, = ax_coords.plot(lat_deg, lon_deg)
    if legendkey is not None:
        line.set_label(legendkey)
    for target in target_coords:
        ax_coords.plot(target[0], target[1], 'rx')
    ax_coords.set_title('Geodetic Coordinates')
    ax_coords.set_xlabel('Latitude (deg)')
    ax_coords.set_ylabel('Longitude (deg)')
    ax_coords.grid(visible='True')


def plotThrustDrag(acft_Guidance):
    fig_forces, ax_forces = plt.subplots(1)
    ax_forces.plot(acft_Guidance.time, acft_Guidance.Tc, label='Thrust Command')
    ax_forces.plot(acft_Guidance.time, acft_Guidance.Thrust, label='Thrust Response')
    ax_forces.plot(acft_Guidance.time, acft_Guidance.drag, label='Drag Response')
    ax_forces.set_title('Thrust and Drag')
    ax_forces.set_xlabel('Time (s)')
    ax_forces.set_ylabel('Force (lbs)')
    ax_forces.legend()
    ax_forces.grid(visible='True')
    return fig_forces, ax_forces


def plotLift(acft_Guidance):
    fig_lift, ax_lift = plt.subplots(1)
    ax_lift.plot(acft_Guidance.time, acft_Guidance.Lc, label='Lift Command')
    ax_lift.plot(acft_Guidance.time, acft_Guidance.Lift, label='Lift Response')
    ax_lift.set_title('Lift')
    ax_lift.set_xlabel('Time (s)')
    ax_lift.set_ylabel('Force (lbs)')
    ax_lift.legend()
    ax_lift.grid(visible='True')
    return fig_lift, ax_lift


def plotBankAngle(acft_Guidance):
    fig_bank, ax_bank = plt.subplots(1)
    mu_deg = [x*r2d for x in acft_Guidance.mu]
    ax_bank.axhline(y=acft_Guidance.Vehicle.mu_max*r2d, color='k', linestyle='--')
    ax_bank.axhline(y=acft_Guidance.Vehicle.mu_max*r2d*-1, color='k', linestyle='--')
    ax_bank.plot(acft_Guidance.time, mu_deg)
    ax_bank.set_title('Bank Angle')
    ax_bank.set_xlabel('Time (s)')
    ax_bank.set_ylabel('Bank Angle (deg)')
    ax_bank.grid(visible='True')
    return fig_bank, ax_bank


def read_kml_coordinates(filepath):
    tree = et.parse(filepath)
    root = tree.getroot()
    coordinates_path = [elem.tag for elem in tree.iter() if 'coordinates' in elem.tag]
    lat = []
    lon = []
    alt = []
    for coordinates in root.findall('.//'+coordinates_path[0]):
        original_coordinates = coordinates.text.replace('\t', '').replace('\n', '')
        original_coordinates = original_coordinates.replace('0 -', '0, -').split(',')
    ii = 0
    for coord in original_coordinates:
        if ii == 0:
            lon.append(float(coord))
        elif ii == 1:
            lat.append(float(coord))
        else:
            alt.append(float(coord))
        ii += 1
        if ii == 3: ii = 0
    return {'lat':lat, 'lon':lon, 'alt':alt}


def writeKMLfromObj(GuidanceSystemObject, saveFolder=None, noise=False, downsample=50):
    if noise is not False:
        print('Noise not currently implemented in this build.')
    
    if saveFolder is None:
        saveFolder = '.'
    lat = [l * r2d for l in GuidanceSystemObject.lat[::downsample]]
    lon = [l * r2d for l in GuidanceSystemObject.lon[::downsample]]
    alt = GuidanceSystemObject.h[::downsample]
    target_coords = [(x*r2d, y*r2d) for (x,y) in GuidanceSystemObject.command.waypoint_history]

    kml = simplekml.Kml()
    lin = kml.newlinestring(name=f'agent_{GuidanceSystemObject.Vehicle.aircraftID}',
                            description='The ground track of an aerial agent using FANGS',
                            coords=zip(lon, lat, alt))
    kml.save(f'{saveFolder}\\agent_{GuidanceSystemObject.Vehicle.aircraftID}.kml')

    ii = 1
    print(target_coords)
    if len(target_coords) > 0:
        for tgt in target_coords:
            print(tgt[::-1])
            # kml = simplekml.Kml()
            pt = kml.newpoint(name=f'agent_{GuidanceSystemObject.Vehicle.aircraftID}_target_{ii}',
                            description='A waypoint defined by the FANGS user for drone fly-over',
                            coords = [tgt[::-1]])
            kml.save(f'{saveFolder}\\agent_{GuidanceSystemObject.Vehicle.aircraftID}_target_{ii}.kml')
    

@contextmanager
def Timer(taskName=None):
    t0 = time.time()
    timeType = 'seconds'
    try: yield
    finally:
        timeToRun = time.time() - t0
        if timeToRun > 60:
                timeToRun = timeToRun/60
                timeType = 'minutes'
        if taskName is None:
            str2print = f'Elapsed time: {round(timeToRun, 2)} {timeType}'
        else:
            str2print = f'[{taskName}] Elapsed time: {round(timeToRun, 2)} {timeType}'
        print(str2print)