#!/usr/bin/env python
""" Utility functions for FANGS.py script

    v1.2.0 Notes:
        1. Added get_bearing
"""
__author__ = "Alex Springer"
__version__ = "1.2.0"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Production"

import numpy as np
import time
import math
import pickle
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


def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    return brng


def get_distance(lat1, long1, lat2, long2):
    dlon = math.radians(long2) - math.radians(long1)
    dlat = math.radians(lat2) - math.radians(lat1)
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


def plotSim(simulation_guidance_object, saveFolder=None, filePrefix=None, showPlots=False):
    acft_Guidance = simulation_guidance_object

    # Plot results (groundspeed)
    fig_gndspd, ax_gndspd = plt.subplots(1)
    v_BN_W_mph = [x*fps2mph for x in acft_Guidance.v_BN_W]
    v_BN_W_c_mph = [x*fps2mph for x in acft_Guidance.command.v_BN_W_history]
    ax_gndspd.plot(acft_Guidance.time, v_BN_W_mph, label='Response')
    ax_gndspd.plot(acft_Guidance.time, v_BN_W_c_mph, label='Commanded')
    ax_gndspd.set_title('Gndspeed Response')
    ax_gndspd.set_xlabel('Time (s)')
    ax_gndspd.set_ylabel('Groundspeed (mph)')
    ax_gndspd.legend()
    ax_gndspd.grid(visible='True')
    ax_gndspd.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Plot results (airspeed)
    fig_airspd, ax_airspd = plt.subplots(1)
    v_airspeed_mph = [x*fps2mph for x in acft_Guidance.airspeed]
    # v_airspeed_c_mph = [x*fps2mph for x in acft_Guidance.command.airspeed_history]
    ax_airspd.plot(acft_Guidance.time, v_airspeed_mph, label='Response')
    # ax_airspd.plot(acft_Guidance.time, v_airspeed_c_mph, label='Commanded')
    ax_airspd.set_title('Airspeed Response')
    ax_airspd.set_xlabel('Time (s)')
    ax_airspd.set_ylabel('Airspeed (mph)')
    ax_airspd.legend()
    ax_airspd.grid(visible='True')
    ax_airspd.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Plot results (flight path angle)
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

    # Plot results (heading)
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

    # Plot results (angle of attack)
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

    # Plot results (height)
    fig_hgt, ax_hgt = plt.subplots(1)
    ax_hgt.plot(acft_Guidance.time, acft_Guidance.h_c, label='Commanded')
    ax_hgt.plot(acft_Guidance.time, acft_Guidance.h, label='Response')
    ax_hgt.set_title('Height')
    ax_hgt.set_xlabel('Time (s)')
    ax_hgt.set_ylabel('Height (ft)')
    ax_hgt.legend()
    ax_hgt.grid(visible='True')

    # Plot results (geodetic coordinates)
    fig_coords, ax_coords = plt.subplots(1)
    lat_deg = [x*r2d for x in acft_Guidance.lat]
    lon_deg = [x*r2d for x in acft_Guidance.lon]
    ax_coords.plot(lat_deg, lon_deg)
    ax_coords.set_title('Geodetic Coordinates')
    ax_coords.set_xlabel('Latitude (deg)')
    ax_coords.set_ylabel('Longitude (deg)')
    ax_coords.grid(visible='True')

    # Plot results (thrust and drag)
    fig_forces, ax_forces = plt.subplots(1)
    ax_forces.plot(acft_Guidance.time, acft_Guidance.Tc, label='Thrust Command')
    ax_forces.plot(acft_Guidance.time, acft_Guidance.Thrust, label='Thrust Response')
    ax_forces.plot(acft_Guidance.time, acft_Guidance.drag, label='Drag Response')
    ax_forces.set_title('Thrust and Drag')
    ax_forces.set_xlabel('Time (s)')
    ax_forces.set_ylabel('Force (lbs)')
    ax_forces.legend()
    ax_forces.grid(visible='True')

    # Plot results (lift)
    fig_lift, ax_lift = plt.subplots(1)
    ax_lift.plot(acft_Guidance.time, acft_Guidance.Lc, label='Lift Command')
    ax_lift.plot(acft_Guidance.time, acft_Guidance.Lift, label='Lift Response')
    ax_lift.set_title('Lift')
    ax_lift.set_xlabel('Time (s)')
    ax_lift.set_ylabel('Force (lbs)')
    ax_lift.legend()
    ax_lift.grid(visible='True')

    # Plot results (bank angle)
    fig_bank, ax_bank = plt.subplots(1)
    mu_deg = [x*r2d for x in acft_Guidance.mu]
    ax_bank.axhline(y=acft_Guidance.Vehicle.mu_max*r2d, color='k', linestyle='--')
    ax_bank.axhline(y=acft_Guidance.Vehicle.mu_max*r2d*-1, color='k', linestyle='--')
    ax_bank.plot(acft_Guidance.time, mu_deg)
    ax_bank.set_title('Bank Angle')
    ax_bank.set_xlabel('Time (s)')
    ax_bank.set_ylabel('Bank Angle (deg)')
    ax_bank.grid(visible='True')

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


def write_kml_coordinates(original_filepath, new_filepath, new_coordinates):
    # Format the coordinates how the kml wants them
    # TODO - make this more user-friendly
    outstring = '\n\t\t\t\t'
    for lat, lon, alt in zip(new_coordinates.get('lat'), new_coordinates.get('lon'), new_coordinates.get('alt')):
        outstring = f'{outstring} {lon},{lat},{alt}'
    outstring = f'{outstring}  \n\t\t\t'

    tree = et.parse(original_filepath)
    root = tree.getroot()
    coordinates_path = [elem.tag for elem in tree.iter() if 'coordinates' in elem.tag]
    for coordinates in root.findall('.//'+coordinates_path[0]):
        coordinates.text = outstring
    
    tree.write(new_filepath)
    

@contextmanager
def Timer(taskName=None):
    t0 = time.time()
    try: yield
    finally:
        if taskName is None:
            str2print = f'Elapsed time: {time.time() - t0} seconds'
        else:
            str2print = f'[{taskName}] Elapsed time: {time.time() - t0} seconds'
        print(str2print)