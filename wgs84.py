# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:53:34 2020

@author: ryan4
"""
import numpy as np


MU = 3.986005 * 10**14  # m^3/s^2
SPEED_OF_LIGHT = 2.99792458 * 10**8  # m/s
EARTH_ROT_RATE = 7.2921151467 * 10**-5  # rad/s
PI = 3.1415926535898
FLATTENING = 1 / 298.257223563
ECCENTRICITY = np.sqrt(FLATTENING * (2 - FLATTENING))
EQ_RAD = 6378137  # m
POL_RAD = EQ_RAD * (1 - FLATTENING)
GRAVITY = 9.7803253359


def calc_earth_rate(lat):
    """
    Earth rotation rate expressed in navigation frame w_(N,I<-E)
    ECI frame: International Celestial Reference System??

    Parameters
    ----------
    lat : float
        latitude of position in radians.

    Returns
    -------
    float
        Earth rotation rate in Earth-Centered Inertial frame

    """
    return EARTH_ROT_RATE * np.array([np.cos(lat), 0, -np.sin(lat)])


def calc_transport_rate(v_N, alt, lat):
    """
    Transport rate expressed in navigation frame w_(N,E<-N)

    Parameters
    ----------
    v_N : 3x1 array of floats
        Velocity in navigation frame (x, y, z)
    alt : float
        altitude
    lat : float
        latitude of position in radians.

    Returns
    -------
    3x1 array of floats
        Transport rate omega_N,E<-N.

    """
    rn = calc_ns_rad(lat)
    re = calc_ew_rad(lat)
    return np.array([v_N[1] / (re + alt),
                     -v_N[0] / (rn + alt),
                     -v_N[1] * np.tan(lat) / (re + alt)])


def calc_ns_rad(lat):
    return EQ_RAD * (1 - ECCENTRICITY**2) / (1 - ECCENTRICITY**2
                                             * np.sin(lat)**2)**1.5


def calc_ew_rad(lat):
    return EQ_RAD / np.sqrt(1 - ECCENTRICITY**2 * np.sin(lat)**2)


def calc_gravity(lat, alt):
    frac = alt / EQ_RAD
    g0 = GRAVITY / np.sqrt(1 - FLATTENING * (2 - FLATTENING)
                           * np.sin(lat)**2) * (1 + 0.0019311853
                                                * np.sin(lat)**2)
    ch = 1 - 2 * (1 + FLATTENING + (EQ_RAD**3 * (1 - FLATTENING)
                  * EARTH_ROT_RATE**2) / MU) * frac + 3 * frac**2
    return np.array([[0], [0], [ch * g0]])
