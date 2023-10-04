# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:33:23 2020

@author: ryan4, modified by ahspringer
"""

import numpy as np
import scipy.linalg as la
import time
from contextlib import contextmanager
import wgs84
import pickle
import matplotlib.pyplot as plt


# Constants
r2d = 180.0 / np.pi
d2r = 1 / r2d
mph2fps = 1.46667
fps2mph = 1 / mph2fps
m2feet = 3.28084

# Earth parameters
Re_bar = 6371000 * m2feet
const_density = 2.3769e-3
const_gravity = 32.17


def skew(x):
    x = x.squeeze()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def wind_vector(v_BN_W, gamma, sigma):
    u_inf = v_BN_W * np.cos(gamma) * np.cos(sigma)
    v_inf = v_BN_W * np.cos(gamma) * np.sin(sigma)
    w_inf = v_BN_W * np.sin(gamma)
    return la.norm([u_inf, v_inf, w_inf])


def trapezoidalODESolver(func, a, b, n=10):
    h = float(b - a) / n
    s = 0.0
    s += func(a)/2
    for i in range(1, n):
        s += func(a + i*h)
    s += func(b)/2
    return s*h


def discretize_sys(A, B, dt):
    big_mat = np.hstack((A, B))
    big_mat = np.vstack((big_mat, np.zeros((B.shape[1], big_mat.shape[1]))))
    big_mat = la.expm(big_mat * dt)

    F = big_mat[0:A.shape[0], 0:A.shape[1]]
    G = big_mat[0:B.shape[0], A.shape[1]:]

    return F, G


def eul_angles_to_dcm(phi, theta, psi):
    """
    Return direction cosine matrix (DCM) C_(N<-B) related to 3-2-1 Euler Angles

    Parameters
    ----------
    phi : float
        yaw (Rad).
    theta : float
        pitch (Rad).
    psi : float
        roll (Rad).

    Returns
    -------
    dcm : 3x3 matrix
        direction cosine matrix.

    """
    cphi = np.cos(phi).squeeze()
    sphi = np.sin(phi).squeeze()
    ctheta = np.cos(theta).squeeze()
    stheta = np.sin(theta).squeeze()
    cpsi = np.cos(psi).squeeze()
    spsi = np.sin(psi).squeeze()

    last_rot = np.array([[1, 0, 0],
                         [0, cphi, sphi],
                         [0, -sphi, cphi]])
    mid_rot = np.array([[ctheta, 0, -stheta],
                        [0, 1, 0],
                        [stheta, 0, ctheta]])
    first_rot = np.array([[cpsi, spsi, 0],
                          [-spsi, cpsi, 0],
                          [0, 0, 1]])
    dcm = last_rot @ mid_rot @ first_rot

    return dcm


def parse_icp(r_file, times):
    max_lines = 0
    with open(r_file, 'r') as fin:
        for line in fin:
            if line.strip():
                max_lines = max_lines + 1

    pranges = -np.ones(max_lines)
    trans_times = np.ones(max_lines)
    with open(r_file, 'r') as fin:
        ii = 0
        for line in fin:
            if line.strip():
                cols = line.split()
                pranges[ii] = float(cols[7])
                trans_times[ii] = float(cols[0])
                if times.size == 0:
                    times = np.array([trans_times[ii]])
                elif abs(times - trans_times[ii]).min() > 0.2:
                    times = np.append(times, trans_times[ii])
                ii = ii + 1

    times.sort()
    return (pranges, trans_times), times


def parse_rinex_v2(r_file):
    with open(r_file, 'r') as fin:
        header_finished = False
        nav_line = 0
        max_nav_lines = 8
        nav_msgs = {}
        for line in fin:
            if header_finished:
                if nav_line == 0:
                    prn = line[0:2].strip()
                    year = float(line[3:5])
                    month = float(line[6:8])
                    day = float(line[9:10])
                    hour = float(line[11:14])
                    minute = float(line[14:17])
                    second = float(line[17:22])
                    clock_bias = float(line[22:37]) * 10**float(line[38:41])
                    clock_drift = float(line[41:56]) * 10**float(line[57:60])
                    clock_drift_rate = float(line[60:75]) \
                        * 10**float(line[76:79])
                elif nav_line == 1:
                    iode = float(line[4:18]) * 10**float(line[19:22])
                    crs = float(line[22:37]) * 10**float(line[38:41])
                    delta_n = float(line[41:56]) * 10**float(line[57:60])
                    m0 = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 2:
                    cuc = float(line[4:18]) * 10**float(line[19:22])
                    eccentricity = float(line[22:37]) * 10**float(line[38:41])
                    cus = float(line[41:56]) * 10**float(line[57:60])
                    sqrt_a = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 3:
                    toe = float(line[4:18]) * 10**float(line[19:22])
                    cic = float(line[22:37]) * 10**float(line[38:41])
                    long_ascend_node = float(line[41:56]) \
                        * 10**float(line[57:60])
                    cis = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 4:
                    i0 = float(line[4:18]) * 10**float(line[19:22])
                    crc = float(line[22:37]) * 10**float(line[38:41])
                    arg_peri = float(line[41:56]) * 10**float(line[57:60])
                    omega_dot = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 5:
                    idot = float(line[4:18]) * 10**float(line[19:22])
                    codes = float(line[22:37]) * 10**float(line[38:41])
                    gps_week = float(line[41:56]) * 10**float(line[57:60])
                    p_dta_flag = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 6:
                    sv_acc = float(line[4:18]) * 10**float(line[19:22])
                    sv_health = float(line[22:37]) * 10**float(line[38:41])
                    tgd = float(line[41:56]) * 10**float(line[57:60])
                    iodc = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 7:
                    trans_time = float(line[4:18]) * 10**float(line[19:22])
                    fit_int = float(line[22:37]) * 10**float(line[38:41])
                nav_line += 1

                if nav_line >= max_nav_lines:
                    nav_line = 0
                    if prn in nav_msgs:
                        # print("Adding to PRN: {p:s}".format(p=prn))
                        if prn == "2":
                            tmp = 1
                        nav_msgs[prn].add_vals(sqrt_a=sqrt_a,
                                               eccentricity=eccentricity,
                                               i0=i0,
                                               long_ascend_node=long_ascend_node,
                                               arg_peri=arg_peri, m0=m0,
                                               idot=idot, omega_dot=omega_dot,
                                               delta_n=delta_n,
                                               cuc=cuc, cus=cus, crc=crc,
                                               crs=crs, cic=cic, cis=cis,
                                               toe=toe, time=trans_time,
                                               clock_bias=clock_bias,
                                               clock_drift=clock_drift,
                                               clock_drift_rate=clock_drift_rate)
                    else:
                        print("Found PRN: {p:s}".format(p=prn))
                        nav_msg = \
                            rinex_nav_msg(prn=prn,
                                          sqrt_a=sqrt_a,
                                          eccentricity=eccentricity,
                                          i0=i0,
                                          long_ascend_node=long_ascend_node,
                                          arg_peri=arg_peri, m0=m0,
                                          idot=idot, omega_dot=omega_dot,
                                          delta_n=delta_n, cuc=cuc, cus=cus,
                                          crc=crc, crs=crs, cic=cic, cis=cis,
                                          toe=toe, time=trans_time,
                                          clock_bias=clock_bias,
                                          clock_drift=clock_drift,
                                          clock_drift_rate=clock_drift_rate)
                        nav_msgs[prn] = nav_msg

            elif "end of header" in line.lower():
                header_finished = True
                continue

    return nav_msgs


class rinex_nav_msg(object):
    def __init__(self, **kwargs):
        self.PRN = kwargs['prn']
        self.sqrt_a = np.array([kwargs['sqrt_a']])
        self.eccentricity = np.array([kwargs['eccentricity']])
        self.i0 = np.array([kwargs['i0']])
        self.long_ascend_node = np.array([kwargs['long_ascend_node']])
        self.arg_peri = np.array([kwargs['arg_peri']])
        self.m0 = np.array([kwargs['m0']])
        self.idot = np.array([kwargs['idot']])
        self.omega_dot = np.array([kwargs['omega_dot']])
        self.delta_n = np.array([kwargs['delta_n']])
        self.cos_cor_arg_lat = np.array([kwargs['cuc']])
        self.sin_cor_arg_lat = np.array([kwargs['cus']])
        self.cos_cor_orb_rad = np.array([kwargs['crc']])
        self.sin_cor_orb_rad = np.array([kwargs['crs']])
        self.cos_cor_inc_ang = np.array([kwargs['cic']])
        self.sin_cor_inc_ang = np.array([kwargs['cis']])
        self.toe = np.array([kwargs['toe']])
        self.tol = kwargs.get('tol', 1 * 10**-8)
        self.trans_time = np.array([kwargs['time']])
        self.clock_bias = np.array([kwargs['clock_bias']])
        self.clock_drift = np.array([kwargs['clock_drift']])
        self.clock_drift_rate = np.array([kwargs['clock_drift_rate']])

    def add_vals(self, **kwargs):
        self.sqrt_a = np.hstack((self.sqrt_a, np.array([kwargs['sqrt_a']])))
        self.eccentricity = np.hstack((self.eccentricity,
                                       np.array([kwargs['eccentricity']])))
        self.i0 = np.hstack((self.i0, np.array([kwargs['i0']])))
        self.long_ascend_node = \
            np.hstack((self.long_ascend_node,
                      np.array([kwargs['long_ascend_node']])))
        self.arg_peri = np.hstack((self.arg_peri,
                                   np.array([kwargs['arg_peri']])))
        self.m0 = np.hstack((self.m0, np.array([kwargs['m0']])))
        self.idot = np.hstack((self.idot, np.array([kwargs['idot']])))
        self.omega_dot = np.hstack((self.omega_dot,
                                   np.array([kwargs['omega_dot']])))
        self.delta_n = np.hstack((self.delta_n, np.array([kwargs['delta_n']])))
        self.cos_cor_arg_lat = np.hstack((self.cos_cor_arg_lat,
                                          np.array([kwargs['cuc']])))
        self.sin_cor_arg_lat = np.hstack((self.sin_cor_arg_lat,
                                          np.array([kwargs['cus']])))
        self.cos_cor_orb_rad = np.hstack((self.cos_cor_orb_rad,
                                          np.array([kwargs['crc']])))
        self.sin_cor_orb_rad = np.hstack((self.sin_cor_orb_rad,
                                          np.array([kwargs['crs']])))
        self.cos_cor_inc_ang = np.hstack((self.cos_cor_inc_ang,
                                          np.array([kwargs['cic']])))
        self.sin_cor_inc_ang = np.hstack((self.sin_cor_inc_ang,
                                          np.array([kwargs['cis']])))
        self.toe = np.hstack((self.toe, [kwargs['toe']]))
        self.trans_time = np.hstack([self.trans_time, [kwargs['time']]])
        self.clock_bias = np.hstack([self.clock_bias, [kwargs['clock_bias']]])
        self.clock_drift = np.hstack([self.clock_bias,
                                      [kwargs['clock_drift']]])
        self.clock_drift_rate = np.hstack([self.clock_bias,
                                           [kwargs['clock_drift_rate']]])
        self.ecc_anom = np.inf * np.ones(self.toe.size)

    def calculate_orbit(self, trans_time=None):
        if trans_time is None:
            trans_time = self.toe
        time_len = trans_time.size
        pos_ECEF = np.zeros((time_len, 3))
        for jj in range(0, time_len):
            # find index of last known values based on desired "transmit time"
            # note this assumes things are sorted in increasing order
            ii = np.argmin(np.abs(self.toe - trans_time[jj]))  # 0
#            for kk in range(0, self.toe.size):
#                last_ind = (kk == self.toe.size - 1)
#                if trans_time[jj] <= self.toe[kk] or last_ind:
#                    ii = kk
#                    break
            semimajor = self.sqrt_a[ii]**2
            mean_motion = np.sqrt(wgs84.MU / semimajor**3) + self.delta_n[ii]
            tk = trans_time[jj] - self.toe[ii]
            if tk > 302400:
                tk = tk - 604800
            elif tk < -302400:
                tk = tk + 604800

            mean_anom = self.m0[ii] + mean_motion * tk
            ecc_anom = mean_anom
            ratio = 0
            first_pass = True
            while np.abs(ratio) > self.tol or first_pass:
                first_pass = False
                ecc_anom = ecc_anom - ratio
                err = ecc_anom - self.eccentricity[ii] * np.sin(ecc_anom) \
                    - mean_anom
                der = 1 - self.eccentricity[ii] * np.cos(ecc_anom)
                ratio = err / der
            if np.isinf(self.ecc_anom[ii]):
                self.ecc_anom[ii] = ecc_anom
            tmp = 1 - self.eccentricity[ii] * np.cos(ecc_anom)
            cos_true_anom = (np.cos(ecc_anom) - self.eccentricity[ii]) \
                / tmp
            sin_true_anom = np.sqrt(1 - self.eccentricity[ii]**2) \
                * np.sin(ecc_anom) / tmp
            true_anom = np.arctan2(sin_true_anom, cos_true_anom)
            arg_lat = true_anom + self.arg_peri[ii]
            cor_arg_lat = arg_lat + self.sin_cor_arg_lat[ii] \
                * np.sin(2 * arg_lat) + self.cos_cor_arg_lat[ii] \
                * np.cos(2 * arg_lat)
            rad = semimajor * (1 - self.eccentricity[ii]
                               * np.cos(ecc_anom)) \
                + self.sin_cor_orb_rad[ii] * np.sin(2 * arg_lat) \
                + self.cos_cor_orb_rad[ii] * np.cos(2 * arg_lat)
            cor_inc = self.i0[ii] + self.idot[ii] * tk \
                + self.sin_cor_inc_ang[ii] * np.sin(2 * arg_lat) \
                + self.cos_cor_inc_ang[ii] * np.cos(2 * arg_lat)
            cor_long_ascend = self.long_ascend_node[ii] \
                + (self.omega_dot[ii] - wgs84.EARTH_ROT_RATE) * tk \
                - wgs84.EARTH_ROT_RATE * self.toe[ii]
            xp = rad * np.cos(cor_arg_lat)
            yp = rad * np.sin(cor_arg_lat)
            c_om = np.cos(cor_long_ascend)
            s_om = np.sin(cor_long_ascend)
            ci = np.cos(cor_inc)
            si = np.sin(cor_inc)
            x = xp * c_om - yp * ci * s_om
            y = xp * s_om + yp * ci * c_om
            z = yp * si
            pos_ECEF[jj, :] = np.array([x, y, z])

        return pos_ECEF


def ecef_to_lla(xyz):
    lon = np.arctan2(xyz[1], xyz[0])
    p = np.sqrt(xyz[0]**2 + xyz[1]**2)
    E = np.sqrt(wgs84.EQ_RAD**2 - wgs84.POL_RAD**2)
    F = 54 * (wgs84.POL_RAD * xyz[2])**2
    G = p**2 + (1 - wgs84.ECCENTRICITY**2) \
        * (xyz[2]**2) - (wgs84.ECCENTRICITY * E)**2
    c = wgs84.ECCENTRICITY**4 * F * p**2 / G**3
    s = (1 + c + np.sqrt(c**2 + 2 * c))**(1 / 3)
    P = (F / (3 * G**2)) / (s + 1 / s + 1)**2
    Q = np.sqrt(1 + 2 * wgs84.ECCENTRICITY**4 * P)
    k1 = -P * wgs84.ECCENTRICITY**2 * p / (1 + Q)
    k2 = 0.5 * wgs84.EQ_RAD**2 * (1 + 1 / Q)
    k3 = -P * (1 - wgs84.ECCENTRICITY**2) * xyz[2]**2 / (Q * (1 + Q))
    k4 = -0.5 * P * p**2
    k5 = p - wgs84.ECCENTRICITY**2 * (k1 + np.sqrt(k2 + k3 + k4))
    U = np.sqrt(k5**2 + xyz[2]**2)
    V = np.sqrt(k5**2 + (1 - wgs84.ECCENTRICITY**2) * xyz[2]**2)
    alt = U * (1 - wgs84.POL_RAD**2 / (wgs84.EQ_RAD * V))
    z0 = wgs84.POL_RAD**2 * xyz[2] / (wgs84.EQ_RAD * V)
    ep = wgs84.EQ_RAD / wgs84.POL_RAD * wgs84.ECCENTRICITY
    lat = np.arctan((xyz[2] + z0 * ep**2) / p)
    return lat, lon, alt


def lla_to_ecef(pos):
    alt = pos[2]
    re = wgs84.calc_ew_rad(pos[0])
    c_lat = np.cos(pos[0]).squeeze()
    s_lat = np.sin(pos[0]).squeeze()
    c_lon = np.cos(pos[1]).squeeze()
    s_lon = np.sin(pos[1]).squeeze()
    return np.array([(re + alt) * c_lat * c_lon,
                     (re + alt) * c_lat * s_lon,
                     ((1 - wgs84.ECCENTRICITY**2) * re + alt) * s_lat])


def lla_to_ned(ref_LLA, pos_LLA):
    c_lat = np.cos(ref_LLA[0]).squeeze()
    s_lat = np.sin(ref_LLA[0]).squeeze()
    c_lon = np.cos(ref_LLA[1]).squeeze()
    s_lon = np.sin(ref_LLA[1]).squeeze()
    R = np.array([[-s_lat * c_lon, -s_lon, -c_lat * c_lon],
                  [-s_lat * s_lon, c_lon, -c_lat * s_lon],
                  [c_lat, 0, -s_lat]])
    ref_E = lla_to_ecef(ref_LLA)
    pos_E = lla_to_ecef(pos_LLA)
    return R.T @ (pos_E - ref_E)


def save_obj(obj, filepath):
    with open(filepath, 'wb') as savepath:
        pickle.dump(obj, savepath)


def load_obj(filepath):
    with open(filepath, 'rb') as loadpath:
        return pickle.load(loadpath)


def plotSim(simulation_guidance_object):
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

    # Plot results (airspeed)
    fig_airspd, ax_airspd = plt.subplots(1)
    v_airspeed_mph = [x*fps2mph for x in acft_Guidance.airspeed]
    ax_airspd.plot(acft_Guidance.time, v_airspeed_mph)
    ax_airspd.set_title('Airspeed Response')
    ax_airspd.set_xlabel('Time (s)')
    ax_airspd.set_ylabel('Airspeed (mph)')
    ax_airspd.grid(visible='True')

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
    ax_aoa.set_ylabel('Heading (deg)')
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

    # Plot results (forces)
    fig_forces, ax_forces = plt.subplots(1)
    ax_forces.plot(acft_Guidance.time, acft_Guidance.Thrust, label='Thrust Response')
    ax_forces.plot(acft_Guidance.time, acft_Guidance.drag, label='Drag Response')
    ax_forces.set_title('Forces')
    ax_forces.set_xlabel('Time (s)')
    ax_forces.set_ylabel('Forces (lbs)')
    ax_forces.legend()
    ax_forces.grid(visible='True')

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

    # Show plots
    plt.show()


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