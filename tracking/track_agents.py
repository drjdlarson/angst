import sys
import pytest
import time
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import csv
from copy import deepcopy
from contextlib import contextmanager

import scipy.stats as stats

import gncpy.filters as gfilts
import gncpy.dynamics.basic as gdyn
import gncpy.distributions as gdistrib
import carbs.swarm_estimator.tracker as tracker
import serums.models as smodels
from serums.enums import GSMTypes, SingleObjectDistance


global_seed = 8675309
debug_plots = True

_meas_mat = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=np.float64)


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


def _state_mat_fun(t, dt, useless):
    # print('got useless arg: {}'.format(useless))
    return np.array(
        [[1, 0, 0, dt, 0, 0],
         [0, 1, 0, 0, dt, 0],
         [0, 0, 1, 0, 0, dt],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]
    )


def _meas_mat_fun(t, useless):
    # print('got useless arg: {}'.format(useless))
    return _meas_mat


def _multidim_dis_process_noise_mat(p_noise, dim=4):
    # Copied from lines 534 - 551 of gncpy/dynamics/basic.py and edited for 6x6
    ray = np.ones(dim)
    ray[0:int(round(dim/2,0))] = 0
    gamma = ray.reshape(dim, 1)
    return gamma @ [[p_noise**2]] @ gamma.T


def _setup_double_int_kf(dt):
    m_noise = 0.15**2
    p_noise = 0.2

    filt = gfilts.KalmanFilter()
    filt.set_state_model(state_mat_fun=_state_mat_fun)
    filt.set_measurement_model(meas_fun=_meas_mat_fun)
    filt.proc_noise = _multidim_dis_process_noise_mat(p_noise, dim=6)
    # print(f"process noise: {filt.proc_noise}")
    filt.meas_noise = m_noise**2 * np.eye(3)
    # print(f"measurement noise: {filt.meas_noise}")

    return filt


def _setup_phd_double_int_birth():
    mu = [np.array([750.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((6, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1, 1, 1])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [gm0]


def _spherical_to_cartesian(meas):
    # Given meas of [bearing, range, elevation] return [x, y, z] relative to measurement origin
    if '' in meas:
        return [np.nan, np.nan, np.nan]
    else:
        phi = float(meas[0]) * np.pi/180
        rho = float(meas[1])
        theta = (90 - float(meas[2])) * np.pi/180
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        return np.array([x, y, z])


def track_agents_PHD(tracks_file):
    print(f'Tracking agents using PHD filter, no spawning. Tracks pulled from {tracks_file}')
    rng = rnd.default_rng(global_seed)

    dt = 0.5

    # Set up filter
    filt = _setup_double_int_kf(dt)  # Double integrator kalman filter
    state_mat_args = (dt, "test arg")  # ?
    meas_fun_args = ("useless arg",)  # ?

    # Set up agent birth model
    b_model = _setup_phd_double_int_birth()  # Double integrator PHD birth model

    # Set up random finite set base
    RFS_base_args = {"prob_detection": 0.99,
                     "prob_survive": 0.98,
                     "in_filter": filt,
                     "birth_terms": b_model,
                     "clutter_den": 1e-7,
                     "clutter_rate": 1e-7}

    # Set up PHD filter
    phd = tracker.ProbabilityHypothesisDensity(**RFS_base_args)
    phd.gating_on = False
    phd.save_covs = True

    # Initialize set of true_agents and global set of true_agents
    # TODO - create true_agents for OSPA
    true_agents = []
    global_true = []
    measurement_history = []

    # Read CSV file and track agents defined for each time step
    with open(tracks_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        ii = 0
        for measurements in csvReader:
            if ii == 0:
                header = measurements
            else:
                # Each measurement is range, bearing, and elevation in relation to the C2
                tt = float(measurements.pop(0))

                # Prune blank measurement slots
                measurements = [i for i in measurements if i != '']
                # Convert measurements from spherical to cartesian
                measurements = [_spherical_to_cartesian(measurements[i*3:(i+1)*3]).reshape(-1,1) for i in range((len(measurements)+3-1)//3)]
                # Prune measurements
                measurements = [i for i in measurements if i.any() != 0]
                # Add to history
                measurement_history.append(measurements)

                filt_args = {"state_mat_args": state_mat_args}
                phd.predict(tt, filt_args=filt_args)

                filt_args = {"meas_fun_args": meas_fun_args}
                phd.correct(tt, measurements, meas_mat_args={}, est_meas_args={}, filt_args=filt_args)

                phd.cleanup(enable_merge=True)

                # if ii > 10:
                #     break  # Only calculate first 11 iterations for troubleshooting

            ii+=1

    if debug_plots:
        states_plot = phd.plot_states([0, 1])  # I want to view these
        states_plot.savefig("PHD_track_states.png")


def track_agents_CPHD(tracks_file):
    # Not working: _correct_prob_density() seems to have a dimensioning issue
    # ValueError: could not broadcast input array from shape (18,) into shape (6,)
    print(f'Tracking agents using CPHD filter, no spawning. Tracks pulled from {tracks_file}')
    rng = rnd.default_rng(global_seed)

    dt = 0.05

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_phd_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    phd = tracker.CardinalizedPHD(**RFS_base_args)
    phd.gating_on = False

    with open(tracks_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        ii = 0
        for measurements in csvReader:
            if ii == 0:
                header = measurements
            else:
                tt = float(measurements.pop(0))
                print(f'\nTime step: {tt} seconds\nIteration: {ii-1}')

                measurements = [i for i in measurements if i != '']
                measurements = [_spherical_to_cartesian(measurements[i*3:(i+1)*3]) for i in range((len(measurements)+3-1)//3)]
                measurements = [i for i in measurements if i.any() != 0]
                print(f"Measurements (x, y, z): {measurements}")

                filt_args = {"state_mat_args": state_mat_args}
                phd.predict(tt, filt_args=filt_args)

                filt_args = {"meas_fun_args": meas_fun_args}
                phd.correct(tt, measurements, meas_mat_args={}, est_meas_args={}, filt_args=filt_args)

                phd.cleanup()

            ii += 1


if __name__ == "__main__":
    tracks_file = 'radar_measurements.csv'
    with Timer():
        track_agents_PHD(tracks_file)
        # track_agents_CPHD(tracks_file)
