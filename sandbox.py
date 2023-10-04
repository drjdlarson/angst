# -*- coding: utf-8 -*-
"""
Created on Thurs September 28 2023

@author: ahspringer

This script is used as a sandbox for testing various functions in the FW_UAV_GNC project.
"""

from FixedWingUAV_Control import *
import numpy as np
import scipy.linalg as la

def exponential_decay(t, y): return -0.5*y

if __name__ == "__main__":
    sol = solve_ivp(exponential_decay, [0, 20], [2, 4, 8])
    print(sol.t, sol.y[0], sol.y[1], sol.y[2])

    a = [1, 2, 3]
    print(np.linalg.norm(a))
    print(la.norm(a))

    print(np.sign(-1))