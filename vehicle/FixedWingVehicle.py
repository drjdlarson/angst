#!/usr/bin/env python
import numpy as np


class FixedWingVehicle:
    """Implements the base vehicle.

    This class allows the user to define the physical model of an aircraft to be used
        as an object of the FixedWingVehicle class.
    The default aircraft parameters, set to np.nan upon unitilization unless specified, are:
        weight_max, weight_min, speed_max, speed_min, Kf, omega_T, omega_L, omega_mu, T_max, K_Lmax, mu_max,
        C_Do, C_Lalpha, alpha_o, wing_area, aspect_ratio, wing_eff
    Other parameters may be defined as required by using the .setAircraftParameters() function,
        passing the new parameter(s) as a key/value pair(s) in a dict.
    """
    units = 'Imperial'
    angles = 'Radians'

    def __init__(self, VehicleParameters):
        """Initalize a Fixed Wing Vehicle object.

        Parameters
        ----------
        VehicleParameters : :dict:`Physical parameters of the aircraft`
            Required keys are:
                weight_max
                weight_min
                speed_max
                speed_min
                Kf
                omega_T
                omega_L
                T_max
                K_Lmax
                mu_max
                C_Do
                C_Lalpha
                alpha_o
                wing_area
                aspect_ratio
                wing_eff
            Optional keys are:
                mdot
        """

        self.weight_max = VehicleParameters['weight_max']  # lbs
        self.weight_min = VehicleParameters['weight_min']  # lbs
        self.speed_max = VehicleParameters['speed_max']  # fps
        self.speed_min = VehicleParameters['speed_min']  # fps
        self.Kf = VehicleParameters['Kf']  # Fuel burning gain value ?
        self.omega_T = VehicleParameters['omega_T']  # rad/s - time constant for engine/airframe response
        self.omega_L = VehicleParameters['omega_L']  # rad/s  - time constant for engine/airframe response
        self.omega_mu = VehicleParameters['omega_mu']  # rad/s  - time constant for engine/airframe response
        self.T_max = VehicleParameters['T_max']  # lbs
        self.K_Lmax = VehicleParameters['K_Lmax']  # lbs/fps^2
        self.mu_max = VehicleParameters['mu_max']  # rad
        self.C_Do = VehicleParameters['C_Do']  # unitless
        self.C_Lalpha = VehicleParameters['C_Lalpha']  # rad^(-1)
        self.alpha_o = VehicleParameters['alpha_o']  # rad
        self.wing_area = VehicleParameters['wing_area']  # ft^2
        self.aspect_ratio = VehicleParameters['aspect_ratio']  # unitless
        self.wing_eff = VehicleParameters['wing_eff']  # ?
        self.mdot = np.nan  # Fuel burn rate, lbs/s
        if 'mdot' in VehicleParameters.keys():
            self.mdot = VehicleParameters['mdot']

    def setAircraftParameters(self, params):
        """ Add or update Parameters of AircraftParams object

        Parameters
        ----------
        params : :dict:`Dict of aircraft parameter(s) to be added/changed`
            Example: {'weight_max': 120, 'max_thrust': 100, 'min_thrust': -20}
                > In this example, weight_max, an existing parameter, is updated to 120.
                > max_thrust and min_thrust, parameters which don't yet exist, are added.
        """
        for param in params.keys():
            self.__setattr__(param, params.get(param))