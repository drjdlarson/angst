# FANGS v1.3

<!-- <a href="https://join.slack.com/t/ngc-goz8665/shared_invite/zt-r01kumfq-dQUT3c95BxEP_fnk4yJFfQ">
<img alt="Join us on Slack" src="https://raw.githubusercontent.com/netlify/netlify-cms/master/website/static/img/slack.png" width="165"/>
</a> -->

![Contributors](https://img.shields.io/github/contributors/ahspringer/FANGS?style=plastic)
![Forks](https://img.shields.io/github/forks/ahspringer/FANGS?style=plastic)
![Stars](https://img.shields.io/github/stars/ahspringer/FANGS?style=plastic)
![Issues](https://img.shields.io/github/issues/ahspringer/FANGS?style=plastic)

## (F)ixed Wing (A)ircraft (N)onlinear (G)uidance (S)ystem

### Description

The algorithms followed for the nonlinear controller are described in the case study for a Nonlinear Aircraft-Performance Simulation by Dr. John Schierman in his Modern Flight Dynamics textbook. This project is a nonlinear controller for a fixed-wing aircraft. The _fangs.GuidanceSystem_ has a baked-in default state estimator utilizing ideal equations of motion, but the user is encouraged to provide their own state solution at each time step using their own state solver or states and rates of their aircraft if this set of algorithms is used for flight.

### FixedWing_NAPS.py

This file is an example run-through of the fixed wing non-linear aircraft performance simulation utilizing the _FANGS.GuidanceSystem_ and a vehicle object of class _FixedWingVehicle_ (or any other object with the required parameters).

### C2_track_build.py

This file is used to simulate one or more drones being launched from a C2 aircraft. The current version (v1.2) simulates 6 drones launched from a stationary (hovering) aircraft. The vehicle guidance system objects are saved as .pkl files to be loaded and used in future simulations, and 3D track files (range, bearing, and elevation angle) are created with simulated noise. These track files are saved as .csv format to be used in the future with the [carbs](https://github.com/drjdlarson/carbs) package for target tracking. The .pkl files, which define each individual drone object, will be used with the [carbs](https://github.com/drjdlarson/carbs) package for drone swarm optimized control and planning.

### Instructions

1. To use the guidance system, first you must create a _FixedWingVehicle_ object class found via _from vehicle.FixedWingVehicle import FixedWingVehicle_. The class documentation will tell you which parameters are required; any other is optional.
2. Next, initiate an object of the _FANGS.GuidanceSystem_ class, passing the _FixedWingVehicle_ object, required transfer function parameters, and initial conditions. You may also specify the time of object creation (this could be launch time, or when the guidance system is "turned on"). If desired, the integration step time can also be set here, but the default value dt=0.01 is probably good enough. Each update to the state parameters will give the user the opportunity to specify a step time, if desired.
3. Set a command trajectory (if desired) using the _setCommandTrajectory()_ function of the _FANGS.GuidanceSystem_ object. If no command trajectory is set, the aircraft will continue along the trajectory defined in the initial conditions.
4. To generate a set of aircraft guidance commands based on the current commanded trajectory from step 3, run the _getGuidanceCommands()_ function of the _FANGS.GuidanceSystem_ object.
5. To update the aircraft's state, run the _updateSystemState()_ function of the _FANGS.GuidanceSystem_ object. This function will progress the aircraft's state one time step forward, using either the user-specified dt parameter to the function or, if no dt is given by the user, using the _FixedWingVehicle_ default dt value, which is the same as the _FANGS.GuidanceSystem_ object default dt value. At this step, the user may supply any, all, or none of the required state parameter values; if any required parameter is not supplied it will be estimated using the _FANGS.GuidanceSystem_ default ideal equations of motion (EOM).
6. Repeat steps 4 and 5 until the aircraft simulation is complete (or the aircraft is done flying).
7. After running the _FANGS.GuidanceSystem_, you can save your guidance object's state to a .pkl binary file by running utils.save_obj() and passing it the _FANGS.GuidanceSystem_ object and a save-to filepath.
8. Finally, you can graph the results of the simulation by running utils.plotSim() and passing it the _FANGS.GuidanceSystem_ object. If you have already run a simulation and saved the resulting _FANGS.GuidanceSystem_ object to a .pkl file, you can load the .pkl file into an object of class _FANGS.GuidanceSystem_ by running _utils.load_obj()_ and passing it the filepath.

### Current Tasks:

1. Use saved drone simulation tracks with [carbs](https://github.com/drjdlarson/carbs) tracking algorithms to evaluate and further understand air-to-air target tracking.
2. Use [carbs](https://github.com/drjdlarson/carbs) and/or [gncpy](https://github.com/drjdlarson/gncpy) to build swarm optimization and command controls.
