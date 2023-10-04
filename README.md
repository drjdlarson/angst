# FW_UAV_GNC

![Contributors](https://img.shields.io/github/contributors/larymak/Python-project-Scripts?style=plastic)
![Forks](https://img.shields.io/github/forks/ahspringer/FW_UAV_GNC)
![Stars](https://img.shields.io/github/stars/ahspringer/FW_UAV_GNC)
![Licence](https://img.shields.io/github/license/ahspringer/FW_UAV_GNC)
![Issues](https://img.shields.io/github/issues/ahspringer/FW_UAV_GNC)

## Fixed Wing Guidance, Navigation, and Control - Nonlinear.

### Description

This project is a nonlinear controller for a fixed-wing aircraft. It includes the main python script FixedWingUAV_Control.py, a developer sandbox, a wgs84.py script for use in calculating Earth approximations (not yet implemented), and a utils.py script which contains useful functions. 

### FixedWingUAV_Control.py

This file contains the main FixedWingVehicle object class, FW_NLPerf_GuidanceSystem object class, and a simple function for running the FW_NLPerf_GuidanceSystem over a designated amount of time.

#### Current problems

The FW_NLPerf_GuidanceSystem algorithms are not currently functioning as required. Troubleshooting is on-going.

### Instructions

1. To use the guidance system, first you must create a FixedWingVehicle object class. The class documentation will tell you which settings are required; all others are optional.
2. Next, initiate an object of the FW_NLPerf_GuidanceSystem class, passing the FixedWingVehicle object, required transfer function parameters, and initial conditions. You may also specify the time of object creation (this could be launch time, or when the guidance system is "turned on"). If desired, the integration step time can also be set here, but the default value dt=0.01 is likely good enough.
3. Set a command trajectory (if desired) using the setCommandTrajectory() function of the FW_NLPerf_GuidanceSystem object. If no command trajectory is set, the aircraft will continue along the trajectory defined in the initial conditions.
4. For each time step, run the stepTime() function of the FW_NLPerf_GuidanceSystem object to progress it forward. You can specify a time step if desired, or use the current FW_NLPerf_GuidanceSystem object dt value (if unchanged, it will be the default 0.01 seconds).
5. After running the simulation, you can save your simulation state to a .pkl binary file by running utils.save_obj() and passing it the FW_NLPerf_GuidanceSystem object and a filepath.
6. Finally, you can graph the results of the simulation by running utils.plotSim() and passing it the FW_NLPerf_GuidanceSystem object.
    NOTE: If you have already run a simulation and saved the resulting FW_NLPerf_GuidanceSystem object to a .pkl file, you can load the .pkl file into an object of class FW_NLPerf_GuidanceSystem by running utils.load_obj() and passing it the filepath.