# FANGS v2.1

<!-- <a href="https://join.slack.com/t/ngc-goz8665/shared_invite/zt-r01kumfq-dQUT3c95BxEP_fnk4yJFfQ">
<img alt="Join us on Slack" src="https://raw.githubusercontent.com/netlify/netlify-cms/master/website/static/img/slack.png" width="165"/>
</a> -->

![Contributors](https://img.shields.io/github/contributors/ahspringer/FANGS?style=plastic)
![Forks](https://img.shields.io/github/forks/ahspringer/FANGS?style=plastic)
![Stars](https://img.shields.io/github/stars/ahspringer/FANGS?style=plastic)
![Issues](https://img.shields.io/github/issues/ahspringer/FANGS?style=plastic)

## (F)ixed Wing (A)ircraft (N)onlinear (G)uidance (S)ystem v2.1

### Description

The algorithms followed for the nonlinear controller are described in the case study for a Nonlinear Aircraft-Performance Simulation by Dr. John Schierman in his Modern Flight Dynamics textbook. This project is a nonlinear controller for a fixed-wing aircraft. The _fangs.GuidanceSystem_ has a baked-in default state estimator utilizing ideal equations of motion, but the user is encouraged to provide their own state solution at each time step using their own state solver or states and rates of their aircraft if this set of algorithms is used for real flight.

Version notes: v2.0.0
- This version added changes to the API to allow the user to input an "absolute" command: Flyover waypoint, Groundspeed, Altitude
- This adds the capability for the user to designate mission-related commands.

Version notes: v2.0.1
- Bug hunting

Version notes: v2.1.0
- Attempted stall prevention.

## (A)gent (T)racking and (A)ssignment (M)anagement (S)ystem v0.1

### Description

This system target assignment algorithms, and future implementation will include tracking of airborne agents (e.g., drone swarms). The tracking functions utilized in this system will be nothing more than wrappers for the Command & control (C2) of Autonomous Random finite set- (RFS-) Based Swarms (CARBS) developed by the Laboratory for Autonomy, GNC,
and Estimation Research (LAGER) at the University of Alabama (UA). The target assignment algorithms currently in use in this system are the:
1. Hungarian assignment algorithm
2. {TBD}

Version notes: v0.1.0
- First roll-out of the ATAMS package.
- Still needs a ton of development but is useable as-is for the FANGS application.
- Only includes assignment algorithms.
- Does not include any agent tracking.
- Suggested roll to v1.0.0 once agent tracking is implemented.

### Running a Simulation

Saved simulations used in the development of these algorithms and for academic work are found in the *saved_simulations* directory. The basic algorithm is as follows:

1. To use the guidance system, first you must create a _FixedWingVehicle_ object class found via _from vehicle.FixedWingVehicle import FixedWingVehicle_. The class documentation will tell you which parameters are required; any other is optional.
2. Next, initiate an object of the _FANGS.GuidanceSystem_ class, passing the _FixedWingVehicle_ object, required transfer function parameters, and initial conditions. You may also specify the time of object creation (this could be launch time or when the guidance system is "turned on"). If desired, the integration step time can also be set here--the default value is _dt=0.01_. Each update to the state parameters will give the user the opportunity to specify a step time if required (for instance, if the guidance system is being used on physical hardware and the time between calculations is not exactly 0.01 seconds).
3. Commanding the agent:
- The user may set a command trajectory at any time using the _setCommandTrajectory()_ function of the _FANGS.GuidanceSystem_ object. If no command trajectory is set, the aircraft will continue along the trajectory defined in the initial conditions. A trajectory is defined as an ATC-style set of commands: velocity, flight-path angle (glideslope), and heading.
- The user may set a flyover waypoint at any time using the _setCommandFlyover()_ function of the _FANGS.GuidanceSystem_ object. This will put the aircraft into a flyover state: at each time step, the aircraft will calculate the required trajectory to meet the flyover conditions (groundspeed, altitude, latitude/longitude) until it is sufficiently within range of the target at which point the aircraft will revert back to a trajectory state and maintain heading, airspeed, and glideslope.
- The user may give a swarm of drones a set of flyover waypoints using _ATAMS_. To do this, the user must first create a matrix of all agent states (latitude, longitude, altitude, groundspeed, heading, and flight path angle). The _ATAMS.assignments.assignAgentsToTargets()_ function will ingest the current agent states and desired flyover waypoints and, using the Hungarian algorithm, calculate the assignment set which has the lowest overall cost. The _ATAMS.assignments.assignAgentsToTargets()_ function will then assign each waypoint as a flyover command automatically, invisible to the user.
4. At each time step, the user must run the _getGuidanceCommands()_ function of the _FANGS.GuidanceSystem_ object for the drone to create its internal set of commands to meet the required guidance system user inputs.
5. At each time step, after running _getGuidanceCommands()_, the user must run the _updateSystemState()_ function of the _FANGS.GuidanceSystem_ object. This function will progress the aircraft's state one time step forward, using either the user-specified _dt_ parameter to the function or, if no _dt_ is given by the user, using the _FixedWingVehicle_ default _dt_ value, which is the same as the _FANGS.GuidanceSystem_ object default _dt_ value. At this step, the user may supply any, all, or none of the required state parameter values; if any required parameter is not supplied it will be estimated using the _FANGS.GuidanceSystem_ default ideal equations of motion (EOM).
6. Repeat steps 3, 4, and 5 until the aircraft simulation is complete.
7. After running the _FANGS.GuidanceSystem_, you can save your guidance object's state to a .pkl binary file by running utils.save_obj() and passing the _FANGS.GuidanceSystem_ object with a save-to filepath. This step will allow you to analyze saved simulations in the future without re-running the simulation, potentially saving a lot of time. Many of the required analysis tools, such as plotting states and tracks, can be found in the _utils.py_ package.
