#!/usr/bin/env python
""" (A)gent (T)racking and (A)ssignment (M)anagement (S)ystem
        This system provides tracking of airborne agents (e.g., drone swarms) and target
        assignment algorithms. The tracking functions utilized in this system are nothing
        more than wrappers for the Command & control (C2) of Autonomous Random finite
        set- (RFS-) Based Swarms (CARBS) developed by the Laboratory for Autonomy, GNC,
        and Estimation Research (LAGER) at the University of Alabama (UA). The target
        assignment algorithms currently in use in this system are the:
            1. Hungarian assignment algorithm
            2. TBD

    Version notes: v0.0.1
        Initial development
    Version notes: v0.0.2
        Loose integration with FANGS
    Version notes: v0.0.3
        Create debug option
    Version notes: v0.1.0
        First roll-out of the ATAMS package.
        Still needs a ton of development but is useable as-is for the FANGS application.
        Only includes assignment algorithms.
        Does not include any agent tracking.
        Suggested roll to v1.0.0 once agent tracking is implemented.
"""
__author__ = "Alex Springer"
__version__ = "0.1.0"
__email__ = "springer.alex.h@gmail.com"
__status__ = "Development"

import numpy as np
import pandas as pd
import controller.utils as utils
from scipy.optimize import linear_sum_assignment


class assignments:
    """ Target assignment control system
    
    These functions will supply an optimal agent-target matching solution given
    a set of agents, a set of targets, and a cost for each assignment."""

    def __init__(self):
        self.n = 1
        self.costMatrix = [[]]*self.n
        self.weights = self.weightingValues()
        self.saveCSV = False
        self.savepath = r''
        self.debug = False
        self.CostMatrixCalcCount = 0
        self.AssignmentCount = 0


    class weightingValues:
        def __init__(self):
            self.distance = 1
            self.heading = 1
            self.altitude = 1
            self.groundspeed = 1
        
    
    def calculateCosts(self, agentStates, targets):
        """ Calculate the cost associated with each possible agent-target
        assignment based on current agent states and target(s).

        NOTE: This is all hard-coded to be compatible with FANGS. Future revisions need to make this less specific.
        
        Parameters
        ----------
        agentStates : n x 6 matrix of drone agent states:
            Each of n rows will correspond to each agent.
            Each of 6 columns will correspond to the following state variables:
                latitude (radians)
                longitude (radians)
                altitude (feet, MSL)
                groundspeed (feet per second)
                heading (radians clockwise from North)
                flight path angle (radians)
        targets : m x 4 matrix of fly-over targets:
            Each of m rows will correspond to each target.
            Each of 4 columns will correspond to the following target variables:
                latitude (radians)
                longitude (radians)
                altitude (feet, MSL)
                groundspeed (feet per second)

        Outputs
        -------
        costMatrix : n x m matrix of agent-target assignment costs:
            Each of n rows will correspond to each agent
            Each of m columns will correspond to each target
        
        Note: costMatrix will also be stored internally as self.costMatrix
        """
        n, _ = np.shape(agentStates)
        m, _ = np.shape(targets)
        self.costMatrix = np.zeros((n, m))
        for agent in range(n):
            for target in range(m):
                distanceToTarget = utils.get_distance(agentStates[agent][0], agentStates[agent][1],
                                                      targets[target][0], targets[target][1]) / utils.miles2feet

                bearing = utils.get_bearing(agentStates[agent][0], agentStates[agent][1], targets[target][0], targets[target][1])
                headingChange = abs(agentStates[agent][4] - bearing)
                if headingChange > np.pi:
                    headingChange = 2*np.pi - headingChange

                altitudeChange = targets[target][2] - agentStates[agent][2]
                if altitudeChange < 0:
                    # It's cheaper to lower altitude than it is to raise it
                    flightPathWeight = 0.75
                else:
                    flightPathWeight = 1

                groundspeedChange = abs(targets[target][-1] - agentStates[agent][3])

                distanceCost = self.weights.distance * distanceToTarget
                headingCost = self.weights.heading * headingChange
                altitudeCost = self.weights.altitude * abs(altitudeChange) * flightPathWeight
                velocityCost = self.weights.groundspeed * groundspeedChange
                totalAssignmentCost = distanceCost + headingCost + altitudeCost + velocityCost

                self.costMatrix[agent][target] = totalAssignmentCost

        self.CostMatrixCalcCount += 1

        if self.debug:
            np.savetxt(f'{self.savepath}\\costmatrix{self.CostMatrixCalcCount}.csv', self.costMatrix, delimiter=',')


    def assignAgentsToTargets(self, agents, targets, setControl=True):
        """ A wrapper for the scipy.optimize.linear_sum_assignment
        function.
        
        This function will find the cost matrix associated with the available
        agents and the given fly-over targets, then solve the cost matrix using
        the scipy.optimize.linear_sum_assignment algorithm. Finally, it will return
        the optimal assignment and total sum cost.
        
        Parameters
        ----------
        agents : :dict:`Dictionary of n FANGS GuidanceSystem objects.`
        targets : :matrix:`m x 4 matrix:
            Each of m rows will correspond to each target.
            Each of 4 columns will correspond to the following target variables:
                latitude (radians)
                longitude (radians)
                altitude (feet, MSL)
                groundspeed (feet per second)`
        """

        agentStates = np.zeros((len(agents.keys()), 6))
        agentStates = [[drone_obj.lat[-1],drone_obj.lon[-1],drone_obj.h[-1],drone_obj.v_BN_W[-1],drone_obj.sigma[-1],drone_obj.gamma[-1]] for _, drone_obj in agents.items()]

        self.calculateCosts(agentStates=agentStates, targets=targets)

        row_ind, col_ind = linear_sum_assignment(self.costMatrix)
        assigned = {}
        ii = 0
        for agentID, agent in agents.items():
            if ii in row_ind:
                assigned[agentID] = targets[col_ind[ii]][:]
                print(f'Assigning {agentID} to target {col_ind[ii]+1}')
                agent.setCommandFlyover(assigned[agentID][3],
                                        assigned[agentID][2],
                                        (assigned[agentID][0], assigned[agentID][1]))
            ii += 1
        self.AssignmentCount += 1

        if self.debug:
            np.savetxt(f'{self.savepath}\\assignments{self.AssignmentCount}.csv', [row_ind, col_ind], delimiter=',')
            agentStatesDict = {str(drone): [drone_obj.lat[-1],drone_obj.lon[-1],drone_obj.h[-1],drone_obj.v_BN_W[-1],drone_obj.sigma[-1],drone_obj.gamma[-1]] for drone, drone_obj in agents.items()}
            tt = [[drone_obj.time[-1]] for _, drone_obj in agents.items()]
            pd.DataFrame.from_dict(data=agentStatesDict, orient='index').to_csv(f'{self.savepath}\\agent_states_{tt[0]}.csv', header=False)


class tracking:

    def __init__(self):
        print('Hello world')