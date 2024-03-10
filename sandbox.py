import controller.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import controller.ATTACS as ATTACS


if __name__ == '__main__':
    time = 150  # randomly chose a number of seconds at which to test ATTACS
    num_drones = 8
    drones = {}
    agentStates = np.zeros((num_drones, 6))
    targets = np.zeros((num_drones, 4))
    # target_flyovers = [[(36.530367)*utils.d2r, (-112.057600)*utils.d2r, 4000, 50*utils.knts2fps],
    #                    [(36.179491)*utils.d2r, (-111.951595)*utils.d2r, 4000, 50*utils.knts2fps],
    #                    [(36.276756)*utils.d2r, (-112.687766)*utils.d2r, 4000, 50*utils.knts2fps],
    #                    [(36.542782)*utils.d2r, (-112.124202)*utils.d2r, 4000, 50*utils.knts2fps],
    #                    [(36.089355)*utils.d2r, (-112.409598)*utils.d2r, 4000, 50*utils.knts2fps],
    #                    [(36.218540)*utils.d2r, (-111.969167)*utils.d2r, 4000, 50*utils.knts2fps],
    #                    [(36.449291)*utils.d2r, (-112.399009)*utils.d2r, 4000, 50*utils.knts2fps],
    #                    [(36.580698)*utils.d2r, (-111.866192)*utils.d2r, 4000, 50*utils.knts2fps]]
    target_flyovers = [[(36.179491)*utils.d2r, (-111.951595)*utils.d2r, 4000, 50*utils.knts2fps],
                       [(36.276756)*utils.d2r, (-112.687766)*utils.d2r, 4000, 50*utils.knts2fps],
                       [(36.542782)*utils.d2r, (-112.124202)*utils.d2r, 4000, 50*utils.knts2fps],
                       [(36.089355)*utils.d2r, (-112.409598)*utils.d2r, 4000, 50*utils.knts2fps],
                       [(36.218540)*utils.d2r, (-111.969167)*utils.d2r, 4000, 50*utils.knts2fps],
                       [(36.449291)*utils.d2r, (-112.399009)*utils.d2r, 4000, 50*utils.knts2fps],
                       [(36.580698)*utils.d2r, (-111.866192)*utils.d2r, 4000, 50*utils.knts2fps]]

    for drone in range(num_drones):
        print(f'Loading drone {drone+1}...')
        drone_obj = utils.load_obj(f'./saved_simulations/Grand_Canyon_Search_and_Rescue/abs_command_drone{drone+1}.pkl')
        drones[str(drone+1)] = drone_obj
        agentStates[drone] = [drone_obj.lat[int(time/drone_obj.dt)],
                              drone_obj.lon[int(time/drone_obj.dt)],
                              drone_obj.h[int(time/drone_obj.dt)],
                              drone_obj.v_BN_W[int(time/drone_obj.dt)],
                              drone_obj.sigma[int(time/drone_obj.dt)],
                              drone_obj.gamma[int(time/drone_obj.dt)]]
    
    targetAssignment = ATTACS.assignments()
    targetAssignment.weights.distance = 10
    targetAssignment.weights.altitude = 1
    targetAssignment.weights.groundspeed = 0.1
    targetAssignment.weights.heading = 100
    # targetAssignment.linearJV()
    targetAssignment.calculateCosts(agentStates=agentStates,
                                    targets=target_flyovers)
    print(targetAssignment.costMatrix)
    targetAssignment.assignAgentsToTargets(agents=drones, targets=target_flyovers)