import controller.utils as utils
import matplotlib.pyplot as plt
import tracking.kml_writer as kml_writer


if __name__ == '__main__':
    # num_drones = 8
    # drones = {}
    # for drone in range(num_drones):
    #     print(f'Loading drone {drone+1}...')
    #     drone_obj = utils.load_obj(f'./saved_simulations/Grand_Canyon_Search_and_Rescue/abs_command_drone{drone+1}.pkl')
    #     drones[str(drone+1)] = drone_obj
    # print(drones.items())
    # utils.plotCoordinates(drones)
    # plt.show()

    obj = utils.load_obj(r'saved_simulations\Grand_Canyon_Search_and_Rescue\abs_command_drone1.pkl')
    kml_writer.writeKMLfromObj(obj)