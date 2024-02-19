import controller.utils as utils
import matplotlib.pyplot as plt
import tracking.kml_writer as kml_writer


if __name__ == '__main__':
    num_drones = 8
    drones = {}
    for drone in range(num_drones):
        print(f'Loading drone {drone+1}...')
        drone_obj = utils.load_obj(f'./saved_simulations/Grand_Canyon_Search_and_Rescue/abs_command_drone{drone+1}.pkl')
        drones[str(drone+1)] = drone_obj
        kml_writer.writeKMLfromObj(drone_obj, saveFolder=r'C:\Users\Alex\Dropbox\UA-MME\2023\Fall\ME-594\Project\fangs\saved_simulations\Grand_Canyon_Search_and_Rescue')
    # print(drones.items())
    utils.plotCoordinates(drones)
    # plt.show()