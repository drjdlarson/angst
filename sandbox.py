import controller.utils as utils
import matplotlib.pyplot as plt


if __name__ == '__main__':
    num_drones = 8
    drones = {}
    for drone in range(num_drones):
        print(f'Loading drone {drone+1}...')
        drone_obj = utils.load_obj(f'./saved_simulations/Grand_Canyon_Search_and_Rescue/abs_command_drone{drone+1}.pkl')
        drones[str(drone+1)] = drone_obj
        utils.writeKMLfromObj(drone_obj, saveFolder=r'C:\Users\sprin\Dropbox\UA-MME\2023\Fall\ME-594\Project\fangs\saved_simulations\Grand_Canyon_Search_and_Rescue')
    utils.plotCoordinates(drones)