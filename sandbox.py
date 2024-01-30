import controller.utils as utils


if __name__ == '__main__':
    testing_drone1 = utils.load_obj('abs_command_drone1.pkl')
    print(testing_drone1.lat[10], testing_drone1.lon[10], testing_drone1.h[10])

    utils.plotSim(testing_drone1, showPlots=True, plotsToMake=('Groundspeed', 'Height', 'Coordinates'))
    # utils.plotSim(testing_drone1, showPlots=True)