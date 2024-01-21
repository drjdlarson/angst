import controller.utils as utils


if __name__ == '__main__':
    testing_drone2 = utils.load_obj('testing_drone2.pkl')
    print(testing_drone2.lat[10], testing_drone2.lon[10], testing_drone2.h[10])

    utils.plotSim(testing_drone2, showPlots=True)