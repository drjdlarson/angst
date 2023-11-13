import controller.utils as utils


if __name__ == '__main__':
    drone4 = utils.load_obj('drone4.pkl')
    print(drone4.lon[-2], drone4.lon[-1])

    drone5 = utils.load_obj('drone5.pkl')
    print(drone5.lon[-2], drone5.lon[-1])