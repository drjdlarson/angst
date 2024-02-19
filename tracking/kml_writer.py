#!/usr/bin/env python
""" Google Earth KML file writer for agent tracks
"""

import simplekml
import controller.utils as utils
import tracking.track_generator as track
import numpy as np


def writeKMLfromObj(GuidanceSystemObject, saveFolder=None, noise=False, downsample=50):
    if saveFolder is None:
        saveFolder = '.'
    lat = [l * utils.r2d for l in GuidanceSystemObject.lat[::downsample]]
    lon = [l * utils.r2d for l in GuidanceSystemObject.lon[::downsample]]
    alt = GuidanceSystemObject.h[::downsample]
    target_coords = [(x*utils.r2d, y*utils.r2d) for (x,y) in GuidanceSystemObject.command.waypoint_history]

    kml = simplekml.Kml()
    lin = kml.newlinestring(name=f'agent_{GuidanceSystemObject.Vehicle.aircraftID}',
                            description='The ground track of an aerial agent using FANGS',
                            coords=zip(lon, lat, alt))
    kml.save(f'{saveFolder}\\agent_{GuidanceSystemObject.Vehicle.aircraftID}.kml')

    ii = 1
    print(target_coords)
    if len(target_coords) > 0:
        for tgt in target_coords:
            print(tgt[::-1])
            # kml = simplekml.Kml()
            pt = kml.newpoint(name=f'agent_{GuidanceSystemObject.Vehicle.aircraftID}_target_{ii}',
                            description='A waypoint defined by the FANGS user for drone fly-over',
                            coords = [tgt[::-1]])
            kml.save(f'{saveFolder}\\agent_{GuidanceSystemObject.Vehicle.aircraftID}_target_{ii}.kml')
