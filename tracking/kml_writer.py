#!/usr/bin/env python
""" Google Earth KML file writer for agent tracks
"""

import simplekml
import controller.utils as utils
import tracking.track_generator as track


def writeKMLfromObj(GuidanceSystemObject, savePath=None, noise=False, downsample=50):
    lat = [l * utils.r2d for l in GuidanceSystemObject.lat[::downsample]]
    lon = [l * utils.r2d for l in GuidanceSystemObject.lon[::downsample]]
    alt = GuidanceSystemObject.h[::downsample]

    kml = simplekml.Kml()
    lin = kml.newlinestring(name=f'agent_{GuidanceSystemObject.Vehicle.aircraftID}',
                            description='Aerial agent track',
                            coords=[(36.24545584912534,-112.2822),
                                    (36.2700304603568,-112.30236322830645)])
    kml.save(f'agent_{GuidanceSystemObject.Vehicle.aircraftID}.kml')
