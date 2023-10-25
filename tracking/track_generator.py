""" Track generators for aerial targets """
import numpy as np
import tracking.coordinate_transforms as ct


class air_to_air_3d_ideal:
    def __init__(self, lat, lon, alt, roll, pitch, heading, time=0):
        # Define the host (measuring) aircraft
        self.host = self.state(lat, lon, alt, roll, pitch, heading, time)

        # Init tracking values
        self.bearing = []
        self.lookup = []
        self.range = []

    class state:
        def __init__(self, lat, lon, alt, roll, pitch, heading, time):
            self.lat = [lat]
            self.lon = [lon]
            self.alt = [alt]
            self.roll = [roll]
            self.pitch = [pitch]
            self.heading = [heading]
            self.time = [time]

        def update_state(self, lat, lon, alt, roll, pitch, heading, time):
            self.lat.append(lat)
            self.lon.append(lon)
            self.alt.append(alt)
            self.roll.append(roll)
            self.pitch.append(pitch)
            self.heading.append(heading)
            self.time.append(time)
    
    def track_target(self, target_lat, target_lon, target_alt, time):
        brng = target_bearing((self.host.lat[-1], self.host.lon[-1], self.host.alt[-1]),
                             (target_lat, target_lon, target_alt),
                             method='lla') - self.host.heading[-1]
        rng = target_range((self.host.lat[-1], self.host.lon[-1], self.host.alt[-1]),
                           (target_lat, target_lon, target_alt),
                           method='lla')
        self.bearing.append(brng)
        self.range.append(rng)
        self.host.time.append(time)


def target_bearing(pos1, pos2=None, angles='Radians', units='feet', method='ned'):
    # TODO: noise/drop-outs
    if method in ['lla', 'LLA']:
        if pos2 is None: raise(ValueError('When method is set to lla, pos2 is required.'))
        lat1, lon1, alt1, lat2, lon2, alt2, dLon = unpack_lla(pos1, pos2, angles)
        ned = ct.lla_to_NED(lat1, lon1, alt1, lat2, lon2, alt2)
        n1 = ned[0][0]
        e1 = ned[1][0]
        d1 = ned[2][0]
    elif method in ['ned', 'NED']:
        if pos2 is not None: print('Reminder: Only pos1 is used when the method is set to NED.')
        n1, e1, d1 = pos1
    brng = np.arctan2(e1,n1)
    if angles in ['Degrees', 'degrees', 'deg', 'Deg', 'd', 'D']:
        print(f'Converting {brng} Radians to Degrees')
        brng = np.degrees(brng)
    return brng


def target_range(pos1, pos2, angles='Radians', units='feet', method='ned'):
    # Convert to NED if needed
    if method in ['lla', 'LLA']:
        if pos2 is None: raise(ValueError('When method is set to lla, pos2 is required.'))
        lat1, lon1, alt1, lat2, lon2, alt2, dLon = unpack_lla(pos1, pos2, angles)
        ned = ct.lla_to_NED(lat1, lon1, alt1, lat2, lon2, alt2)
        n1 = ned[0][0]
        e1 = ned[1][0]
        d1 = ned[2][0]
    elif method in ['ned', 'NED']:
        if pos2 is not None: print('Reminder: Only pos1 is used when the method is set to NED.')
        n1, e1, d1 = pos1
    # Compute euclidian distance using NED coordinates
    return np.sqrt(n1**2 + e1**2 + d1**2)


def target_lookupAngle(pos1, pos2, angles='Radians'):
    return


def unpack_lla(pos1, pos2, angles='Radians'):
    lat1, lon1, alt1 = pos1
    lat2, lon2, alt2 = pos2
    dLon = lon2 - lon1
    if angles in ['Degrees', 'degrees', 'deg', 'Deg', 'd', 'D']:
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dLon = np.radians(dLon)
    return lat1, lon1, alt1, lat2, lon2, alt2, dLon