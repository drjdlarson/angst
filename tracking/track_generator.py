""" Track generators for aerial targets """
import numpy as np
import tracking.coordinate_transforms as ct
import controller.utils as utils
import pandas as pd


class ideal_a2a:
    def __init__(self, lat, lon, alt, roll, pitch, heading, time=0):
        # Define the observer (measuring) aircraft
        self.observer = self.observer_state(lat, lon, alt, roll, pitch, heading, time)
        self.target = self.track()
        self.time = []
        self.angle_units = 'Radians'
        self.distance_units = 'feet'

    class observer_state:
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

    class track:
        def __init__(self):
            # Init tracking values
            self.bearing = []
            self.elevation = []
            self.range = []

    def track_target_ideal(self, target_lat, target_lon, target_alt, time):
        brng = target_bearing((self.observer.lat[-1], self.observer.lon[-1], self.observer.alt[-1]),
                              (target_lat, target_lon, target_alt),
                              bearing_angle_units=self.angle_units)
        brng = brng - self.observer.heading[-1]
        rng = target_range((self.observer.lat[-1], self.observer.lon[-1], self.observer.alt[-1]),
                           (target_lat, target_lon, target_alt),
                           range_units=self.distance_units)
        elev = target_elevation((self.observer.lat[-1], self.observer.lon[-1], self.observer.alt[-1]),
                                (target_lat, target_lon, target_alt),
                                elevation_angle_units=self.angle_units)
        self.target.bearing.append(brng)
        self.target.range.append(rng)
        self.target.elevation.append(elev)
        self.time.append(time)


    def DataFrame(self, downsample=1):
        trackdict = {f'time (seconds)': self.time[::downsample],
                     f'bearing ({self.angle_units})': self.target.bearing[::downsample],
                     f'range ({self.distance_units})': self.target.range[::downsample],
                     f'elevation ({self.angle_units})': self.target.elevation[::downsample]}
        return pd.DataFrame(trackdict)

    def to_csv(self, filename, downsample=1):
        df = self.DataFrame(downsample)
        df.to_csv(filename, index=False)

class noisy_a2a(ideal_a2a):
    def __init__(self, lat, lon, alt, roll, pitch, heading, noise_mean=0, noise_std=0.1, time=0, noise_type='Gaussian'):
        ideal_a2a.__init__(self, lat, lon, alt, roll, pitch, heading, time)
        self.update_noise_parameters(noise_mean, noise_std, noise_type)

    def track_target(self, target_lat, target_lon, target_alt, time):
        # Create ideal track
        self.track_target_ideal(target_lat, target_lon, target_alt, time)
        # Add noise
        self.target.bearing[-1] = self.target.bearing[-1] + self.noise()
        self.target.range[-1] = self.target.range[-1] + self.noise()
        self.target.elevation[-1] = self.target.elevation[-1] + self.noise()

    def update_noise_parameters(self, noise_mean, noise_std, noise_type=None):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        if noise_type is not None: self.noise_type = noise_type

    def noise(self):
        if self.noise_type == 'Gaussian':
            return np.random.normal(self.noise_mean, self.noise_std)
        else:
            print(f'{self.noise_type} unsupported.')
            return 0


def target_bearing(lla1, lla2, bearing_angle_units="Radians"):
    # TODO: noise/drop-outs
    lat1, lon1, alt1 = lla1
    lat2, lon2, alt2 = lla2
    ned = ct.lla_to_NED(lat1, lon1, alt1, lat2, lon2, alt2)
    n1 = ned[0][0]
    e1 = ned[1][0]
    brng = np.arctan2(e1, n1)
    if bearing_angle_units in ["Degrees", "degrees", "deg", "Deg", "d", "D"]:
        brng = np.degrees(brng)
    return brng


def target_range(lla1, lla2, range_units="feet"):
    lat1, lon1, alt1 = lla1
    lat2, lon2, alt2 = lla2
    ned = ct.lla_to_NED(lat1, lon1, alt1, lat2, lon2, alt2)
    n1 = ned[0][0]
    e1 = ned[1][0]
    d1 = ned[2][0]
    rng = np.sqrt(n1**2 + e1**2 + d1**2)
    if range_units in ['feet', 'ft', 'Feet', 'Ft']:
        rng = rng * utils.m2feet
    return rng


def target_elevation(lla1, lla2, elevation_angle_units="Radians"):
    lat1, lon1, alt1 = lla1
    lat2, lon2, alt2 = lla2
    ned = ct.lla_to_NED(lat1, lon1, alt1, lat2, lon2, alt2)
    d1 = ned[2][0]
    rng = target_range(lla1, lla2, range_units='meters')
    angle = np.arctan2(d1, rng)
    if elevation_angle_units in ["Degrees", "degrees", "deg", "Deg", "d", "D"]:
        angle = np.degrees(angle)
    return angle

