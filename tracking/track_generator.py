""" Track generators for aerial targets """
import numpy as np
import tracking.coordinate_transforms as ct


class ideal_a2a:
    def __init__(self, lat, lon, alt, roll, pitch, heading, time=0):
        # Define the c2 (measuring) aircraft
        self.c2 = self.c2_state(lat, lon, alt, roll, pitch, heading, time)
        self.target = self.track()
        self.time = [time]

    class c2_state:
        def __init__(self, lat, lon, alt, roll, pitch, heading, time):
            self.lat = [lat]
            self.lon = [lon]
            self.alt = [alt]
            self.roll = [roll]
            self.pitch = [pitch]
            self.heading = [heading]

        def update_state(self, lat, lon, alt, roll, pitch, heading, time):
            self.lat.append(lat)
            self.lon.append(lon)
            self.alt.append(alt)
            self.roll.append(roll)
            self.pitch.append(pitch)
            self.heading.append(heading)

    class track:
        def __init__(self):
            # Init tracking values
            self.bearing = []
            self.lookup = []
            self.range = []

    def track_target_ideal(self, target_lat, target_lon, target_alt, dt=0):
        brng = target_bearing((self.c2.lat[-1], self.c2.lon[-1], self.c2.alt[-1]), (target_lat, target_lon, target_alt), method="lla" ) - self.c2.heading[-1]
        rng = target_range((self.c2.lat[-1], self.c2.lon[-1], self.c2.alt[-1]), (target_lat, target_lon, target_alt),  method="lla")
        self.target.bearing.append(brng)
        self.target.range.append(rng)
        self.time.append(self.time[-1] + dt)


class noisy_a2a(ideal_a2a):
    def __init__(
        self, lat, lon, alt, roll, pitch, heading, noise_mean=0, noise_std=0.01, time=0
    ):
        ideal_a2a.__init__(self, lat, lon, alt, roll, pitch, heading, time)
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def track_target(self, target_lat, target_lon, target_alt, dt=0):
        # TODO Change dt to time, pass time variable
        # TODO Fix whatever stupid-ass failure is happening with track_target_ideal.
        self.track_target_ideal(self, target_lat, target_lon, target_alt, dt)
        self.target.bearing[-1] = self.target.bearing[-1] + np.random(
            self.noise_mean, self.noise_std
        )


def target_bearing(pos1, pos2=None, angles="Radians", units="feet", method="ned"):
    # TODO: noise/drop-outs
    if method in ["lla", "LLA"]:
        if pos2 is None:
            raise (ValueError("When method is set to lla, pos2 is required."))
        lat1, lon1, alt1, lat2, lon2, alt2, dLon = unpack_lla(pos1, pos2, angles)
        ned = ct.lla_to_NED(lat1, lon1, alt1, lat2, lon2, alt2)
        n1 = ned[0][0]
        e1 = ned[1][0]
        d1 = ned[2][0]
    elif method in ["ned", "NED"]:
        if pos2 is not None:
            print("Reminder: Only pos1 is used when the method is set to NED.")
        n1, e1, d1 = pos1
    brng = np.arctan2(e1, n1)
    if angles in ["Degrees", "degrees", "deg", "Deg", "d", "D"]:
        print(f"Converting {brng} Radians to Degrees")
        brng = np.degrees(brng)
    return brng


def target_range(pos1, pos2, angles="Radians", units="feet", method="ned"):
    # Convert to NED if needed
    if method in ["lla", "LLA"]:
        if pos2 is None:
            raise (ValueError("When method is set to lla, pos2 is required."))
        lat1, lon1, alt1, lat2, lon2, alt2, dLon = unpack_lla(pos1, pos2, angles)
        ned = ct.lla_to_NED(lat1, lon1, alt1, lat2, lon2, alt2)
        n1 = ned[0][0]
        e1 = ned[1][0]
        d1 = ned[2][0]
    elif method in ["ned", "NED"]:
        if pos2 is not None:
            print("Reminder: Only pos1 is used when the method is set to NED.")
        n1, e1, d1 = pos1
    # Compute euclidian distance using NED coordinates
    return np.sqrt(n1**2 + e1**2 + d1**2)


def target_lookupAngle(pos1, pos2, angles="Radians"):
    return


def unpack_lla(pos1, pos2, angles="Radians"):
    lat1, lon1, alt1 = pos1
    lat2, lon2, alt2 = pos2
    dLon = lon2 - lon1
    if angles in ["Degrees", "degrees", "deg", "Deg", "d", "D"]:
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dLon = np.radians(dLon)
    return lat1, lon1, alt1, lat2, lon2, alt2, dLon
