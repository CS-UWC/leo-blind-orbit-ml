"""
reconstruct_check.py : do the 9 closest-approach labels (Trajectory_GPU) determine the trajectory?

    python reconstruct_check.py [path/to/closest_approach.csv]

For every satellite, rebuild the Earth-fixed position and velocity at closest approach
from the label values ALONE (as Trajectory_GPU/labels.py defines them):

    altitude, elevation, folded azimuth (0-90 deg),
        east_of_station, north_of_station                             -> position
    angular rate, radial velocity, northbound, eastbound              -> velocity

and compare with the true state that closest_approach.py saved. A position and a
velocity at one instant fix the whole trajectory (propagate them with the force
model). Uses numpy and scipy only (no orbdetpy).

Position: the folded azimuth and the two side flags give the azimuth; the line of
sight is set by azimuth and elevation, and the range is found where the geodetic
altitude (WGS-84) equals the label.
Velocity: at closest approach it is perpendicular to the line of sight; its size is
angular rate x range, its component along the satellite's radius is the radial
velocity, which leaves two directions. Keep the one moving north/south and
east/west as the two flags say. Exactly one should survive.

With --without-north-of-station the azimuth is left with two candidates (north or
south of the station); about 2% of satellites (nearly due east or west of the
station on a near-polar pass) then have two candidates that fit every other label.

(reconstruct_check_v2.py is the later check for Trajectory_GPU_v2's labels.)
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

A_WGS, E2_WGS = 6378137.0, 6.69437999014e-3
STATION_LAT, STATION_LON = 0.0, -2.7272   # generate_params.py


def geodetic_alt(p):
    x, y, z = p
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1 - E2_WGS))
    for _ in range(6):
        n = A_WGS / math.sqrt(1 - E2_WGS * math.sin(lat) ** 2)
        h = r / math.cos(lat) - n
        lat = math.atan2(z, r * (1 - E2_WGS * n / (n + h)))
    return h


def main(path, use_north_of_station=True):
    ca = pd.read_csv(path)
    ca = ca[ca["status"] == "ok"]
    lat, lon = STATION_LAT, STATION_LON
    n0 = A_WGS / math.sqrt(1 - E2_WGS * math.sin(lat) ** 2)
    station = np.array([n0 * math.cos(lat) * math.cos(lon), n0 * math.cos(lat) * math.sin(lon),
                        n0 * (1 - E2_WGS) * math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])

    n_unique, n_none, n_many, pos_err, vel_err = 0, 0, 0, [], []
    for _, r in ca.iterrows():
        az_true, hd = math.radians(r["azimuth_ca_deg"]), math.radians(r["heading_ca_deg"])
        # --- the labels, as labels.py builds them ---
        alt, el = r["altitude_ca_m"], math.radians(r["elevation_ca_deg"])
        fold = math.acos(min(1.0, abs(math.cos(az_true))))
        east_of, north_of = math.sin(az_true) > 0, math.cos(az_true) > 0
        rate, vr = math.radians(r["ang_rate_ca_deg_s"]), r["radial_vel_ca_mps"]
        northbound, eastbound = math.cos(hd) > 0, math.sin(hd) > 0

        # --- rebuild from the labels only ---
        azs = (fold, math.pi - fold) if east_of else (2 * math.pi - fold, math.pi + fold)
        if use_north_of_station:
            azs = azs[:1] if north_of else azs[1:]
        found = []
        for az in azs:
            d = math.cos(el) * math.sin(az) * east + math.cos(el) * math.cos(az) * north + math.sin(el) * up
            rho = brentq(lambda s: geodetic_alt(station + s * d) - alt, 1e3, 6e6, xtol=1e-6)
            pos = station + rho * d
            speed = rate * rho
            rhat = pos / np.linalg.norm(pos)
            u1 = rhat - (rhat @ d) * d
            u1 /= np.linalg.norm(u1)
            u2 = np.cross(d, u1)
            alpha = vr / (u1 @ rhat)
            beta = math.sqrt(max(speed ** 2 - alpha ** 2, 0.0))
            for sgn in (1, -1):
                v = alpha * u1 + sgn * beta * u2
                if ((v @ north) > 0) == northbound and ((v @ east) > 0) == eastbound:
                    found.append((pos, v))
        if len(found) == 1:
            n_unique += 1
            pos, v = found[0]
            pos_err.append(np.linalg.norm(pos - r[["x_ecef_ca_m", "y_ecef_ca_m", "z_ecef_ca_m"]].to_numpy(float)))
            vel_err.append(np.linalg.norm(v - r[["vx_ecef_ca_mps", "vy_ecef_ca_mps", "vz_ecef_ca_mps"]].to_numpy(float)))
        elif not found:
            n_none += 1
        else:
            n_many += 1

    print(f"{len(ca)} satellites: {n_unique} rebuilt uniquely, {n_many} with more than one candidate, "
          f"{n_none} with none")
    if pos_err:
        pe, ve = np.array(pos_err), np.array(vel_err)
        print(f"  position error vs true state: median {np.median(pe):.3g} m, max {pe.max():.3g} m")
        print(f"  velocity error vs true state: median {np.median(ve):.3g} m/s, max {ve.max():.3g} m/s")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    args = [x for x in sys.argv[1:] if x != "--without-north-of-station"]
    main(args[0] if args else os.path.join(here, "output", "closest_approach.csv"),
         use_north_of_station="--without-north-of-station" not in sys.argv)
