"""
reconstruct_check_v2.py : do the closest-approach labels (Trajectory_GPU_v2) determine the trajectory?

    python reconstruct_check_v2.py [path/to/closest_approach.csv]

For every satellite, rebuild the Earth-fixed position and velocity at closest approach
from the label values ALONE (as Trajectory_GPU_v2/labels.py defines them):

    altitude, elevation, folded azimuth (0-90 deg)       continuous
    angular rate, radial velocity                        continuous
    eastbound, northbound, passes_east                   one flag per ambiguity

and compare with the true state that closest_approach.py saved. A position and a
velocity at one instant fix the whole trajectory (propagate them with the force model).
Uses numpy and scipy only (no orbdetpy).

The folded azimuth leaves 4 azimuths (NE/SE/SW/NW mirror images). For each, the line
of sight is set by azimuth and elevation, and the range is found where the geodetic
altitude (WGS-84) equals the label. At closest approach the velocity is perpendicular
to the line of sight; its size is angular rate x range and its component along the
satellite's radius is the radial velocity, which leaves 2 directions. That gives
4 x 2 = 8 candidates -- the 8-fold ambiguity. The three flags are computed for each
candidate exactly as for the truth, and the candidates that match all three are kept.

It also reports how many satellites stay ambiguous when only some of the flags are
known (eastbound alone; eastbound + northbound), i.e. what each flag resolves.
"""
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

A_WGS, E2_WGS = 6378137.0, 6.69437999014e-3
STATION_LAT, STATION_LON = 0.0, -2.7272   # generate_params.py
FLAGS = ("eastbound", "northbound", "passes_east")


def geodetic_alt(p):
    x, y, z = p
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1 - E2_WGS))
    for _ in range(6):
        n = A_WGS / math.sqrt(1 - E2_WGS * math.sin(lat) ** 2)
        h = r / math.cos(lat) - n
        lat = math.atan2(z, r * (1 - E2_WGS * n / (n + h)))
    return h


def flags(heading_rad, turn_sense):
    """The three flags, from the direction of motion and the turn sense -- the same
    definitions as Trajectory_GPU_v2/labels.py."""
    eastbound = math.sin(heading_rad) > 0
    northbound = math.cos(heading_rad) > 0
    passes_east = (turn_sense > 0) == northbound   # station left of a northbound track / right of a southbound one
    return {"eastbound": eastbound, "northbound": northbound, "passes_east": passes_east}


def main(path):
    ca = pd.read_csv(path)
    ca = ca[ca["status"] == "ok"]
    lat, lon = STATION_LAT, STATION_LON
    n0 = A_WGS / math.sqrt(1 - E2_WGS * math.sin(lat) ** 2)
    station = np.array([n0 * math.cos(lat) * math.cos(lon), n0 * math.cos(lat) * math.sin(lon),
                        n0 * (1 - E2_WGS) * math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])

    known_sets = {"no flags": (), "eastbound only": ("eastbound",),
                  "eastbound + northbound": ("eastbound", "northbound"), "all three flags": FLAGS}
    n_cands = {k: [] for k in known_sets}
    pos_err, vel_err = [], []
    for _, r in ca.iterrows():
        # --- the labels, as labels.py builds them ---
        az_true = math.radians(r["azimuth_ca_deg"])
        alt, el = r["altitude_ca_m"], math.radians(r["elevation_ca_deg"])
        fold = math.acos(min(1.0, abs(math.cos(az_true))))
        rate, vr = math.radians(r["ang_rate_ca_deg_s"]), r["radial_vel_ca_mps"]
        truth = flags(math.radians(r["heading_ca_deg"]), r["turn_sense"])

        # --- the 8 candidates, from the continuous labels only ---
        cands = []
        for az, sgn in itertools.product((fold, math.pi - fold, math.pi + fold, 2 * math.pi - fold), (1, -1)):
            d = math.cos(el) * math.sin(az) * east + math.cos(el) * math.cos(az) * north + math.sin(el) * up
            rho = brentq(lambda s: geodetic_alt(station + s * d) - alt, 1e3, 6e6, xtol=1e-6)
            pos = station + rho * d
            rhat = pos / np.linalg.norm(pos)
            u1 = rhat - (rhat @ d) * d
            u1 /= np.linalg.norm(u1)
            u2 = np.cross(d, u1)
            alpha = vr / (u1 @ rhat)
            v = alpha * u1 + sgn * math.sqrt(max((rate * rho) ** 2 - alpha ** 2, 0.0)) * u2
            turn = 1 if (d @ east) * (v @ north) - (d @ north) * (v @ east) > 0 else -1
            cands.append((pos, v, flags(math.atan2(v @ east, v @ north), turn)))

        for name, known in known_sets.items():
            match = [c for c in cands if all(c[2][f] == truth[f] for f in known)]
            n_cands[name].append(len(match))
            if name == "all three flags" and len(match) == 1:
                pos, v, _ = match[0]
                pos_err.append(np.linalg.norm(pos - r[["x_ecef_ca_m", "y_ecef_ca_m", "z_ecef_ca_m"]].to_numpy(float)))
                vel_err.append(np.linalg.norm(v - r[["vx_ecef_ca_mps", "vy_ecef_ca_mps", "vz_ecef_ca_mps"]].to_numpy(float)))

    print(f"{len(ca)} satellites. Candidates (of 8) that fit the continuous labels and the known flags:")
    for name, counts in n_cands.items():
        c = pd.Series(counts).value_counts().sort_index()
        print(f"  {name:>24}: " + ", ".join(f"{k} candidate{'s' if k != 1 else ''}: {v}" for k, v in c.items()))
    if pos_err:
        pe, ve = np.array(pos_err), np.array(vel_err)
        print(f"With all three flags, unique rebuilds vs the true state: position error median {np.median(pe):.3g} m, "
              f"max {pe.max():.3g} m; velocity error median {np.median(ve):.3g} m/s, max {ve.max():.3g} m/s")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "output", "closest_approach.csv"))
