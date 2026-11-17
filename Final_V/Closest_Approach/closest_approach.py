# =============================================================================
# closest_approach.py -- altitude, elevation and azimuth of every satellite at
#                        its point of closest approach to the ground station
# =============================================================================
#
# PURPOSE
# -------
# For every satellite in random_orbital_parameters.csv, find the instant of
# closest approach (minimum range to the station) on the trajectory that
# simv3.py / simv4.py actually simulated, and report the satellite's geodetic
# altitude, elevation and azimuth (as seen from the station) at that instant.
#
#     python closest_approach.py                       all satellites
#     python closest_approach.py --test-limit 20       first 20 only (quick check)
#     python closest_approach.py --params PATH --out PATH
#
# WHY RE-PROPAGATE INSTEAD OF USING generate_params.py's SEED POINT
# -----------------------------------------------------------------
# generate_params.py builds each orbit around a planned closest approach
# (two-body motion), and saves that point's zenith angle but not its azimuth.
# The simulated trajectory differs from that plan:
#   - simv3.py shifts each satellite's start time by a random jitter of up to
#     +/- half a cadence step (15 s), while the station keeps turning with the
#     Earth (up to ~7 km at the equator in 15 s);
#   - Orekit propagates with its full force model (J2 and higher), not two-body.
# So this file repeats simv3.py's propagation exactly -- same elements, same
# per-satellite start offset, same jitter sequence (random.Random(1234), one
# draw per satellite in file order), same window, cadence and station -- and
# finds the closest approach on that trajectory.
#
# METHOD
# ------
# 1. Propagate (orbdetpy, RANGE_RATE only, as simv3.py) and take the visible
#    samples. Satellites are sent to orbdetpy's Java server in batches; it
#    propagates the satellites of a batch in parallel on all CPU cores.
# 2. Find the sample pair where the range rate changes sign from negative
#    (approaching) to positive (receding) -- the same rule as build_dataset.py.
# 3. Transform the two bracketing true states to the Earth-fixed frame and
#    join them with a cubic Hermite curve (positions AND velocities at both
#    ends, so it is accurate to well under a metre over 30 s). Solve
#    (r_sat - r_station) . v_sat = 0 on that curve: the instant of minimum
#    range.
# 4. At that instant: geodetic altitude (WGS-84, orbdetpy pos_to_lla),
#    elevation and azimuth in the station's East-North-Up frame (the same
#    vacuum geometry simv3.py uses), slant range, geocentric radius, and the
#    sub-satellite latitude and longitude.
#
# CHECK AGAINST THE SAVED SIMULATION
# ----------------------------------
# If delay_differences.csv sits next to the params file, the range rates
# propagated here are compared with the ones simv3.py saved there. They should
# agree to rounding; a larger difference means the trajectories were not
# reproduced (for example a different jitter seed or a noisy simulation).
#
# OUTPUT (one row per satellite, columns)
# ----------------------------------------
#   sat_idx
#   t_ca_s             time of closest approach, s after the satellite's own
#                      propagation start (same clock as simv3.py's times_s)
#   t_ca_from_aos_s    time of closest approach after the first visible sample
#   range_ca_m         slant range station -> satellite at closest approach
#   altitude_ca_m      geodetic altitude above the WGS-84 ellipsoid
#   radius_ca_m        distance from the Earth's centre
#   elevation_ca_deg   elevation above the station's horizon
#   azimuth_ca_deg     azimuth from north, clockwise (0 = N, 90 = E)
#   subsat_lat_deg, subsat_lon_deg   point on the ground below the satellite
#   speed_ca_mps       satellite speed in the Earth-fixed frame (all of it is
#                      across the line of sight: the range rate is zero here)
#   ang_rate_ca_deg_s  angular rate of the line of sight seen from the station
#                      (speed / range)
#   az_rate_ca_deg_s, el_rate_ca_deg_s   the same split into azimuth and
#                      elevation rates (el rate is ~0 for a near-circular orbit)
#   heading_ca_deg     direction of motion across the station's sky: azimuth of
#                      the velocity's horizontal part (0 = moving north, 90 = east)
#   turn_sense         +1 if, seen from above, the satellite goes round the
#                      station anticlockwise (station on its left), -1 if clockwise
#   radial_vel_ca_mps  rate of change of the satellite's distance from the
#                      Earth's centre (climbing > 0); zero for a circular orbit
#   x/y/z_ecef_ca_m, vx/vy/vz_ecef_ca_mps   full Earth-fixed (ITRF) state at
#                      closest approach
#   seed_elevation_deg the elevation generate_params.py planned (90 - z0), for comparison
#   status             ok / no_pass / no_crossing / propagation_failed
# =============================================================================

import argparse
import csv
import math
import os
import random
import time

import numpy as np
from scipy.optimize import brentq

from orbdetpy import configure, add_station, MeasurementType, Frame, PositionAngle
from orbdetpy.conversion import get_J2000_epoch_offset, elem_to_pv, lla_to_pos, pos_to_lla, transform_frame
from orbdetpy.propagation import propagate_orbits

ECEF_FRAME = Frame.ITRF_CIO_CONV_2010_ACCURATE_EOP
EPOCH_BASE_STR = "2019-05-01T00:00:00"   # must match generate_params.py's EPOCH_STR
RSO_MASS = 2500.0                         # simv3.py default
JITTER_SEED = 1234                        # simv3.py default
C_LIGHT = 299792458.0

COLUMNS = ["sat_idx", "t_ca_s", "t_ca_from_aos_s", "range_ca_m", "altitude_ca_m", "radius_ca_m",
           "elevation_ca_deg", "azimuth_ca_deg", "subsat_lat_deg", "subsat_lon_deg",
           "speed_ca_mps", "ang_rate_ca_deg_s", "az_rate_ca_deg_s", "el_rate_ca_deg_s",
           "heading_ca_deg", "turn_sense", "radial_vel_ca_mps",
           "x_ecef_ca_m", "y_ecef_ca_m", "z_ecef_ca_m", "vx_ecef_ca_mps", "vy_ecef_ca_mps", "vz_ecef_ca_mps",
           "seed_elevation_deg", "status"]


def enu_basis(lat, lon):
    """East/North/Up unit vectors (in ECEF) at the station (same as simv3.py)."""
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    return east, north, up


def hermite(p0, v0, p1, v1, h, s):
    """Position and velocity at s in [0, h] on the cubic Hermite curve through (p0, v0) and (p1, v1)."""
    u = s / h
    h00, h10, h01, h11 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u, -2*u**3 + 3*u**2, u**3 - u**2
    d00, d10, d01, d11 = 6*u**2 - 6*u, 3*u**2 - 4*u + 1, -6*u**2 + 6*u, 3*u**2 - 2*u
    pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
    vel = (d00 * p0 + d10 * h * v0 + d01 * p1 + d11 * h * v1) / h
    return pos, vel


def propagate(cfgs):
    """propagate_orbits on a batch; if the batch fails, retry one by one so a
    single bad orbit only loses itself (None in its place)."""
    try:
        return [list(r.array) for r in propagate_orbits(cfgs)]
    except Exception:
        out = []
        for cfg in cfgs:
            try:
                out.append(list(propagate_orbits([cfg])[0].array))
            except Exception:
                out.append(None)
        return out


def load_params(path, limit):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def load_saved_range_rates(delay_path, delta_t):
    """sat_idx -> range rates simv3.py saved (delay * c / delta_t), or {} if the file is absent."""
    if not os.path.exists(delay_path):
        return {}
    saved = {}
    with open(delay_path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        i_n, i_dd = header.index("num_valid_samples"), header.index("dd_000")
        for row in rd:
            n = int(row[i_n])
            saved[int(row[0])] = np.array([float(v) for v in row[i_dd:i_dd + n]]) * C_LIGHT / delta_t
    return saved


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Altitude, elevation and azimuth at closest approach.")
    ap.add_argument("--params", default=os.path.join(here, "..", "Clean_Testing", "output",
                                                     "random_orbital_parameters.csv"),
                    help="random_orbital_parameters.csv (default: ../Clean_Testing/output/)")
    ap.add_argument("--out", default=os.path.join(here, "output", "closest_approach.csv"))
    ap.add_argument("--test-limit", type=int, default=None, help="only the first N satellites")
    ap.add_argument("--batch", type=int, default=100, help="satellites per orbdetpy call (run in parallel)")
    a = ap.parse_args()

    rows = load_params(a.params, a.test_limit)
    if not rows:
        raise SystemExit(f"No satellites in {a.params}")
    first = rows[0]
    st_name = first["station_name"]
    st_lat, st_lon, st_alt = (float(first[k]) for k in ("station_lat_rad", "station_lon_rad", "station_alt_m"))
    cadence = float(first["cadence_s"])
    window = float(first["search_window_s"])
    delta_t = float(first["delta_t_s"])
    if float(first["range_rate_error_ms"]) != 0.0:
        print("note: range_rate_error_ms is not 0; closest approach is computed from the true "
              "(noise-free) trajectory either way, but the range-rate check below will show the noise.")

    epoch_base = get_J2000_epoch_offset(EPOCH_BASE_STR)
    station = np.array(lla_to_pos(epoch_base, [st_lat, st_lon, st_alt]))
    east, north, up = enu_basis(st_lat, st_lon)

    # Same jitter stream as simv3.py: one draw per satellite, in file order.
    jitter_rng = random.Random(JITTER_SEED)
    prop_start = []
    for r in rows:
        prop_start.append(epoch_base + float(r["epoch_offset_s"]) + jitter_rng.uniform(-cadence / 2, cadence / 2))

    saved_rr = load_saved_range_rates(os.path.join(os.path.dirname(a.params), "delay_differences.csv"), delta_t)
    max_rr_diff = 0.0

    print(f"{len(rows)} satellites from {a.params}, batches of {a.batch}")
    t0 = time.time()
    results = []
    for b0 in range(0, len(rows), a.batch):
        batch = rows[b0:b0 + a.batch]
        starts = prop_start[b0:b0 + a.batch]
        states = elem_to_pv(Frame.EME2000, starts,
                            [float(r["sma_m"]) for r in batch], [float(r["ecc"]) for r in batch],
                            [float(r["inc_rad"]) for r in batch], [float(r["raan_rad"]) for r in batch],
                            [float(r["argp_rad"]) for r in batch], [float(r["anom_rad"]) for r in batch],
                            [PositionAngle.MEAN] * len(batch))
        cfgs = []
        for t_start, state in zip(starts, states):
            cfg = configure(rso_mass=RSO_MASS, prop_start=t_start, prop_initial_state=list(state.array),
                            prop_end=t_start + window, prop_step=cadence, sim_measurements=True)
            add_station(cfg, st_name, st_lat, st_lon, st_alt)
            cfg.measurements[MeasurementType.RANGE_RATE].error[:] = [0.0]
            cfgs.append(cfg)
        meas_all = propagate(cfgs)

        # bracketing sample pairs (approach -> recession), then one bulk frame transform
        pairs, todo = [], []
        for k, (r, t_start, meas) in enumerate(zip(batch, starts, meas_all)):
            sat = int(r["sat_idx"])
            row = dict.fromkeys(COLUMNS, float("nan"))
            row.update(sat_idx=sat, seed_elevation_deg=90.0 - math.degrees(float(r["z0_seed_rad"])))
            results.append(row)
            if meas is None:
                row["status"] = "propagation_failed"
                continue
            if not meas:
                row["status"] = "no_pass"
                continue
            rr = [m.values[0] for m in meas]
            if sat in saved_rr and len(saved_rr[sat]) == len(rr):
                max_rr_diff = max(max_rr_diff, float(np.max(np.abs(np.array(rr) - saved_rr[sat]))))
            i = next((i for i in range(len(rr) - 1) if rr[i] < 0 <= rr[i + 1]), None)
            if i is None:
                row["status"] = "no_crossing"
                continue
            pairs += [(meas[i].time, list(meas[i].true_state)), (meas[i + 1].time, list(meas[i + 1].true_state))]
            todo.append((row, t_start, meas[0].time, meas[i].time, meas[i + 1].time))
        if not todo:
            continue

        ecef = transform_frame(Frame.EME2000, [p[0] for p in pairs], [p[1] for p in pairs], ECEF_FRAME)
        ca_times, ca_pos = [], []
        for j, (row, t_start, t_aos, ta, tb) in enumerate(todo):
            sa, sb = np.array(ecef[2 * j].array), np.array(ecef[2 * j + 1].array)
            p0, v0, p1, v1 = sa[:3], sa[3:6], sb[:3], sb[3:6]
            h = tb - ta
            f = lambda s: float(np.dot(hermite(p0, v0, p1, v1, h, s)[0] - station,
                                       hermite(p0, v0, p1, v1, h, s)[1]))
            s_ca = brentq(f, 0.0, h, xtol=1e-6) if f(0.0) * f(h) <= 0 else (0.0 if abs(f(0.0)) < abs(f(h)) else h)
            pos, vel = hermite(p0, v0, p1, v1, h, s_ca)
            d = pos - station
            rng_ = float(np.linalg.norm(d))
            de, dn, du = float(d @ east), float(d @ north), float(d @ up)
            ve, vn, vu = float(vel @ east), float(vel @ north), float(vel @ up)
            el = math.asin(max(-1.0, min(1.0, du / rng_)))
            speed = float(np.linalg.norm(vel))
            row.update(
                t_ca_s=ta + s_ca - t_start, t_ca_from_aos_s=ta + s_ca - t_aos, range_ca_m=rng_,
                radius_ca_m=float(np.linalg.norm(pos)),
                elevation_ca_deg=math.degrees(el),
                azimuth_ca_deg=math.degrees(math.atan2(de, dn)) % 360.0,
                speed_ca_mps=speed,
                ang_rate_ca_deg_s=math.degrees(float(np.linalg.norm(np.cross(d, vel))) / rng_ ** 2),
                az_rate_ca_deg_s=math.degrees((dn * ve - de * vn) / (de * de + dn * dn)),
                el_rate_ca_deg_s=math.degrees((vu * rng_ - du * float(d @ vel) / rng_) / (rng_ ** 2 * math.cos(el))),
                heading_ca_deg=math.degrees(math.atan2(ve, vn)) % 360.0,
                turn_sense=1 if (de * vn - dn * ve) > 0 else -1,
                radial_vel_ca_mps=float(pos @ vel) / float(np.linalg.norm(pos)),
                x_ecef_ca_m=pos[0], y_ecef_ca_m=pos[1], z_ecef_ca_m=pos[2],
                vx_ecef_ca_mps=vel[0], vy_ecef_ca_mps=vel[1], vz_ecef_ca_mps=vel[2],
                status="ok",
            )
            ca_times.append(ta + s_ca)
            ca_pos.append(list(pos))

        lla = pos_to_lla(ECEF_FRAME, ca_times, ca_pos)
        for (row, *_), g in zip(todo, lla):
            lat, lon, alt = g.array
            row.update(altitude_ca_m=alt, subsat_lat_deg=math.degrees(lat), subsat_lon_deg=math.degrees(lon))

        done = min(b0 + a.batch, len(rows))
        el = time.time() - t0
        print(f"  {done:>6} / {len(rows)}  ({el / 60:.1f} min, about {el / done * (len(rows) - done) / 60:.0f} min left)",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nSaved {a.out}: {len(ok)} of {len(results)} satellites ok")
    for s in ("no_pass", "no_crossing", "propagation_failed"):
        n = sum(r["status"] == s for r in results)
        if n:
            print(f"  {n} {s}")
    if saved_rr:
        print(f"Check against the saved simulation: largest range-rate difference {max_rr_diff:.2e} m/s "
              f"({'same trajectories' if max_rr_diff < 1e-3 else 'NOT the same trajectories -- see the header'})")
    if ok:
        for col, unit, scale in (("altitude_ca_m", "km", 1e-3), ("elevation_ca_deg", "deg", 1.0),
                                 ("azimuth_ca_deg", "deg", 1.0), ("range_ca_m", "km", 1e-3),
                                 ("speed_ca_mps", "m/s", 1.0), ("ang_rate_ca_deg_s", "deg/s", 1.0),
                                 ("el_rate_ca_deg_s", "deg/s", 1.0)):
            v = np.array([r[col] for r in ok]) * scale
            print(f"  {col:>17}: min {v.min():8.1f}  median {np.median(v):8.1f}  max {v.max():8.1f} {unit}")
        d = np.array([r["elevation_ca_deg"] - r["seed_elevation_deg"] for r in ok])
        print(f"  elevation at closest approach minus generate_params.py's planned elevation: "
              f"median {np.median(d):+.2f} deg, 95% within +/-{np.percentile(np.abs(d), 95):.2f} deg")


if __name__ == "__main__":
    main()
