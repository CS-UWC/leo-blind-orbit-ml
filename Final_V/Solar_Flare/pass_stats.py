"""
pass_stats.py : how long are the simulated passes, and do they fit the observation window?

Reads the clean (sigma 0) delay_differences.csv and reports, for the 10,000 passes:
  - the number of readings per pass
  - the duration of each pass, from its first reading to its last
  - how many passes start or end within one cadence step of the edge of the window

Run from a folder that contains delay_differences.csv:

    python pass_stats.py

or point it at the file:

    python pass_stats.py --delays path/to/delay_differences.csv

How the duration is worked out
  The readings of a pass are consecutive cadence steps (the contiguous column is 1 for every
  satellite), starting at t_aos_s, the time of the first reading measured from the start of
  that satellite's own window. So
      duration = (number of readings - 1) * cadence
      end time = t_aos_s + duration
  Orekit clips the last reading of a pass that is still in progress when the window ends, so
  that final interval can be shorter than the cadence. The duration is therefore accurate to
  within one cadence step (30 s).

Writes (in output/)
  pass_stats.csv    one row per satellite: sat_idx, num_readings, duration_min, start_s, end_s
  pass_summary.csv  the summary numbers that are printed
"""
import argparse
import os

import numpy as np
import pandas as pd

CADENCE_S = 30.0          # s, measurement cadence (generate_params.py, simv3.py)
WINDOW_S = 24 * 60.0      # s, observation window per satellite (simv3.py)
OUT = "output"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", default="delay_differences.csv")
    a = ap.parse_args()
    if not os.path.exists(a.delays):
        raise SystemExit(f"{a.delays} not found. Run this next to the clean sigma 0 delay_differences.csv, "
                         f"or pass --delays <path>.")
    os.makedirs(OUT, exist_ok=True)

    d = pd.read_csv(a.delays, usecols=["sat_idx", "num_valid_samples", "t_aos_s", "contiguous"])
    n = d["num_valid_samples"].to_numpy()
    dur_s = (n - 1) * CADENCE_S
    start = d["t_aos_s"].to_numpy()
    end = start + dur_s

    per_sat = pd.DataFrame({"sat_idx": d["sat_idx"], "num_readings": n,
                            "duration_min": dur_s / 60.0, "start_s": start, "end_s": end})
    per_sat.to_csv(os.path.join(OUT, "pass_stats.csv"), index=False)

    near_start = int((start < CADENCE_S).sum())
    near_end = int((end > WINDOW_S - CADENCE_S).sum())
    rows = [
        ("satellites", len(d)),
        ("satellites with a continuous pass (contiguous = 1)", int((d["contiguous"] == 1).sum())),
        ("readings per pass: min", int(n.min())),
        ("readings per pass: median", float(np.median(n))),
        ("readings per pass: max", int(n.max())),
        ("duration (min): min", round(float(dur_s.min() / 60), 1)),
        ("duration (min): median", round(float(np.median(dur_s) / 60), 1)),
        ("duration (min): mean", round(float(dur_s.mean() / 60), 1)),
        ("duration (min): 90th percentile", round(float(np.percentile(dur_s, 90) / 60), 1)),
        ("duration (min): max", round(float(dur_s.max() / 60), 1)),
        ("passes longer than 12 min", int((dur_s > 12 * 60).sum())),
        ("window length (min)", WINDOW_S / 60),
        ("earliest first reading (s after window start)", round(float(start.min()), 1)),
        ("latest last reading (s after window start)", round(float(end.max()), 1)),
        (f"passes starting within {CADENCE_S:.0f} s of the window start", near_start),
        (f"passes ending within {CADENCE_S:.0f} s of the window end", near_end),
    ]
    pd.DataFrame(rows, columns=["measure", "value"]).to_csv(os.path.join(OUT, "pass_summary.csv"), index=False)

    print(f"Read {len(d)} passes from {a.delays}\n")
    for name, val in rows:
        print(f"  {name:<55} {val}")
    print(f"\nSaved per-satellite numbers and the summary to {OUT}/")
