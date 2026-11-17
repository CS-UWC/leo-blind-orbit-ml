"""
add_noise.py : makes one noise level from an existing CLEAN simulation, without simulating again.

    python add_noise.py --clean-dir sweep/sigma_0/output --out-dir output --sigma 0.5

Reads the clean files written by generate_params.py and simv4.py (run with sigma = 0):
    delay_differences.csv, random_orbital_parameters.csv (and initial_state_vectors.csv if present)
and writes, into --out-dir:
    delay_differences.csv           the same table with noise added to every real sample
    random_orbital_parameters.csv   copy with range_rate_error_ms set to --sigma
    initial_state_vectors.csv       plain copy

The noise is the same as simv4.py adds: zero-mean Gaussian noise on the range rate (m/s),
one independent draw per real sample, from numpy.random.default_rng([noise_seed, sat_idx]).
Every noise level uses the same draws scaled by sigma. The table stores delays, and a delay
is range_rate * delta_t / c, so the noise is added as sigma * draw * delta_t / c.
Padding cells (the zeros after a pass ends) are left alone. With --sigma 0 the files are
copied unchanged.
"""
import argparse, csv, os, shutil
import numpy as np

C = 299792458.0   # speed of light [m/s], same value as simv4.py


def main():
    ap = argparse.ArgumentParser(description="Add range-rate noise to a clean simulation.")
    ap.add_argument("--clean-dir", required=True, help="folder holding the clean delay_differences.csv")
    ap.add_argument("--out-dir", default="output", help="folder to write the noisy files into (default: output)")
    ap.add_argument("--sigma", type=float, required=True, help="noise standard deviation on range rate [m/s]")
    ap.add_argument("--noise-seed", type=int, default=2024, help="seed, same default as simv4.py (2024)")
    a = ap.parse_args()
    if a.sigma < 0:
        raise SystemExit("--sigma must be 0 or more")

    src_delay = os.path.join(a.clean_dir, "delay_differences.csv")
    src_params = os.path.join(a.clean_dir, "random_orbital_parameters.csv")
    for p in (src_delay, src_params):
        if not os.path.exists(p):
            raise SystemExit(f"Clean file not found: {p}\nRun generate_params.py and simv4.py (sigma 0) first.")
    if os.path.abspath(a.clean_dir) == os.path.abspath(a.out_dir):
        raise SystemExit("--out-dir must be a different folder from --clean-dir")
    os.makedirs(a.out_dir, exist_ok=True)

    # --- parameters: copy, with the noise level recorded ---
    with open(src_params, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        prm = list(rd)
    i_sig, i_dt = header.index("range_rate_error_ms"), header.index("delta_t_s")
    delta_t = float(prm[0][i_dt])          # one value for the whole batch, as in simv4.py
    for row in prm:
        row[i_sig] = repr(float(a.sigma))
    with open(os.path.join(a.out_dir, "random_orbital_parameters.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(prm)
    src_state = os.path.join(a.clean_dir, "initial_state_vectors.csv")
    if os.path.exists(src_state):
        shutil.copy(src_state, os.path.join(a.out_dir, "initial_state_vectors.csv"))

    # --- delays: add noise to the real samples of every satellite ---
    with open(src_delay, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    i_n, i_dd = header.index("num_valid_samples"), header.index("dd_000")
    n_noisy = 0
    for pos, row in enumerate(rows):
        sat_i, n = int(row[0]), int(row[i_n])
        if sat_i != pos:
            raise SystemExit(f"Row {pos} has sat_idx {sat_i}; expected the rows in order 0, 1, 2, ...")
        if a.sigma > 0 and n > 0:
            z = np.random.default_rng([a.noise_seed, sat_i]).standard_normal(n)
            for k in range(n):
                row[i_dd + k] = repr(float(row[i_dd + k]) + a.sigma * float(z[k]) * delta_t / C)
            n_noisy += 1
    with open(os.path.join(a.out_dir, "delay_differences.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"sigma = {a.sigma} m/s (delta_t = {delta_t} s, seed = {a.noise_seed}): "
          f"noise added to {n_noisy} of {len(rows)} satellites -> {a.out_dir}")


if __name__ == "__main__":
    main()
