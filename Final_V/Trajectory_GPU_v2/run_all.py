"""
run_all.py : trains and tests every model on the closest-approach labels, unattended.

    python run_all.py                      clean data only (noise 0)
    python run_all.py --sigmas 0 0.5 1     also with 0.5 and 1.0 m/s range-rate noise
    python run_all.py --fast               skip the CNN (quick test)

Nothing is simulated here. The inputs are an existing CLEAN simulation and its
closest-approach labels:
    --clean-dir   folder with delay_differences.csv, random_orbital_parameters.csv
                  (and initial_state_vectors.csv); default ../Clean_Testing/output
    --ca          closest_approach.csv from ../Closest_Approach/closest_approach.py;
                  default ../Closest_Approach/output/closest_approach.csv
The noise levels share the same orbits, so the same labels apply to every level.

Each level gets its own folder sweep/sigma_X/, where it runs:
    1. copy the clean files (level 0) or add_noise.py (levels above 0)
    2. build_dataset.py     padded range-rate table (same as Noise_Testing)
    3. labels.py            attach the 8 closest-approach labels
    4. forests.py           LightGBM + Random Forest: fit, test
    5. cnn.py               1D CNN on the GPU: fit, test
Then summarize.py collects every level into results_summary.csv and figures.

Safe to stop and restart: a step whose output already exists is skipped.
Everything printed is also written to sweep/run_log.txt.
"""
import argparse, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "sweep")
PY = sys.executable
LOG = None
CLEAN_FILES = ["delay_differences.csv", "random_orbital_parameters.csv", "initial_state_vectors.csv"]
NEEDED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "lightgbm", "torch"]


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n"); LOG.flush()


def preflight():
    import importlib
    missing = []
    for m in NEEDED:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append({"sklearn": "scikit-learn"}.get(m, m))
    if missing:
        say("MISSING PACKAGES: " + ", ".join(missing) + ".  Fix with:  pip install " + " ".join(missing))
        sys.exit(1)


def level_name(s):
    return "sigma_" + f"{s:g}".replace(".", "p")


def run(cmd, cwd, echo=False):
    env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        LOG.write("    " + line); LOG.flush()
        if echo:
            print("    " + line, end="", flush=True)
    return p.wait() == 0


def step(label, cmd, cwd, expect):
    if all(os.path.exists(os.path.join(cwd, e)) for e in expect):
        say(f"  skip  {label} (already done)")
        return True
    t0 = time.time()
    say(f"  start {label}")
    ok = (cmd(cwd) if callable(cmd) else run(cmd, cwd)) and all(os.path.exists(os.path.join(cwd, e)) for e in expect)
    say(f"  {'done ' if ok else 'FAIL '} {label} ({(time.time() - t0) / 60:.1f} min)")
    return ok


def run_level(sigma, a):
    d = os.path.join(SWEEP, level_name(sigma))
    os.makedirs(os.path.join(d, "output"), exist_ok=True)
    say(f"=== noise level {sigma} m/s -> {d}")
    S = lambda f: os.path.join(HERE, f)
    clean_out = os.path.join(SWEEP, level_name(0.0), "output")

    def copy_clean(cwd):
        for f in CLEAN_FILES:
            src = os.path.join(a.clean_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(cwd, "output", f))
        return True

    if sigma == 0:
        first = ("copy clean simulation", copy_clean, ["output/delay_differences.csv"])
    else:
        first = ("add_noise", [PY, S("add_noise.py"), "--clean-dir", clean_out, "--out-dir", "output",
                               "--sigma", str(sigma)], ["output/delay_differences.csv"])
    steps = [first,
             ("build_dataset", [PY, S("build_dataset.py")], ["output/training_table_padded.csv"]),
             ("labels", [PY, S("labels.py"), "--ca", a.ca], ["output/training_table_ca.csv"]),
             ("forests", [PY, S("forests.py")], ["output/forest_COMPLETE.txt"])]
    if not a.fast:
        steps.append(("cnn", [PY, S("cnn.py")], ["output/cnn_COMPLETE.txt"]))
    for label, cmd, expect in steps:
        if not step(f"{level_name(sigma)} / {label}", cmd, d, expect):
            say(f"  level {sigma} stopped at {label}; see sweep/run_log.txt")
            return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.0])
    ap.add_argument("--fast", action="store_true", help="skip the CNN")
    ap.add_argument("--clean-dir", default=os.path.join(HERE, "..", "Clean_Testing", "output"))
    ap.add_argument("--ca", default=os.path.join(HERE, "..", "Closest_Approach", "output", "closest_approach.csv"))
    a = ap.parse_args()
    a.clean_dir, a.ca = os.path.abspath(a.clean_dir), os.path.abspath(a.ca)
    for p in (os.path.join(a.clean_dir, "delay_differences.csv"), a.ca):
        if not os.path.exists(p):
            raise SystemExit(f"Not found: {p}")
    os.makedirs(SWEEP, exist_ok=True)
    LOG = open(os.path.join(SWEEP, "run_log.txt"), "a", encoding="utf-8")
    preflight()
    t0 = time.time()
    say(f"start: sigmas {a.sigmas}, clean data {a.clean_dir}, labels {a.ca}")
    levels = [0.0] + sorted(set(s for s in a.sigmas if s > 0))   # the noisy levels are made from level 0
    ok = {}
    for s in levels:
        ok[s] = run_level(s, a) if (s == 0 or ok.get(0.0)) else False
    run([PY, os.path.join(HERE, "summarize.py")], HERE, echo=True)
    say(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. " + ", ".join(f"{s}: {'ok' if v else 'FAILED'}" for s, v in ok.items()))
