"""
run_clean.py : runs the whole CLEAN pipeline in order, unattended.

Put this file (and orbdetpy_guard.py) in Clean_Testing, then:

    cd C:\\Users\\raeez\\blind_orbit_ml\\Clean_Testing
    python run_clean.py

Order: generate_params -> simv3 -> build_dataset -> forest_trainer_v3 -> forest_testing -> cnn_trainer_v3 -> cnn_testing -> plot_predictions
Uses your scripts exactly as they are. Results land in Clean_Testing\\output (files with the
same names are overwritten). Everything printed is also saved to run_clean_log.txt.
If a step fails, it stops there (later steps need its output).

Run only part of it (a step name is the script name, with or without .py):

    python run_clean.py --from forest_trainer_v3   that step and every step after it
    python run_clean.py --only forest_testing      just that one step

With no options it runs every step. Steps that are skipped are not checked: a step needs the
output of the earlier steps to already be in output\\.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)          # so "from Clean_Testing.splits import ..." works
PY = sys.executable
LOG = open(os.path.join(HERE, "run_clean_log.txt"), "a", encoding="utf-8")

env = dict(os.environ)
env["PYTHONPATH"] = PARENT + os.pathsep + env.get("PYTHONPATH", "")
env["MPLBACKEND"] = "Agg"               # simv3 calls plt.show(); this stops it waiting for a window
env["PYTHONIOENCODING"] = "utf-8"


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


NEEDED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "lightgbm", "torch", "orbdetpy", "psutil"]


def preflight(need=NEEDED):
    """Fail in seconds, not after hours, if a package is missing in THIS python."""
    import importlib
    missing = []
    for m in need:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append({"sklearn": "scikit-learn"}.get(m, m))
    if missing:
        say("MISSING PACKAGES in " + PY + ": " + ", ".join(missing))
        say("Fix with:  pip install " + " ".join(missing) + "   (with your .venv active), then run again.")
        sys.exit(1)
    say("all required packages found")


def run(script, extra=()):
    p = subprocess.Popen([PY, *extra, os.path.join(HERE, script)], cwd=HERE, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        LOG.write("    " + line); LOG.flush()
    return p.wait() == 0


STEPS = [  # (script, clear orbdetpy servers first?)
    ("generate_params.py", True),
    ("simv3.py", True),
    ("build_dataset.py", False),
    ("forest_trainer_v3.py", False),
    ("forest_testing.py", False),
    ("cnn_trainer_v3.py", False),
    ("cnn_testing.py", False),
    ("plot_predictions.py", False),
]

def pick_steps(argv=None):
    """Apply --from / --only to STEPS. Exits with a message on a bad step name."""
    ap = argparse.ArgumentParser(description="Run the clean pipeline, all of it or part of it.")
    ap.add_argument("--from", dest="start", metavar="STEP", help="run this step and every step after it")
    ap.add_argument("--only", metavar="STEP", help="run just this one step")
    a = ap.parse_args(argv)
    if a.start and a.only:
        ap.error("use --from or --only, not both")
    names = [s for s, _ in STEPS]

    def find(x):
        x = x if x.endswith(".py") else x + ".py"
        if x not in names:
            ap.error(f"unknown step '{x}'. Steps are: " + ", ".join(n[:-3] for n in names))
        return names.index(x)

    if a.only:
        return [STEPS[find(a.only)]]
    if a.start:
        return STEPS[find(a.start):]
    return STEPS


if __name__ == "__main__":
    steps = pick_steps()
    if os.name == "nt":                 # keep Windows awake while this runs
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    t_all = time.time()
    say(f"clean run start, python {PY}, steps: " + ", ".join(s[:-3] for s, _ in steps))
    preflight()
    for script, guard in steps:
        if guard and os.path.exists(os.path.join(HERE, "orbdetpy_guard.py")):
            run("orbdetpy_guard.py", extra=("-I",))   # kill leftover orbdetpy Java server
        t0 = time.time(); say(f"start {script}")
        ok = run(script)
        say(f"{'done ' if ok else 'FAIL '} {script} ({(time.time() - t0) / 60:.1f} min)")
        if not ok:
            say("stopped. See the lines above this one in run_clean_log.txt for the error.")
            sys.exit(1)
    say(f"ALL DONE in {(time.time() - t_all) / 3600:.2f} h")
