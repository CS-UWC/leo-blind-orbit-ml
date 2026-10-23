"""
run_all.py : runs the whole noise sweep, in order, unattended.

Put this file in Noise_Testing next to the other scripts, then from that folder:

    python run_all.py

The satellites are simulated ONCE, with no noise, in sweep/sigma_0/ (generate_params.py, then
simv4.py). That is also the clean (0 m/s) level. Every other noise level gets its own folder
sweep/sigma_X/ whose data is made from that clean simulation by add_noise.py, so the same
orbits and the same noise draws are used at every level and nothing is simulated twice.

Inside each level's folder it then runs:
    1. add_noise.py         noisy copy of the clean delays (levels above 0 only)
    2. build_dataset.py     padded training table
    3. forest_trainer_v3.py LightGBM + Random Forest: fit and save the models
    4. forest_testing.py    score the saved forest models on the held-out test satellites
    5. cnn_trainer_v3.py    1D-CNN: fit and save the networks
    6. cnn_testing.py       score the saved CNNs on the held-out test satellites + scatter figure
There is no cross-validation in this sweep: every model is trained once per level and
scored on the same held-out test satellites at every level.
Then it stacks every level's results into sweep_summary.csv and draws noise_sensitivity.png.
The clean level is always made first, even if 0 is not in --sigmas, because the others need it.

Safe to stop and restart: a step whose output file already exists is skipped.
Everything it prints is also written to sweep/run_log.txt.
A failed step is logged and the sweep moves on to the next level.

Quick test (about a minute):  python run_all.py --n-samples 20 --fast
"""
import argparse, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "sweep")
SIGMAS = [0.0, 0.1, 0.5, 1.0]     # m/s. 0.0 is the clean simulation the other levels are made from.
CLEAN = "sigma_0"                 # folder of the clean simulation
PY = sys.executable
LOG = None


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n"); LOG.flush()


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


def level_name(s):
    return "sigma_" + f"{s:g}".replace(".", "p")


def run(cmd, cwd):
    """Run one command inside cwd, stream its output into the log. True if it exited cleanly."""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        LOG.write("    " + line); LOG.flush()
    return p.wait() == 0


def step(label, cmd, cwd, expect, guard=False):
    if all(os.path.exists(os.path.join(cwd, e)) for e in expect):
        say(f"  skip  {label} (already done)")
        return True
    if guard:   # kill any leftover orbdetpy Java server first (see orbdetpy_guard.py)
        run([PY, "-I", os.path.join(HERE, "orbdetpy_guard.py")], cwd)
    t0 = time.time()
    say(f"  start {label}")
    ok = run(cmd, cwd) and all(os.path.exists(os.path.join(cwd, e)) for e in expect)
    say(f"  {'done ' if ok else 'FAIL '} {label} ({(time.time() - t0) / 60:.1f} min)")
    return ok


def run_level(sigma, n, fast):
    d = os.path.join(SWEEP, level_name(sigma))
    out = os.path.join(d, "output")
    os.makedirs(out, exist_ok=True)
    say(f"=== noise level {sigma} m/s -> {d}")
    # Same train/test split at every level, if a saved one is available.
    split = os.path.join(HERE, "train_test_split.csv")
    if os.path.exists(split) and not os.path.exists(os.path.join(out, "train_test_split.csv")):
        shutil.copy(split, os.path.join(out, "train_test_split.csv"))
    S = lambda f: os.path.join(HERE, f)
    if sigma == 0:      # the one real simulation
        steps = [
            ("generate_params", [PY, S("generate_params.py"), "--sigma", "0", "--out-dir", "output",
                                 "--n-samples", str(n)], ["output/random_orbital_parameters.csv"], True),
            ("simv4", [PY, S("simv4.py"), "--dir", "output"], ["output/delay_differences.csv"], True),
        ]
    else:               # noise added to the clean simulation, no new simulation
        steps = [
            ("add_noise", [PY, S("add_noise.py"), "--clean-dir", os.path.join("..", CLEAN, "output"),
                           "--out-dir", "output", "--sigma", str(sigma)],
             ["output/delay_differences.csv", "output/random_orbital_parameters.csv"], False),
        ]
    steps += [
        ("build_dataset", [PY, S("build_dataset.py")], ["output/training_table_padded.csv"], False),
        ("forest_trainer", [PY, S("forest_trainer_v3.py")], ["output/models/forest/COMPLETE.txt"], False),
        ("forest_testing", [PY, S("forest_testing.py")], ["output/test_results_v3.csv"], False),
    ]
    if not fast:
        steps.append(("cnn_trainer", [PY, S("cnn_trainer_v3.py")], ["output/models/cnn/COMPLETE.txt"], False))
        steps.append(("cnn_testing", [PY, S("cnn_testing.py")], ["output/test_results_cnn_v3.csv"], False))
    for label, cmd, expect, guard in steps:
        if not step(f"{level_name(sigma)} / {label}", cmd, d, expect, guard):
            say(f"  level {sigma} stopped at {label}; moving on")
            return False
    return True


def summarize():
    import pandas as pd
    frames = []
    for name in sorted(os.listdir(SWEEP)):
        if not name.startswith("sigma_"):
            continue
        sigma = float(name[len("sigma_"):].replace("p", "."))
        outd = os.path.join(SWEEP, name, "output")
        # (file, method label to add; None = the file already has a method column)
        sources = [("test_results_v3.csv", "single_split"), ("cv_results_v3.csv", "cv"),
                   ("test_results_cnn_v3.csv", "single_split"), ("cv_results_cnn_v3.csv", "cv")]
        if not os.path.exists(os.path.join(outd, "test_results_v3.csv")):
            sources.insert(0, ("model_results_v3.csv", None))        # older single-file forest results
        if not os.path.exists(os.path.join(outd, "test_results_cnn_v3.csv")):
            sources.append(("model_results_cnn_v3.csv", None))       # older single-file CNN results
        for fn, method in sources:
            f = os.path.join(outd, fn)
            if os.path.exists(f):
                df = pd.read_csv(f)
                df.insert(0, "noise_sigma_ms", sigma)
                df.insert(1, "source_file", fn)
                if method:
                    df.insert(2, "method", method)
                frames.append(df)
    if not frames:
        say("nothing to summarize"); return
    allr = pd.concat(frames, ignore_index=True).sort_values(["noise_sigma_ms"], kind="stable")
    allr.to_csv(os.path.join(HERE, "sweep_summary.csv"), index=False)
    say(f"wrote sweep_summary.csv ({len(allr)} rows)")
    try:   # simple picture: test score vs noise, one line per model, one panel per target
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = allr[allr["method"] == "single_split"]          # held-out test rows
        targets = list(dict.fromkeys(t["target"]))
        fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for ax, tg in zip(axs.ravel(), targets[:6]):
            sub = t[t["target"] == tg]
            col = "circ_corr" if sub["r2"].isna().all() else "r2"
            for m, g in sub.groupby("model"):
                g = g.sort_values("noise_sigma_ms")
                ax.plot(g["noise_sigma_ms"], g[col], marker="o", label=m)
            ax.set_title(f"{tg} ({col})"); ax.set_xlabel("noise sigma (m/s)"); ax.grid(alpha=.3)
        axs.ravel()[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "noise_sensitivity.png"), dpi=130)
        say("wrote noise_sensitivity.png")
    except Exception as e:
        say(f"figure skipped: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--fast", action="store_true", help="skip the CNN (for a quick test)")
    ap.add_argument("--sigmas", type=float, nargs="*", default=SIGMAS)
    a = ap.parse_args()
    os.makedirs(SWEEP, exist_ok=True)
    LOG = open(os.path.join(SWEEP, "run_log.txt"), "a", encoding="utf-8")
    if os.name == "nt":   # stop Windows sleeping while this runs (the screen may still turn off)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    t0 = time.time()
    preflight()
    say(f"sweep start: sigmas {a.sigmas}, {a.n_samples} satellites, python {PY}")
    if any(s < 0 for s in a.sigmas):
        raise SystemExit("sigmas must be 0 or more")
    levels = [0.0] + sorted(set(s for s in a.sigmas if s != 0))   # clean level first, each level once
    if 0.0 not in a.sigmas:
        say("note: the clean level (0 m/s) is also run, because the noisy levels are made from it")
    results = {}
    for s in levels:
        if s != 0 and not results.get(0.0):
            say(f"  level {s} skipped: the clean simulation failed")
            results[s] = False
            continue
        results[s] = run_level(s, a.n_samples, a.fast)
    summarize()
    say(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. " +
        ", ".join(f"{s}: {'ok' if ok else 'FAILED'}" for s, ok in results.items()))
