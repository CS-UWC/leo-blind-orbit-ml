# Development Record: Dataset Generation Pipeline

**Project:** ML-Enabled Blind Timing-Based Orbit Estimation and Anomaly Detection with LEO Satellite Signals
**Scope of this record:** the dataset generation stage: parameter sampling, delay simulation, and the corrections applied to both.

This document records what was built, what was found to be wrong, and what was changed to fix it. It is written as source material for the report, so each change includes the reasoning behind it. Items that have not been independently verified are marked as such.

---

## 1. Pipeline architecture

The dataset generation pipeline consists of two scripts producing three output files.

```
generate_params.py
    ├─► output/random_orbital_parameters.csv    (Keplerian elements + settings)
    └─► output/initial_state_vectors.csv        (ECEF state vectors, ML labels)
                    │
                    ▼
simulationv2.py  (reads the parameters CSV)
    ├─► output/delay_differences.csv            (delay sequences, ML features)
    └─► output/doppler_all_satellites.png       (combined plot)
```

All three CSVs are aligned row-for-row by `sat_idx`: satellite *i* is row *i* in each file.

### 1.1 Division of responsibility

| Script | Role |
|---|---|
| `generate_params.py` | Samples random Keplerian orbits and converts each to an ECEF state vector via `elem_to_pv` + `transform_frame`. No propagation. |
| `simulationv2.py` | Reads the parameter CSV, propagates each orbit with orbdetpy/Orekit, and extracts the delay observable over the visible pass. |

`simulationv2.py` reads the Keplerian elements, not the state-vector file. This is intentional: orbit propagation must be initialised in an inertial frame, so the simulation performs its own `elem_to_pv` conversion to EME2000 internally. The ECEF state-vector file exists solely as the ML label set and is not consumed by the simulation.

### 1.2 Version predecessor

`simulationv1.py` computed absolute one-way delay, `τ(t) = ρ(t)/c`, for a single hardcoded orbit. This was superseded because absolute delay presupposes a shared time reference between satellite and ground station, which is unavailable in the blind (uncooperative) scenario that defines this project. Version 2 uses a delay-difference observable instead, which a one-way receiver can measure by timing pulse arrivals.

---

## 2. Corrections applied

Four substantive problems were identified and corrected. One was a physics error; the others were numerical-stability, yield and validation issues.

### 2.1 Delay observable computed incorrectly (critical)

**Terminology used in the report.** The original method is called the *differenced formulation* (discarded). The current method is called the *analytic formulation*.

**Symptom.** Delay values were physically implausible and showed no curve structure: jagged, sign-inconsistent, and roughly an order of magnitude too small.

**Differenced formulation (discarded).** The original implementation ran two propagations per satellite. Propagation A started at the reference epoch; propagation B started at `epoch + δt` (δt = 0.001 s). Both used the same initial state vector. Range measurements from the two runs were paired by index and differenced:

```
delay_diff[i] = (range_B[i] − range_A[i]) / c
```

**Root cause (the only reason the method is rejected).** The method is degenerate when both propagations are initialised from the same epoch state. At equal elapsed propagation time the satellite occupies an identical position in both runs, so the only quantity differing between paired samples is the ground station position, displaced by δt of Earth rotation. The resulting quantity is dominated by Earth rotation rather than satellite motion.

**Numerical verification.** An independent check (circular orbit, rotating ground station, no orbdetpy dependency) compared the true range change against what the paired-index scheme computed:

| Time into pass | True Δrange | Method returned | Fraction captured |
|---|---|---|---|
| 0 s | −0.26 m | −0.09 m | 36% |
| 240 s | −1.07 m | −0.06 m | 6% |
| 600 s | −2.21 m | +0.00 m | 0% (sign inverted) |

Supporting magnitudes:

- Ground station surface velocity at the Maui latitude: about **435 m/s**, giving 0.435 m of displacement per δt
- Satellite orbital velocity: about **7,657 m/s**, giving about 7.66 m of displacement per δt
- Values recorded in the dataset: **0.15–1.0 m** of equivalent range change

The recorded values matched the Earth-rotation scale, not the satellite-motion scale, which confirms the diagnosis.

**Claims withdrawn.** An earlier version of this record also attributed the failure to catastrophic cancellation (differencing large range values) and to a √2 amplification of measurement noise. Neither claim was verified and neither is needed: the degeneracy above explains the observed failure on its own. Both have been removed and should not appear in the report.

**Correction (analytic formulation).** Replaced with a single propagation using `MeasurementType.RANGE_RATE`. Orekit computes the line-of-sight range rate analytically at each visible timestep. The delay-difference observable is then obtained by unit conversion:

```
delay_diff = range_rate · δt / c        [seconds]
```

Read as: rate of range change, then range change over δt, then the delay equivalent of that distance. In the report this corresponds to implementing the right-hand side of Equation (2), Δτ ≈ ρ̇ · δt / c.

**Effect on project framing.** None. The observable keeps the same units, meaning and role in the ML pipeline, and the CSV schema is unchanged. Only the method of computing it changed, from an incorrect finite difference to an analytic value. The blind-receiver premise is unaffected: range rate is used as a simulation ground truth from which the receiver-observable delay is derived.

**Verification after correction.** Satellite 0, full pass:

| Time | Range rate |
|---|---|
| 0 s | −4,880 m/s (approaching) |
| 180 s | −3,246 m/s |
| 360 s | +163 m/s (closest approach) |
| 540 s | +3,460 m/s (receding) |
| 630 s | +4,408 m/s |

The curve is smooth, monotonic, and crosses zero once. Across the dataset at that stage, range rates spanned **−7,388 to +7,358 m/s**, within the roughly ±8,000 m/s physical envelope for LEO.

---

### 2.2 Orbital parameter bounds causing propagator failure

**Symptom.** `OrekitException: minimal step size (1.00E-03) reached, integration needs 8.66E-04`, raised via gRPC from the Java backend.

**Root cause.** Perigee radius is `sma · (1 − ecc)`. The original sampling ranges were comments in the placeholder `generate_params.py` (`sma` from 6,571,000 m, `ecc` up to 0.05). They permitted perigee altitudes inside or below the atmosphere; in the worst case the perigee radius was 6,242,450 m, which is below Earth's surface (6,378,137 m). The adaptive-step integrator cannot resolve such trajectories.

**Correction.**

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `sma` | 6,571,000 – 7,371,000 m (~200–1000 km) | **6,778,000 – 7,578,000 m (~400–1200 km)** | Keeps perigee altitude above ~330 km even at maximum eccentricity |
| `ecc` | 0.0 – 0.05 | **0.0 – 0.01** | Near-circular; high eccentricity and low altitude together put the perigee in the atmosphere |
| `inc` | 0 – π | 0 – π (unchanged) | Full inclination range kept for training diversity |

The revised range also matches real operational LEO (ISS about 420 km, Starlink about 550 km). Orbits below this floor decay within days.

**Note on where perigee appears.** Perigee is discussed only in code comments in `generate_params.py`; no executable code computes it. The perigee reasoning is a design justification added during this work, not a quantity the pipeline calculates.

**Additional mitigation.** Each satellite's propagation is wrapped in `try/except`. Any orbit Orekit cannot integrate is logged and recorded as an empty pass, so a single pathological draw cannot abort a batch of thousands.

**Stale file warning.** If warnings show eccentricities above 0.01, `simulationv2.py` is reading a parameters file generated before this fix. Re-run `generate_params.py` before `simulationv2.py` whenever the sampling bounds change.

---

### 2.3 Pass yield

**Problem.** With random RAAN and mean anomaly, each satellite begins at an arbitrary point on an arbitrarily oriented orbit. Only a fraction cross the Maui station within the propagation window. The initial configuration produced one visible pass in ten satellites.

**Distinction.** Widening the search window does not lengthen how long any satellite is visible; that is fixed by altitude and geometry. It only increases the probability that a satellite's (short) overpass falls inside the interval being searched.

**Upper bound on the window.** The window must stay below the shortest orbital period in the sampled range, otherwise the same satellite could be captured on two separate passes within one search, which would not be a single continuous arc. The window was set to **80 min**.

**Pending verification (not in the current report draft).** Kepler's third law, `T = 2π√(a³/μ)`, gives the figures below. These come from the standalone script `check_period.py` (with μ = 3.986004418e14 m³/s²), which is not part of the project files or the supervisor's outline. The Kepler formula and the 92.6 min / 12.6 min figures were deliberately left out of the report draft and are to be checked before the next term submission.

| Semi-major axis | Altitude | Period |
|---|---|---|
| 6,778,000 m | ~400 km | 92.56 min (shortest) |
| 7,578,000 m | ~1200 km | 109.42 min |

Margin between the shortest period and the 80 min window: 12.56 min.

**Measured yield.**

| Search window | Batch size | Non-empty passes | Yield | Status |
|---|---|---|---|---|
| 20 min | 10 | 1 | 10.0% | test run |
| 60 min | 200 | 53 | 26.5% | test run |
| 80 min | 200 | 74 | 37.0% | test run |
| **80 min** | **10,000** | **3,925** | **39.3%** | **full run, verified by the user** |

In the 200-satellite tests, sample counts per non-empty pass were consistent (2–32 samples, mean about 16–17 at 30 s cadence), which shows that window width affects how many passes are captured, not the length of an individual pass.

Empty rows are retained in the CSV with `num_valid_samples = 0` and are removed during data cleaning before training.

---

### 2.4 Measurement noise disabled for physics validation

Noise was set to zero (`RANGE_RATE_ERROR_MS = 0.0`) so the curve shape could be validated on clean data, as requested by the supervisor. With noise off, a correct pass must produce a smooth monotonic curve crossing zero once; any deviation points to a physics error rather than measurement error.

An earlier configuration also had a scale problem: with δt = 0.001 s, the true delay-difference signal corresponds to only about 1–6 m of range change, while the range noise then in use was 10 m, larger than the signal. This explains why the early outputs were dominated by noise.

Realistic noise (order 0.1–1.0 m/s on range rate) is to be reintroduced once the clean-curve validation is accepted.

---

## 3. Additional changes

### 3.1 ECI to ECEF label frame

Keplerian elements are only meaningful in an inertial frame, so `elem_to_pv` is called in **EME2000** (J2000 inertial). The resulting state is then transformed to Earth-fixed coordinates:

```python
state_ecef = transform_frame(Frame.EME2000, epoch, state_eci,
                             Frame.ITRF_CIO_CONV_2010_ACCURATE_EOP)
```

The full six-element vector is transformed, not position alone: ECEF velocity differs from ECI velocity by the Earth-rotation term (ω × r), which `transform_frame` handles correctly.

Output columns: `x_ecef_m, y_ecef_m, z_ecef_m, vx_ecef_ms, vy_ecef_ms, vz_ecef_ms`.

### 3.2 Removal of duplicated parameter output

`simulationv2.py` previously re-wrote its own copy of the orbital parameters, duplicating the generator's output. This write was removed. Parameters now exist in exactly one file (`random_orbital_parameters.csv`); the simulation reads it and produces only `delay_differences.csv`, which aligns to it by `sat_idx`.

### 3.3 Parameter hand-off

`simulationv2.py` originally contained two hardcoded ISS-like satellites in its `__main__` block. It now reads `random_orbital_parameters.csv`, extracting the six Keplerian element columns and the shared settings (station location, δt, cadence, noise level) from the first row. Guard clauses report a missing or empty parameter file rather than failing obscurely.

### 3.4 Instrumentation

- Automatic range-rate band check: prints observed min/max and flags any value exceeding ±8,000 m/s. This check would have caught the §2.1 defect immediately.
- Pass counter: reports non-empty passes as a fraction of the batch.
- Run timer: start marker and elapsed time on completion.
- Per-satellite output tables suppressed by default (`VERBOSE_PER_SAT = False`), because at large batch sizes the volume exceeded terminal scrollback and obscured the summary output.

### 3.5 Reproducibility

Sampling uses a local seeded generator, `random.Random(seed)` with `seed=42`, so a given batch is exactly reproducible. A local instance is used so that seeding does not affect randomness elsewhere. This seed governs dataset generation only and is independent of any `random_state` used in train/test splitting or model training.

---

## 4. Report wording decisions (Section 3, Implementation)

Decisions made while drafting the report text. Use these when reconciling the report with this record.

| Report section | Decision |
|---|---|
| §3.1 | Eq. (1), ρ = ‖r_sat − r_rx‖ and τ = ρ/c, comes from the supervisor's project outline. The structure of Eq. (2) comes from `STRUCTURE.md` and the `simulationv2.py` docstring. Only the right-hand side is implemented in code. |
| §3.1.1 | The two approaches are named the *differenced formulation* and the *analytic formulation*. |
| §3.1.1 | The reason for rejecting the differenced formulation is the degeneracy only (shared initial state, so Earth rotation dominates), supported by the 435 m/s vs 7,657 m/s comparison and the 0–36% toy-check result. The cancellation and √2 arguments are dropped. A shortened version of this paragraph has been proposed; **the user has not yet confirmed it as final.** |
| §3.2 | Opens with the statement that the models are trained on many simulated satellites, each with a known but distinct set of orbital elements, rather than on repeated observations of one trajectory. Table 1 holds the sampling bounds, which trace directly to the constants in `generate_params.py`. |
| §3.2 | The bounds paragraph is written without the perigee formula, without a numeric perigee figure and without dashes. It states that the closest approach must stay well above the atmosphere or the integrator fails, that the semi-major axis range corresponds to roughly 400–1200 km, and that the eccentricity cap keeps the worst-case closest approach at a safe altitude. |
| §3.3 | Station: Maui, lat 0.3614 rad, lon −2.7272 rad, alt 3059.0 m. Pass sampled at 30 s cadence; the sample count depends on time above the horizon. |
| §3.3.1 | The window is stated as 80 minutes, chosen to stay below the shortest orbital period in the sampled range. **The Kepler formula and the 92.6 / 12.6 min figures are left out of this draft** and are to be verified for the next term submission. Yield stated as 3,925 of 10,000 orbits, 39.3%. |

LaTeX notes: Table 1 uses `booktabs` with `[H]` so it does not float; write `\texttt{add\_station()}` rather than `add$_$station()`; use an ASCII hyphen-minus in `$-$2.7272`, not the Unicode minus U+2212; an underfull hbox warning is cosmetic.

**Cross-chat boundary.** §3.3.2 (pass geometry), §3.4 (17 features, `build_dataset.py`) and §3.5 (LightGBM and Random Forest, 70/15/15 split) came from a separate chat and have not been verified here. The same applies to the eccentricity A/B experiment results.

---

## 5. Current configuration

| Setting | Value |
|---|---|
| Ground station | Maui: lat 0.3614 rad, lon −2.7272 rad, alt 3059 m |
| Reference epoch | 2019-05-01T00:00:00 UTC |
| Semi-major axis | 6,778,000 – 7,578,000 m (~400–1200 km altitude) |
| Eccentricity | 0.0 – 0.01 |
| Inclination | 0 – π rad (full range) |
| RAAN / argp / anomaly | 0 – 2π rad (uniform) |
| Measurement type | `MeasurementType.RANGE_RATE` |
| Observable | `delay = range_rate · δt / c` |
| δt (pulse-pair separation) | 0.001 s |
| Sampling cadence | 30 s |
| Search window | 80 min |
| Range-rate noise | 0.0 m/s (off, pending validation) |
| Label frame | ECEF (`ITRF_CIO_CONV_2010_ACCURATE_EOP`) |
| Seed | 42 |
| Yield, 200-satellite test | 37.0% (74 of 200) |
| Yield, full run | 39.3% (3,925 of 10,000), verified by the user |
| Measured throughput | 0.879 s per satellite |

**Runtime projections** (from measured throughput; usable-pass counts use the 39.3% full-run yield):

| Batch | Runtime | Expected usable passes |
|---|---|---|
| 1,000 | ~15 min | ~393 |
| 10,000 | ~2.4 hr | 3,925 (observed) |
| 20,000 | ~4.9 hr | ~7,850 |

---

## 6. Dataset structure

**Features (X):** `delay_differences.csv`. One row per satellite: the delay-difference sequence sampled at 30 s intervals across the visible pass, zero-padded to uniform width, with `num_valid_samples` recording the true unpadded length.

**Labels (y):** `initial_state_vectors.csv`. One row per satellite: the ECEF state vector `[x, y, z, vx, vy, vz]` at the reference epoch.

The learning task maps the observed delay sequence to the orbital state. The direction matters: at inference only the delay sequence is available, so X must be the observable and y the orbit.

**Note on padding.** A value of `0.0` in a padded column denotes "no measurement", but zero Doppler is also a genuine physical state at closest approach. Padding must therefore be masked using `num_valid_samples`, or features engineered from the trimmed sequence, rather than feeding padded columns directly to a model.

---

## 7. Outstanding items

- Verify Kepler's third law, μ and the 92.6 min / 12.6 min figures before the next term submission, then decide whether to add them to §3.3.1.
- Confirm the shortened §3.1.1 paragraph (degeneracy argument only) as final.
- Reintroduce realistic measurement noise once clean-curve validation is accepted.
- Feature engineering from the trimmed sequences (zero-crossing time, extrema, slope at crossing, curvature, arc length).
- Model training and evaluation (LightGBM named by the supervisor, plus a comparison model).
- Anomaly injection and detection (manoeuvre, noise burst, dropout): later project phase.

**Open question for supervisor.** Predicting all six ECEF state components from a single partial overpass may not be equally achievable across components. Some quantities (altitude, inclination) are strongly encoded in the shape of the delay curve; others may be weakly constrained from a single arc. Per-component error is expected to vary, and this is a legitimate reportable finding rather than a defect.

---

## 8. Verification checklist

Applicable to any future run of the pipeline:

1. **Yield:** the printed non-empty pass count is a sensible fraction of the batch (about 37–39% at current settings).
2. **Range-rate band:** reported min/max within ±8,000 m/s; the automatic check prints `OK` or `WARN`.
3. **Curve shape:** with noise off, an individual pass is smooth and monotonic, crossing zero once.
4. **Label magnitudes:** ECEF position magnitude √(x²+y²+z²) is approximately the sampled semi-major axis; speeds are approximately 7,500 m/s.
