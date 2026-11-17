# Session Timeline: Dataset Generation Stage

Chronological record of work completed, in the order it happened.
Companion to `CHANGELOG.md`, which covers the same material organised by topic.

Project: *ML-Enabled Blind Timing-Based Orbit Estimation and Anomaly Detection with LEO Satellite Signals*

Items marked "pending verification" have not been independently checked and are not in the current report draft.

---

## Phase 1: Establishing the pipeline

**1. Decided which script generates the parameter sets.**
Compared `simulationv1.py`, `simulationv2.py`, and the `generate_params.py` placeholder. Concluded `generate_params.py` is the correct place for parameter sampling, feeding `simulationv2.py` as the simulation engine.

**2. Confirmed v2 over v1 as the engine.**
v1 computes absolute delay `τ = ρ/c`, which presupposes a shared clock between satellite and station, unavailable in the blind scenario. v2's delay-difference observable is what a one-way receiver can actually measure. The Keplerian setup from v1 was already present inside v2 (`elem_to_pv`, EME2000), so nothing was lost.

**3. Chose the CSV hand-off design.**
`generate_params.py` writes a parameters CSV; `simulationv2.py` reads it. Selected over passing Python lists directly, so the parameter set becomes a reproducible, inspectable artifact.

**4. Implemented `generate_random_params()`.**
Random sampling of six Keplerian elements, seeded with `random.Random(42)` for reproducibility. Initial test batch: 10 satellites.

**5. Reverted an over-narrow inclination range.**
Inclination had been restricted to 40–75° to guarantee Maui passes. Reverted to the full 0–π range, because narrowing would have given the ML model an unrepresentative slice of orbit space. Empty passes accepted as something to clean downstream.

**6. Created HTML documentation** (`generate_params_docs.html`) recording the parameters, pipeline flow, and design decisions.

---

## Phase 2: Labels and output structure

**7. Added ECEF state-vector generation.**
Each sampled orbit converted to a Cartesian state via `elem_to_pv` (EME2000), then transformed to Earth-fixed coordinates with `transform_frame(..., Frame.ITRF_CIO_CONV_2010_ACCURATE_EOP)`. The full six-element vector is transformed, not position alone, so the Earth-rotation term (ω × r) is handled correctly.

Output: `initial_state_vectors.csv` with columns `x_ecef_m … vz_ecef_ms`.

**8. Wired v2 to read the parameters CSV.**
Replaced v2's two hardcoded ISS-like satellites with a reader that pulls the six element columns plus shared settings from the CSV. Added guard clauses for a missing or empty parameter file.

**9. Removed a duplicated output file.**
v2 had been re-writing its own copy of the orbital parameters, duplicating the generator's output. Deleted that write. Parameters now live in exactly one file; v2 produces only `delay_differences.csv` and the plot.

**10. Documented the feature/label split** in the HTML for supervisor review: X = delay sequences, y = orbit description. Four open questions were listed (label choice, target parameters, padding handling, empty-pass removal).

---

## Phase 3: Yield problem

**11. First data run returned 1 visible pass out of 10 satellites.**
Diagnosed as a yield issue: with random RAAN and mean anomaly, few satellites happen to cross Maui inside the propagation window. This is distinct from any accuracy problem.

**12. Separated yield from precision.**
Established that the search window and batch size control yield, while cadence and noise control the quality of individual measurements. A cadence change (30 to 10 s) made at this point was reverted as irrelevant to yield.

**13. Widened the search window and enlarged the batch.**
Window 20 to 60 min; batch 10 to 200 satellites.

**14. Hit an Orekit integration failure.**
`OrekitException: minimal step size (1.00E-03) reached, integration needs 8.66E-04`.

Root cause: perigee radius `sma · (1 − ecc)` fell inside or below the atmosphere. The original sampling ranges were comments in the placeholder file (sma from 6,571,000 m, ecc up to 0.05). At the extreme of that range the perigee radius was 6,242,450 m, below Earth's surface (6,378,137 m).

Fix: `sma` raised to 6,778,000–7,578,000 m (~400–1200 km), `ecc` capped at 0.01. This keeps perigee above about 330 km at maximum eccentricity and matches real operational LEO. Perigee appears only in code comments, not in executable code.

**15. Added a per-satellite `try/except`** so a single un-propagatable orbit is logged and recorded as an empty pass rather than aborting the whole batch. Added a counter reporting non-empty passes.

**16. Traced a follow-up error to a stale CSV.**
Warnings still showed eccentricities above the new 0.01 cap, indicating `simulationv2.py` was reading a parameters file generated before the bounds fix. Confirmed that the generator must be re-run before the simulation whenever sampling changes.

**17. Clarified why v2 reads the parameters file, not the state vectors.**
Propagation must be initialised in an inertial frame, so v2 performs its own `elem_to_pv` conversion internally. The ECEF file exists solely as ML labels and is not consumed by the simulation.

**18. Measured yield at 60 min: 53 of 200 (26.5%).**

**19. Set the window to 80 min, below the shortest orbital period.**
The window must stay below the shortest orbital period in the sampled range, or the same satellite could be captured on two separate passes within one search. Window set to **80 min**.

*Pending verification:* a Kepler's third law calculation, `T = 2π√(a³/μ)`, gave a shortest period of 92.56 min and a margin of 12.56 min. This came from the standalone script `check_period.py` (μ = 3.986004418e14 m³/s²), which is my own addition and not part of the project files or the supervisor's outline. It was left out of the current report draft and is to be verified before the next term submission.

**20. Measured yield at 80 min: 74 of 200 (37.0%).**

| Window | Batch | Non-empty | Yield | Status |
|---|---|---|---|---|
| 20 min | 10 | 1 | 10.0% | test run |
| 60 min | 200 | 53 | 26.5% | test run |
| 80 min | 200 | 74 | 37.0% | test run |
| 80 min | 10,000 | 3,925 | 39.3% | full run, verified by the user |

Sample counts per pass stayed consistent in the 200-satellite tests (2–32, mean about 16–17), confirming that window width affects acquisition count, not individual pass duration.

---

## Phase 4: Discovery and correction of the delay bug

**21. Supervisor reported the subtraction method was producing inaccuracies**, and requested that noise be disabled until the physics was validated.

**22. Diagnosed the root cause (differenced formulation).**
The two-propagation method initialised propagation B with the epoch state but started it at `epoch + δt`. At each paired index the satellite had therefore accumulated the same elapsed propagation time and occupied an identical position in both runs. The only quantity genuinely differing was the ground station position, rotated by 1 ms of Earth rotation.

The method was measuring Earth rotation, not satellite motion. This degeneracy is the only reason the method is rejected in the report.

**23. Verified numerically** with an independent check (circular orbit, rotating station, no orbdetpy dependency):

| Time into pass | True Δrange | Method returned | Fraction captured |
|---|---|---|---|
| 0 s | −0.26 m | −0.09 m | 36% |
| 240 s | −1.07 m | −0.06 m | 6% |
| 600 s | −2.21 m | +0.00 m | 0% (sign inverted) |

Supporting magnitudes: station surface velocity about 435 m/s (0.435 m per δt) versus satellite orbital velocity about 7,657 m/s (about 7.66 m per δt). Recorded dataset values of 0.15–1.0 m matched the Earth-rotation scale.

Earlier versions of this timeline also claimed catastrophic cancellation and √2 noise amplification in the old method. Those claims were never verified and have been removed. The degeneracy alone explains the observed failure.

**24. Replaced with a single `RANGE_RATE` propagation (analytic formulation).**
Orekit computes line-of-sight range rate analytically. The project observable becomes a unit conversion:

```
delay_diff = range_rate · δt / c
```

Propagation B was deleted entirely. This does not alter the research question: same observable, same units, same CSV schema, same blind-receiver premise. Only the computation method changed.

**25. Disabled measurement noise** (`RANGE_RATE_ERROR_MS = 0.0`) per supervisor instruction, so curve shape could be validated on clean data.

**26. Added an automatic physics check** printing min/max range rate and flagging any value outside ±8000 m/s. This check would have caught the bug immediately.

**27. Verified the correction.** Satellite 0, full pass:

| Time | Range rate |
|---|---|
| 0 s | −4,880 m/s (approaching) |
| 180 s | −3,246 m/s |
| 360 s | +163 m/s (closest approach) |
| 540 s | +3,460 m/s (receding) |
| 630 s | +4,408 m/s |

The curve is smooth, monotonic, with one zero crossing. Range rates at that stage ran from −7,388 to +7,358 m/s, inside the LEO envelope, and about 10 times larger than before, consistent with real satellite motion no longer cancelling out.

**28. Updated the HTML documentation.** Earlier work was relabelled "Stint 1" and a new "Current stint" section was added covering the bug, the numerical proof, and the changes.

---

## Phase 5: Scaling and tooling

**29. Suppressed per-satellite console output.**
At large batch sizes the volume exceeded terminal scrollback and buried the summary figures. Retained behind a `VERBOSE_PER_SAT` flag.

**30. Simplified run reporting** to a start marker, elapsed time, and finish marker.

**31. Measured throughput: 175.8 s for 200 satellites = 0.879 s per satellite.**

| Batch | Projected runtime | Expected usable passes |
|---|---|---|
| 1,000 | ~15 min | ~393 |
| 10,000 | ~2.4 hr | 3,925 (observed, verified by the user) |
| 20,000 | ~4.9 hr | ~7,850 |

**32. Evaluated and rejected GPU acceleration.**
The bottleneck is orbdetpy/Orekit propagation, which runs on the JVM on CPU, so a GPU would sit idle. Google Colab's free tier (about 2 vCPU) would likely be slower than the local machine.

**33. Evaluated batched propagation.**
`propagate_orbits` accepts a list and the underlying Java class is `ParallelPropagation`, so batching could use more cores. Deferred: at about 2.4 hr for 10,000 the saving did not justify the added complexity and the risk of one bad orbit affecting a whole batch.

**34. Model selection discussed.**
At 74 samples, tree ensembles and KNN were appropriate and deep learning was not. With about 3,900 passes from a 10,000-satellite run, a 1D CNN becomes viable. The supervisor named **LightGBM**. The plan carried into the report is LightGBM plus Random Forest.

Noted as a prerequisite: padded zeros must be masked using `num_valid_samples`, since `0.0` ambiguously denotes both "no measurement" and genuine zero Doppler at closest approach.

**35. Built the LEO TEK Streamlit dashboard** (`leo_tek_dashboard.py`). It reads the three output CSVs without re-running any propagation. Sections: telemetry readouts, per-satellite Doppler curve, all-pass overlay, orbit distributions, feature/label preview, and a model section awaiting training results.

**36. Confirmed power settings** required for a multi-hour unattended run (sleep disabled; screen-off is harmless).

---

## Phase 6: Report drafting (Section 3, Implementation)

**37. Traced the equations in §3.1 to their sources.**
Eq. (1), ρ = ‖r_sat − r_rx‖ with τ = ρ/c, comes from the supervisor's project outline PDF. The structure of Eq. (2) comes from `STRUCTURE.md` and the `simulationv2.py` docstring. Only the right-hand side is implemented in code (`values = [rr * delta_t / c for rr in range_rates]`).

**38. Corrected an inaccurate range figure.**
A draft quoted a range of about 7×10⁶ m. That value is the semi-major axis, not the station-to-satellite range (which is about 10⁵–10⁶ m). It became moot once the cancellation argument was dropped.

**39. Renamed the two approaches.**
The old method is the *differenced formulation* and the current one is the *analytic formulation*, replacing earlier wording.

**40. Dropped the cancellation and √2 claims** from §3.1.1 after checking them. Neither was verified. The paragraph is reduced to the degeneracy argument, supported by the 435 m/s vs 7,657 m/s comparison and the toy-check result (0–36% recovered, sign flipping). A shortened version of this paragraph was proposed; **the user has not yet confirmed it as final.**

**41. Drafted §3.2 (dataset design).**
Opening: the models are trained on many simulated satellites with known but distinct orbital elements, rather than repeated observations of one trajectory. Table 1 values trace to the constants in `generate_params.py`. The bounds paragraph was rewritten without the perigee formula, without the 330 km figure and without dashes.

**42. Drafted §3.3 (ground station) and §3.3.1 (observation window).**
Station: Maui, lat 0.3614 rad, lon −2.7272 rad, alt 3059.0 m, sampled at 30 s. For §3.3.1, the Kepler formula was left out of this draft at the user's request; the window is described as 80 minutes, chosen to remain below the shortest orbital period, with a yield of 3,925 of 10,000 (39.3%).

**43. Fixed LaTeX issues in Overleaf.**
Table 1 uses `booktabs` with `[H]` so it does not float. `add$_$station()` was replaced with `\texttt{add\_station()}`. A Unicode minus (U+2212) was replaced with an ASCII hyphen-minus in `$-$2.7272`. An underfull hbox warning was identified as cosmetic.

**44. Identified the cross-chat boundary.**
§3.3.2 (pass geometry), §3.4 (17 features, `build_dataset.py`) and §3.5 (LightGBM and Random Forest, 70/15/15 split), together with the eccentricity A/B experiment, came from a separate chat and were not verified here.

---

## Current configuration

| Setting | Value |
|---|---|
| Ground station | Maui: lat 0.3614 rad, lon −2.7272 rad, alt 3059 m |
| Epoch | 2019-05-01T00:00:00 UTC |
| Semi-major axis | 6,778,000 – 7,578,000 m |
| Eccentricity | 0.0 – 0.01 |
| Inclination | 0 – π rad |
| RAAN / argp / anomaly | 0 – 2π rad |
| Measurement type | `MeasurementType.RANGE_RATE` |
| Observable | `delay = range_rate · δt / c` |
| δt | 0.001 s |
| Cadence | 30 s |
| Search window | 80 min |
| Range-rate noise | 0.0 m/s (off, pending validation) |
| Label frame | ECEF (`ITRF_CIO_CONV_2010_ACCURATE_EOP`) |
| Seed | 42 |
| Yield, 200-satellite test | 37.0% |
| Yield, full run | 39.3% (3,925 of 10,000), verified by the user |
| Measured throughput | 0.879 s/satellite |

---

## Outstanding

- Verify Kepler's third law, μ and the 92.6 min / 12.6 min figures before the next term submission, then decide whether to add them to §3.3.1.
- Confirm the shortened §3.1.1 paragraph as final.
- Reintroduce realistic measurement noise once clean-curve validation is accepted.
- Feature engineering from trimmed sequences (zero-crossing time, extrema, slope at crossing, curvature, arc length).
- Model training and evaluation (LightGBM plus Random Forest).
- Anomaly injection and detection (manoeuvre, noise burst, dropout).
- Verify §3.3.2–§3.5 against the other chat's material if the same level of checking is wanted.

**Open question for supervisor:** predicting all six ECEF state components from a single partial overpass may not be equally achievable across components. Altitude and inclination are strongly encoded in the delay curve; others may be weakly constrained from one arc. Varying per-component error is expected and is a reportable finding rather than a defect.
