# Running Trajectory_GPU_v2 on Colab

This pipeline trains LightGBM, Random Forest and the 1D CNN to predict where the satellite is, and how it is moving, at its closest approach to the station. The CNN trains on the GPU.

No simulation runs here, so no Java or orbdetpy is needed. The pipeline reads two existing inputs from the same Drive folder:
- `../Clean_Testing/output/`: the clean 10,000-satellite simulation
- `../Closest_Approach/output/closest_approach.csv`: the closest-approach labels

## The labels (`common.py`)

| Label | Kind | Meaning |
|---|---|---|
| `label_alt_ca_km` | number | geodetic altitude at closest approach |
| `label_elev_ca_deg` | number | elevation seen from the station |
| `label_az_fold_deg` | number | azimuth folded to 0–90°: 0 = due north or south of the station, 90 = due east or west |
| `label_ang_rate_ca_deg_s` | number | angular rate of the satellite across the sky |
| `label_radial_vel_ca_mps` | number | rate of change of distance from Earth's centre (the eccentricity signal) |
| `label_eastbound` | 1/0 | moving east (1) or west (0): prograde vs retrograde |
| `label_northbound` | 1/0 | moving north (1) or south (0) |
| `label_passes_east` | 1/0 | passes east (1) or west (0) of the station, across its track: station on the left of a northbound pass, or the right of a southbound one |

The five numbers leave 8 candidate passes: 4 mirror-image positions around the station (NE, SE, SW, NW) × 2 directions of motion. Each yes/no label resolves one ambiguity:

| Label | Ambiguity | Expected from the Doppler curve | Candidates left |
|---|---|---|---|
| `eastbound` | prograde vs retrograde | learnable (Earth's rotation shows it) | 8 → 4 |
| `northbound` | the pass vs its mirror image through the equator | coin flip (an exact symmetry for an equatorial station) | 4 → 2 |
| `passes_east` | which side of its track the satellite passes the station | weakly learnable | 2 → 1 |

`../Closest_Approach/reconstruct_check_v2.py` confirms the 8 → 4 → 2 → 1 chain for all 10,000 clean satellites. From the 8 label values alone it rebuilds the true Earth-fixed position and velocity at closest approach, and so the trajectory, to rounding error. The exceptions are 204 satellites within about 1° of a compass axis, which keep two nearly identical candidates. The orbit's orientation in space (RAAN) also needs the absolute time of the pass, which blind timing doesn't give.

## Steps

1. **Runtime:** Runtime > Change runtime type > **T4 GPU** (High-RAM if available).
2. **Mount Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Go to the folder:**
   ```python
   %cd "/content/drive/MyDrive/teaching/Spring2026Courses/projects/Raeez/Code/V3 V4/Trajectory_GPU_v2"
   ```
4. **Install packages** (numpy, torch, pandas, scipy and matplotlib are already on Colab):
   ```python
   !pip install -q lightgbm scikit-learn
   ```
5. **Check the GPU:**
   ```python
   import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
   ```
6. **Quick test** (trees only, no CNN):
   ```python
   !python run_all.py --fast
   ```
   It should end with `ALL DONE ... 0.0: ok`.
7. **Full run** (adds the CNN; the steps already done are skipped):
   ```python
   !python run_all.py
   ```
   To also train on noisy data, add the noise levels: `!python run_all.py --sigmas 0 0.5 1`. The noisy copies are made from the clean simulation; nothing is simulated.

Progress is printed in the cell and saved to `sweep/run_log.txt`. In a Colab terminal you can follow it with `tail -f sweep/run_log.txt`. If Colab disconnects, repeat steps 2–5 and run the same command; it continues from the first unfinished step.

## Results (in this folder)

- `results_summary.csv`: every label × model (× noise level). For the number labels: RMS, median and 90th-percentile error, bias, R², and the "guess the mean" baseline. For the 1/0 labels: accuracy and the "guess the more common answer" baseline.
- `results.png`: predicted vs true for each number label, and accuracy of each model for each 1/0 label.
- `results_vs_noise.png`: only when more than one noise level was run.
- `sweep/sigma_X/output/`: the per-satellite test predictions and the saved CNN models for each level.

## Time

The trees take about 20 minutes per level, judging from the noise sweep. The CNN trains all 8 networks together on the GPU, probably about 30–40 minutes per level on a T4. That estimate comes from the noise sweep (6 networks per level) and hasn't been measured for this pipeline.
