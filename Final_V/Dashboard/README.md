# Dashboard

Results of the final version: clean data, measurement noise, noise bursts, and a viewer for single test passes.
The "How it works" button explains the pipeline and the direction labels.

It reads the result files in `Final_V` directly (`Trajectory_GPU_v2` and `Solar_Flare`), so keep this folder inside `Final_V`.

## Run

From PowerShell or Command Prompt:

```
cd C:\Users\raeez\blind_orbit_ml\Final_V\Dashboard
..\..\.venv\Scripts\streamlit run app.py
```

Run it from this folder so the colours in `.streamlit\config.toml` are used. It opens at http://localhost:8501.
