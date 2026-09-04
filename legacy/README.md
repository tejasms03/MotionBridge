# Legacy prototypes

Early, superseded versions kept for history — not part of the current app (see the top-level README for what's current).

- **`arm_motion_completion.py`** — the original single-file prototype: OpenCV + MediaPipe, flat exponential-smoothing amplification, no Kalman filter, no biosensors, no game modules. Predates `tracker.py` / `estimator.py` / `completion.py`.
- **`game.py`** — the original 2D OpenCV balloon game, explicitly marked "(legacy)" in the team's own dev log (`docs/PROJECT_LOG.md`) once `virtual_arm.py`'s 3D PyOpenGL version replaced it.

Neither file is imported by anything in the current app.
