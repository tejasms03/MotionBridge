# MotionBridge — Sensor-Driven Motion Amplification and Gamified Rehabilitation for Post-Stroke Upper-Limb Motor Recovery

> **Team project** — Bao Do, Anudamodaran Kumarakurup, Tejas Mathur Sriganesh.
> Advanced Mechatronics (ROB-GY 6103), New York University, Spring 2026.
>
> This repository is a copy published under my own account for portfolio purposes.
> The original upload lives at [baodo0710/MotionBridge](https://github.com/baodo0710/MotionBridge)
> and remains the canonical source.


MotionBridge is a **camera-based upper-limb motion sensing and amplification system** integrating monocular pose estimation, per-joint Kalman filtering, and kinematic motion completion to transform partial arm movements into full on-screen gestures that drive four therapeutic game paradigms. A dual-axis inertial and surface-electromyography sensor pack provides real-time movement-quality quantification and muscle-activation triggering. The system is designed to enable **clinically meaningful, high-repetition arm practice** for stroke survivors outside supervised clinical sessions, using only a commodity USB webcam and an embedded microcontroller.

**Team:** Bao Do, Anudamodaran Kumarakurup, Tejas Mathur Sriganesh

---

## Clinical Motivation and Problem Statement

Stroke is the leading cause of long-term adult disability in the United States, with 55–75% of survivors exhibiting persistent upper-limb motor impairment. Recovery depends on **high-repetition, goal-directed movement practice** — yet adherence to unsupervised home exercise programs remains consistently low due to cognitive fatigue, lack of intrinsic motivation, and the absence of quantitative performance feedback. Existing rehabilitation gaming systems fail to bridge this gap across three dimensions:

- **The Accessibility Barrier** — Handheld controllers assume intact grip strength and fine finger dexterity, prerequisites that many hemiparetic stroke survivors do not possess.
- **The Adherence Barrier** — Unstructured repetitive exercise is cognitively taxing and lacks the reward scheduling necessary to sustain engagement without clinical supervision.
- **The Economic Barrier** — Clinical-grade motion capture systems (marker-based optical tracking, depth-sensing arrays such as Microsoft Kinect, and commercial platforms such as Pablo or Tyromotion) are prohibitively expensive and geographically constrained to institutional settings.

MotionBridge addresses these barriers through a **technique-centric design philosophy**: the patient's own arm serves as the controller (eliminating grip requirements), partial movements are kinematically amplified into complete on-screen gestures (ensuring rewarding feedback even with severe residual weakness), and the entire sensing stack operates on a commodity webcam and an ESP32 microcontroller — orders of magnitude lower in cost than clinical-grade alternatives.

---

## System Architecture

```
USB Webcam (30 Hz) ──▶ tracker.py (MediaPipe Pose, 33 landmarks → 5 right-arm joints)
                              │
                              ▼
                      estimator.py (per-joint 6-state Kalman filter, Mahalanobis outlier rejection)
                              │
                              ▼
                      completion.py (anchored amplification × velocity lead × per-axis weighting)
                              │
                              ▼
               main.py ──▶ pygame / PyOpenGL window (3D arm avatar + therapeutic game paradigm)
                              ▲
                              │
ESP32 ──UART 115200 baud, 100 Hz──▶ arduino.py (dual MPU6050 IMU + MyoWare EMG → 
                                                   steadiness metric, EMG burst trigger)
```

The camera and biosensor subsystems are **architecturally decoupled by design**: the monocular pose pipeline drives the on-screen avatar through a camera-only Kalman filter (see [Filter Design Rationale](#filter-design-rationale) for the methodological justification), while the IMU/EMG side channel provides an independent **movement-quality metric** (steadiness) and **muscle-activation triggering** (EMG burst detection). This separation ensures that neuromuscular effort not visually manifesting as joint displacement still registers as meaningful therapeutic input.

---

## Computer Vision Pipeline (`tracker.py`)

A commodity USB webcam feeds MediaPipe's `PoseLandmarker` (VIDEO mode) at approximately 25–30 frames per second on the Jetson CPU. The tracker extracts the right shoulder, elbow, wrist, and index fingertip from MediaPipe's 33-landmark skeletal model, yielding a 5-joint kinematic chain for the upper extremity.

---

## Per-Joint State Estimation (`estimator.py`)

A **6-state constant-velocity Kalman filter** (`[px, py, pz, vx, vy, vz]`) is instantiated independently for each joint. The filter predicts forward at the camera frame rate and updates only when the incoming measurement passes a **Mahalanobis distance gate**, rejecting single-frame MediaPipe outliers (e.g., self-occlusion, motion blur) by coasting on the velocity state until a statistically consistent observation is received.

| Parameter | Value | Design Rationale |
|---|---|---|
| `q_vel` | 0.08 | Process noise on velocity — reduced from 0.30 to suppress jitter while preserving responsiveness to genuine acceleration transients |
| `r_cam` (XY) | 0.012 | Camera measurement noise in the image plane — low because MediaPipe's lateral landmark accuracy is high under typical illumination |
| `r_cam_z` | 0.20 | Camera measurement noise in depth — an order of magnitude higher, reflecting the fundamental ambiguity of monocular depth estimation |
| `init_p` | 0.01 | Initial position covariance, ensuring rapid convergence from an arbitrary startup pose |
| `gate_thresh` | 9.2 | Mahalanobis gate threshold (≈χ²₆, 0.975) — frames exceeding this distance are classified as outliers and discarded |

---

## Motion Completion and Amplification (`completion.py`)

The core therapeutic mechanism is a **kinematic amplification transform** that maps the patient's reduced-amplitude movement into a full-range on-screen gesture:

```
completed = neutral + (filtered − neutral) × amplification_factor
```

Default amplification is **3.0×** — a 5 cm wrist excursion becomes 15 cm on screen. The transform is augmented by three design features:

- **Velocity lead compensation** — A predictive term nudges the rendered position slightly ahead in the instantaneous direction of motion, canceling the Kalman filter's residual phase lag without reintroducing high-frequency jitter.
- **Per-axis differential weighting** — Full amplification is applied vertically (the upward-reach motion most critically impaired in post-stroke hemiparesis), reduced amplification laterally, and minimal amplification in depth (reflecting the elevated `r_cam_z` measurement uncertainty).
- **Workspace clamping** — The shoulder serves as a fixed anchor; only the elbow, wrist, and fingertip are amplified. Output is bounded to the patient's calibrated reachable workspace, ensuring amplification cannot project the avatar outside the screen or into physiologically impossible configurations.
- **Live tunability** — Amplification gain is adjustable mid-session via `+`/`-` key input, enabling a therapist or the patient to identify and lock in the minimal effective amplification that produces a rewarding gesture.

---

## Biosensing Interface (`arduino.py` + `esp32_firmware/`)

An ESP32 microcontroller reads two MPU6050 inertial measurement units on a shared I2C bus — wrist at address `0x68` (AD0→GND), elbow at `0x69` (AD0→3.3V) — plus a MyoWare 2.0 surface EMG sensor on a 12-bit analog-to-digital converter pin. The firmware streams a 13-field comma-separated telemetry packet over USB serial at 115200 baud, 100 Hz:

```
emg, wax,way,waz, wgx,wgy,wgz, eax,eay,eaz, egx,egy,egz
```

The `ArduinoReader` module parses this stream non-blocking and **tolerates total sensor absence** — the game loop continues uninterrupted without biosensor data, ensuring graceful degradation. The IMU data currently drives only the **steadiness metric** (rolling standard deviation of the combined wrist-and-elbow accelerometer magnitude over a 20-sample sliding window); it is intentionally *not* fused into the pose Kalman filter (see [Filter Design Rationale](#filter-design-rationale)). EMG bursts exceeding a configurable threshold (default 500 on the 0–4095 ADC range, with an 8-second inter-trigger cooldown) activate an in-game special action, providing positive reinforcement for neuromuscular engagement independent of visible joint displacement.

---

## Therapeutic Game Paradigms

Four pygame / PyOpenGL games, selected through an in-application launcher, share the unified Kalman → completion pipeline and a common heads-up display (steadiness bar, EMG status indicator):

| Game | Module | Therapeutic Target |
|---|---|---|
| **Monster Defense** | `virtual_arm.py` | Full range of motion — the patient sweeps the virtual arm through any direction to repel approaching monsters, encouraging multi-planar shoulder and elbow recruitment |
| **Balloon Rise** | `game_rise.py` | Anti-gravity upward reach — the motion most frequently impaired in post-stroke hemiparesis; the patient pops rising balloons before they escape the workspace, promoting active elevation against gravity |
| **Reach & Hold** | `game_targets.py` | Terminal-position accuracy and endurance — the patient holds a glowing target within a spatial tolerance zone for 1.5 seconds, training isometric control and sustained muscle activation |
| **Fish Swim** | `game_fish.py` | Forearm pronation/supination and sustained shoulder-angle maintenance — arm orientation in 3D space steers a fish character, encouraging controlled rotation and postural stability |

---

## Hardware Platform and Deployment

Developed on Windows and deployed on an **NVIDIA Jetson Orin** for demonstration. Computational load is partitioned across CPU and GPU: pose estimation, Kalman filtering, biosensor parsing, and game logic execute on CPU; all three-dimensional rendering (arm avatar, joint visualization, lighting, game graphics) is accelerated on GPU via PyOpenGL. MediaPipe runs CPU-only — GPU-accelerated inference requires a CUDA-enabled source build that was not pursued, as the CPU pipeline already satisfies the real-time latency budget for therapeutic interaction.

---

## Known Limitations and Future Work

- **IMU-pose fusion remains disabled** pending a rigorous extrinsic calibration between the physical IMU mounting frame and the camera world frame. The code path is fully plumbed (`arduino=...` parameter threaded through the pipeline), but `main.py` currently invokes `completer.complete(..., imu=None)` unconditionally. Re-enabling fusion requires a calibration procedure (e.g., least-squares axis alignment or manual rotation matrix) that was not completed within the project timeline.
- **The `use_lstm` option** in `config.yaml` and the corresponding LSTM blending path in `completion.py` reference an `lstm_predictor` module that was not implemented. A learned motion-completion model (e.g., LSTM autoencoder predicting future joint positions from a 30-frame history) remains on the research roadmap. Leave `use_lstm: false`.
- **Amplification and filter parameters are global**, not patient-specific. A clinical deployment would require per-patient calibration of amplification gain, workspace bounds, and Kalman process noise to account for individual kinematic range and movement variability.
- **The steadiness metric is display-only** — it is not logged or trended per session, limiting its utility for longitudinal clinical progress tracking. Adding session-level data persistence and visualization is a priority for future development.
- **No formal clinical validation** with stroke survivors has been conducted. The system was evaluated with sighted participants under blindfold conditions; threshold sensitivity, amplification acceptability, and game engagement require validation with the target clinical population.

---

## References

- Feigin VL et al., *World Stroke Organization: Global Stroke Fact Sheet 2025*, Int J Stroke 20(2):132–144, 2025.
- Laver KE et al., *Virtual reality for stroke rehabilitation*, Cochrane Database of Systematic Reviews, 2017.
- Subramanian SK et al., *Arm motor recovery using a virtual reality intervention in chronic stroke*, Neurorehabilitation and Neural Repair, 2010.
- Norouzi-Gheidari N et al., *Effects of robot-assisted therapy on stroke rehabilitation*, Journal of Rehabilitation Research and Development, 2012.
- Luinge HJ, Veltink PH, *Measuring orientation of human body segments using miniature gyroscopes and accelerometers*, Medical, Biological Engineering and Computing, 2005.
