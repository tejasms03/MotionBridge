"""
Arm Motion Completion — Stroke Rehab Prototype
================================================
How it works:
  1. CALIBRATION (3s): Hold your arm still. This sets the neutral position.
  2. LIVE: Move your arm slightly. The green skeleton = your real movement.
           The blue skeleton = the completed/amplified version.

Controls:
  R — recalibrate neutral position
  +/- — increase/decrease amplification factor
  Q — quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
AMPLIFICATION      = 3.0   # How much to scale movement (increase for weaker patients)
CALIBRATION_SECS   = 3     # How long to hold still during calibration
SMOOTHING_ALPHA    = 0.4   # Exponential smoothing factor (0=no smoothing, 1=no lag)

# MediaPipe right arm landmark indices
SHOULDER = 12
ELBOW    = 14
WRIST    = 16
FINGER   = 20  # right pinky tip (rough hand marker)

ARM_CHAIN = [SHOULDER, ELBOW, WRIST, FINGER]

# Colors (BGR)
COLOR_REAL      = (0,   220,  80)   # green  — your actual movement
COLOR_COMPLETED = (255, 160,   0)   # blue   — amplified/completed motion
COLOR_NEUTRAL   = (100, 100, 100)   # grey   — neutral rest position
COLOR_TEXT      = (255, 255, 255)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_arm_landmarks(results, image_shape):
    """Extract right arm joint pixel coords from MediaPipe results."""
    if not results.pose_landmarks:
        return None

    h, w = image_shape[:2]
    lm = results.pose_landmarks.landmark
    joints = {}
    for idx in ARM_CHAIN:
        lm_pt = lm[idx]
        if lm_pt.visibility < 0.5:     # skip low-confidence points
            return None
        joints[idx] = np.array([lm_pt.x * w, lm_pt.y * h], dtype=float)
    return joints


def amplify_joints(joints, neutral, factor):
    """
    For each joint, compute delta from neutral and scale it up.
    Completed position = neutral + delta * factor
    """
    completed = {}
    for idx, pos in joints.items():
        delta = pos - neutral[idx]
        completed[idx] = neutral[idx] + delta * factor
    return completed


def draw_arm(frame, joints, color, thickness=3, radius=8):
    """Draw skeleton chain and joint circles."""
    pts = [joints[idx] for idx in ARM_CHAIN if idx in joints]
    for i in range(len(pts) - 1):
        p1 = tuple(pts[i].astype(int))
        p2 = tuple(pts[i+1].astype(int))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, tuple(pt.astype(int)), radius, color, -1, cv2.LINE_AA)


def draw_ui(frame, amp_factor, phase, countdown=None):
    """Overlay HUD text."""
    h, w = frame.shape[:2]

    # Dark strip at top
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)

    if phase == "calibrating":
        msg = f"CALIBRATING — hold still  {countdown:.1f}s"
        cv2.putText(frame, msg, (15, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        amp_txt = f"Amplification: {amp_factor:.1f}x   [+/-] to adjust   [R] recalibrate   [Q] quit"
        cv2.putText(frame, amp_txt, (15, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, COLOR_TEXT, 1, cv2.LINE_AA)

    # Legend
    cv2.circle(frame, (w - 220, h - 25), 7, COLOR_REAL,      -1)
    cv2.putText(frame, "Your movement",   (w - 205, h - 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_REAL,      1, cv2.LINE_AA)
    cv2.circle(frame, (w - 220, h - 48), 7, COLOR_COMPLETED, -1)
    cv2.putText(frame, "Completed motion", (w - 205, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_COMPLETED, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    mp_pose    = mp.solutions.pose
    pose       = mp_pose.Pose(
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    neutral         = None
    smooth_joints   = None
    amp_factor      = AMPLIFICATION
    calibrating     = True
    calib_start     = time.time()
    calib_samples   = []

    print("Starting calibration — hold your arm still...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels natural
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        joints = get_arm_landmarks(results, frame.shape)

        # ── CALIBRATION PHASE ─────────────────
        if calibrating:
            elapsed   = time.time() - calib_start
            countdown = max(0.0, CALIBRATION_SECS - elapsed)

            if joints is not None:
                calib_samples.append(joints)

            if countdown <= 0 and len(calib_samples) > 0:
                # Average all samples to get a stable neutral
                neutral = {}
                for idx in ARM_CHAIN:
                    neutral[idx] = np.mean(
                        [s[idx] for s in calib_samples if idx in s], axis=0)
                smooth_joints = {k: v.copy() for k, v in neutral.items()}
                calibrating   = False
                calib_samples = []
                print(f"Calibration done. Amplification = {amp_factor}x")
            else:
                draw_ui(frame, amp_factor, "calibrating", countdown)
                if joints:
                    draw_arm(frame, joints, COLOR_NEUTRAL, thickness=2)

            cv2.imshow("Arm Motion Completion", frame)

        # ── LIVE PHASE ────────────────────────
        else:
            if joints is not None:
                # Exponential smoothing to reduce jitter
                for idx in ARM_CHAIN:
                    if idx in smooth_joints and idx in joints:
                        smooth_joints[idx] = (SMOOTHING_ALPHA * joints[idx]
                                              + (1 - SMOOTHING_ALPHA) * smooth_joints[idx])
                    elif idx in joints:
                        smooth_joints[idx] = joints[idx].copy()

                completed = amplify_joints(smooth_joints, neutral, amp_factor)

                # Draw neutral ghost
                draw_arm(frame, neutral,       COLOR_NEUTRAL,   thickness=1, radius=5)
                # Draw real movement
                draw_arm(frame, smooth_joints, COLOR_REAL,      thickness=3, radius=8)
                # Draw completed motion
                draw_arm(frame, completed,     COLOR_COMPLETED, thickness=4, radius=10)

            draw_ui(frame, amp_factor, "live")
            cv2.imshow("Arm Motion Completion", frame)

        # ── KEY HANDLING ──────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            calibrating   = True
            calib_start   = time.time()
            calib_samples = []
            neutral       = None
            print("Recalibrating...")
        elif key == ord('+') or key == ord('='):
            amp_factor = min(amp_factor + 0.5, 10.0)
            print(f"Amplification: {amp_factor}x")
        elif key == ord('-'):
            amp_factor = max(amp_factor - 0.5, 1.0)
            print(f"Amplification: {amp_factor}x")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
