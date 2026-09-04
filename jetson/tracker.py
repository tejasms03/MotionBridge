import time

import cv2
import mediapipe as mp
import numpy as np

JOINTS = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "right_elbow":    14,
    "right_wrist":    16,
    "right_index":    20,
}

ARM_CONNECTIONS = [
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("right_wrist",    "right_index"),
]

# Fractions of frame height — keeps proportions consistent at any resolution
_RADII = {
    "right_shoulder": 0.026,
    "right_elbow":    0.019,
    "right_wrist":    0.013,
    "right_index":    0.008,
}


def _capsule(img, pt1, pt2, r1, r2, color):
    p1 = np.array(pt1, np.float32)
    p2 = np.array(pt2, np.float32)
    v = p2 - p1
    d = float(np.linalg.norm(v))
    if d < 1:
        return
    perp = np.array([-v[1], v[0]]) / d

    poly = np.array(
        [p1 + perp * r1, p2 + perp * r2, p2 - perp * r2, p1 - perp * r1],
        np.int32,
    )
    cv2.fillPoly(img, [poly], color)
    cv2.circle(img, (int(p1[0]), int(p1[1])), int(r1), color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(p2[0]), int(p2[1])), int(r2), color, -1, cv2.LINE_AA)

    # Brighten toward white ~40% for specular highlight
    hi = tuple(min(255, int(c + (255 - c) * 0.4)) for c in color)
    cv2.line(
        img,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        hi,
        max(1, int((r1 + r2) * 0.22)),
        cv2.LINE_AA,
    )


class PoseTracker:
    def __init__(self, cfg):
        delegate_str = cfg.get("delegate", "cpu").upper()
        if delegate_str == "GPU":
            delegate = mp.tasks.BaseOptions.Delegate.GPU
            print("[Tracker] Using GPU delegate for MediaPipe.")
        else:
            delegate = mp.tasks.BaseOptions.Delegate.CPU
            print("[Tracker] Using CPU delegate for MediaPipe.")

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=cfg["model_path"],
                delegate=delegate,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=cfg["min_detection_confidence"],
            min_pose_presence_confidence=cfg["min_detection_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]
        return {
            name: np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=np.float32)
            for name, idx in JOINTS.items()
        }

    def draw_arm_solid(self, frame, joints, color, outline=(20, 20, 20)):
        h, w = frame.shape[:2]

        def px(name):
            return (int(joints[name][0] * w), int(joints[name][1] * h))

        def r(name):
            return _RADII.get(name, 0.01) * h

        if outline:
            for a, b in ARM_CONNECTIONS:
                if a not in joints or b not in joints:
                    continue
                _capsule(frame, px(a), px(b), r(a) + 3, r(b) + 3, outline)

        for a, b in ARM_CONNECTIONS:
            if a not in joints or b not in joints:
                continue
            _capsule(frame, px(a), px(b), r(a), r(b), color)

    def draw_arm(self, frame, joints, color, thickness=2):
        h, w = frame.shape[:2]
        for a, b in ARM_CONNECTIONS:
            if a not in joints or b not in joints:
                continue
            pt1 = (int(joints[a][0] * w), int(joints[a][1] * h))
            pt2 = (int(joints[b][0] * w), int(joints[b][1] * h))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        for name in ("right_shoulder", "right_elbow", "right_wrist"):
            if name in joints:
                pt = (int(joints[name][0] * w), int(joints[name][1] * h))
                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)

    def close(self):
        self._landmarker.close()
