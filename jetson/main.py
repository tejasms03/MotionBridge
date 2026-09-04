import argparse
import importlib
import time

import cv2
import numpy as np
import pygame as _pygame
import yaml

from arduino import ArduinoReader
from calibration import CalibrationManager
from completion import MotionCompleter
from menu import run_menu
from streamer import UDPStreamer
from tracker import PoseTracker


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def draw_hud(frame, completer, calib_ready: bool, fps: float,
             steadiness: float, burst_ready: bool, burst_remaining: float):
    h, w = frame.shape[:2]

    if burst_remaining == 0.0 and not burst_ready:
        emg_txt = "EMG: N/A"
    elif burst_ready:
        emg_txt = "EMG: READY"
    else:
        emg_txt = f"EMG: {burst_remaining:.1f}s"
    lines = [
        f"FPS: {fps:.0f}",
        f"Amp: {completer.amp_factor:.1f}x",
        f"Calib: {'ready' if calib_ready else 'NOT SET -- press R'}",
        emg_txt,
        "+/-  Amp    R  Recalibrate    Q  Quit",
    ]
    for i, text in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(frame, text, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        color = (0, 255, 80) if (i == 3 and burst_ready) else (255, 255, 255)
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 10, h - 30, 140, 14
    max_shake = 2.0
    stability = max(0.0, 1.0 - steadiness / max_shake)
    filled    = int(bar_w * stability)
    r = int(255 * (1 - stability))
    g = int(255 * stability)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      (r, g, 0), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (180, 180, 180), 1)
    cv2.putText(frame, "Steadiness", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


_IMU_DECAY = 0.75  # velocity decay per frame — prevents integration drift

def _acc_to_world(acc: dict, gravity: np.ndarray,
                  vel: np.ndarray, dt: float, scale: float) -> np.ndarray:
    raw   = np.array([acc["ax"], acc["ay"], acc["az"]], np.float32)
    lin   = raw - gravity
    world = np.array([lin[0], -lin[2], lin[1]], np.float32)
    return vel * _IMU_DECAY + world * dt / scale


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    tracker        = PoseTracker(cfg["mediapipe"])
    calib          = CalibrationManager(cfg["calibration"]["hold_duration"],
                                        cfg["calibration"]["save_path"])
    completion_cfg = {**cfg.get("lstm", {}), **cfg["completion"]}
    completer      = MotionCompleter(completion_cfg, cfg.get("estimator", {}))
    streamer       = UDPStreamer(cfg["udp"])
    arduino        = ArduinoReader(cfg.get("arduino", {}))
    arm_scale      = float(cfg["game"].get("arm_scale", 3.2))

    if calib.load():
        print("Loaded existing calibration.")
        completer.seed_from_calibration(calib.data)
    else:
        print("No calibration found — press R to begin.")

    gst = cfg["camera"].get("gstreamer_pipeline")
    if gst:
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        print(f"[Camera] Using GStreamer pipeline.")
    else:
        cap = cv2.VideoCapture(cfg["camera"]["index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
        cap.set(cv2.CAP_PROP_FPS,          cfg["camera"]["fps"])

    cv2.namedWindow("Arm Motion Completion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Arm Motion Completion", 480, 270)
    cv2.moveWindow("Arm Motion Completion",  0, 0)

    wrist_grav:  np.ndarray | None = None
    elbow_grav:  np.ndarray | None = None
    wrist_vel    = np.zeros(3, np.float32)
    elbow_vel    = np.zeros(3, np.float32)
    grav_buf_w: list = []
    grav_buf_e: list = []

    running = True
    while running:
        selection = run_menu(cfg)
        if selection is None:
            break

        mod       = importlib.import_module(selection["module"])
        GameClass = getattr(mod, selection["class"])
        game      = GameClass(cfg["game"])
        completer.reset_smoothing()

        prev_t       = time.perf_counter()
        game_running = True

        while game_running:
            ret, frame = cap.read()
            if not ret:
                break

            now    = time.perf_counter()
            dt     = max(now - prev_t, 1e-6)
            prev_t = now

            frame  = cv2.flip(frame, 1)
            joints = tracker.process(frame)

            if calib.active:
                w = arduino.imu_wrist()
                e = arduino.imu_elbow()
                if w:
                    grav_buf_w.append([w["ax"], w["ay"], w["az"]])
                if e:
                    grav_buf_e.append([e["ax"], e["ay"], e["az"]])

                if calib.update(joints):
                    print("Calibration complete — saved to",
                          cfg["calibration"]["save_path"])
                    if grav_buf_w:
                        wrist_grav = np.mean(grav_buf_w, axis=0, dtype=np.float32)
                        grav_buf_w.clear()
                    if grav_buf_e:
                        elbow_grav = np.mean(grav_buf_e, axis=0, dtype=np.float32)
                        grav_buf_e.clear()
                    wrist_vel[:] = 0.0
                    elbow_vel[:] = 0.0
                    completer.seed_from_calibration(calib.data)
                    completer.reset_smoothing()

            calib.draw_overlay(frame)

            if joints:
                tracker.draw_arm_solid(frame, joints, color=(30, 180, 30))

            imu_data: dict | None = None
            w = arduino.imu_wrist()
            e = arduino.imu_elbow()
            if wrist_grav is not None and w:
                wrist_vel = _acc_to_world(w, wrist_grav, wrist_vel, dt, arm_scale)
                imu_data  = imu_data or {}
                imu_data["right_wrist"] = wrist_vel.copy()
            if elbow_grav is not None and e:
                elbow_vel = _acc_to_world(e, elbow_grav, elbow_vel, dt, arm_scale)
                imu_data  = imu_data or {}
                imu_data["right_elbow"] = elbow_vel.copy()

            world_joints = {}
            if joints and calib.is_ready() and not calib.active:
                # IMU fusion disabled — axis mapping needs physical calibration first
                completed = completer.complete(joints, calib.data, dt, imu=None)
                if completed:
                    tracker.draw_arm_solid(frame, completed, color=(210, 90, 20))
                    streamer.send(completed)
                    world_joints = game.mp_to_world(completed)

            game.update(world_joints)
            if arduino.check_burst():
                game.emg_burst()
            b_ready, b_rem = arduino.burst_ready()
            game.render(world_joints, arduino.steadiness(), b_ready, b_rem)
            game_running, pg_keys = game.pump_events()

            fps = 1.0 / dt
            draw_hud(frame, completer, calib.is_ready(),
                     fps, arduino.steadiness(), b_ready, b_rem)
            cv2.imshow("Arm Motion Completion", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or _pygame.K_q in pg_keys:
                game_running = False
                running      = False
            elif key == ord("r") or _pygame.K_r in pg_keys:
                calib.start()
                grav_buf_w.clear()
                grav_buf_e.clear()
                completer.reset_smoothing()
            elif key in (ord("+"), ord("=")) or _pygame.K_PLUS in pg_keys or _pygame.K_EQUALS in pg_keys:
                completer.set_amplification(completer.amp_factor + 0.5)
            elif key == ord("-") or _pygame.K_MINUS in pg_keys:
                completer.set_amplification(completer.amp_factor - 0.5)

        game.close()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    streamer.close()
    arduino.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
