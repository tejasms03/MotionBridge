from __future__ import annotations

import numpy as np

from estimator import MultiJointEstimator

_CLAMP_XY      = (0.02, 0.98)
_WARMUP_FRAMES = 10
_Z_AMP_SCALE   = 0.0

_LEAD_S        = 0.10
_LEAD_DEAD     = 0.03
_LEAD_KNEE     = 0.20

_JOINT_AMP_SCALE = {
    "right_elbow": 0.85,
    "right_wrist": 1.00,
}

_MAX_XY_DELTA = 0.30
_MAX_Z_DELTA  = 0.60


class MotionCompleter:
    def __init__(self, cfg: dict, estimator_cfg: dict | None = None):
        self.amp_factor   = float(cfg.get("amplification_factor", 3.0))
        self._estimator   = MultiJointEstimator(estimator_cfg or {})
        self._frame_count = 0
        self._last_good:  dict = {}

        self._use_lstm = bool(cfg.get("use_lstm", False))
        self._lstm: "LSTMPredictor | None" = None

        if self._use_lstm:
            self._lstm_cfg = cfg
            print("[MotionCompleter] LSTM mode ENABLED — "
                  "model will be loaded on first frame.")

    def set_amplification(self, factor: float):
        self.amp_factor = max(1.0, round(factor, 1))

    def complete(self,
                 joints:     dict | None,
                 calib_data: dict,
                 dt:         float,
                 imu:        dict | None = None,
                 emg:        dict | None = None) -> dict:

        estimated = self._estimator.step(joints, dt, imu=imu, emg=emg)
        if not estimated:
            return self._last_good

        self._frame_count += 1
        if self._frame_count < _WARMUP_FRAMES:
            return {}

        neutral = calib_data.get("neutral", {})

        if "right_wrist" in estimated and "right_wrist" in neutral:
            n     = np.asarray(neutral["right_wrist"], dtype=np.float32)
            delta = estimated["right_wrist"] - n
            if (abs(delta[0]) > _MAX_XY_DELTA or
                abs(delta[1]) > _MAX_XY_DELTA or
                abs(delta[2]) > _MAX_Z_DELTA):
                return self._last_good

        lstm_estimated: dict | None = None
        if self._use_lstm:
            lstm_estimated = self._lstm_predict(estimated)

        working = lstm_estimated if lstm_estimated is not None else estimated

        completed: dict[str, np.ndarray] = {}

        for key in ("right_shoulder", "left_shoulder"):
            if key in working:
                completed[key] = working[key].copy()
        if "left_shoulder" not in completed and "right_shoulder" in working:
            completed["left_shoulder"] = working["right_shoulder"].copy()

        # velocity always taken from the Kalman filter (LSTM has none)
        for joint in ("right_elbow", "right_wrist"):
            if joint not in working:
                continue
            if joint not in neutral:
                completed[joint] = working[joint].copy()
                continue

            vel      = self._estimator.velocity(joint)
            speed_xy = float(np.linalg.norm(vel[:2]))

            if speed_xy <= _LEAD_DEAD:
                lead = 0.0
            else:
                lead = _LEAD_S * min((speed_xy - _LEAD_DEAD) /
                                     (_LEAD_KNEE - _LEAD_DEAD), 1.0)

            predicted   = working[joint] + vel * lead
            neutral_pos = np.asarray(neutral[joint], dtype=np.float32)
            delta       = predicted - neutral_pos
            scale       = _JOINT_AMP_SCALE.get(joint, 1.0)
            eff_amp     = 1.0 + (self.amp_factor - 1.0) * scale

            amp = np.array([
                1.0 + (eff_amp - 1.0) * 0.4,
                eff_amp,
                1.0 + (eff_amp - 1.0) * _Z_AMP_SCALE,
            ], dtype=np.float32)

            pos    = neutral_pos + delta * amp
            pos[0] = np.clip(pos[0],  _CLAMP_XY[0], _CLAMP_XY[1])
            pos[1] = np.clip(pos[1],  _CLAMP_XY[0], _CLAMP_XY[1])
            pos[2] = np.clip(pos[2], -0.5, 0.5)
            completed[joint] = pos

        if ("right_wrist" in completed and
            "right_index" in working and
            "right_wrist" in working):
            offset = working["right_index"] - working["right_wrist"]
            pos    = completed["right_wrist"] + offset
            pos[0] = np.clip(pos[0],  _CLAMP_XY[0], _CLAMP_XY[1])
            pos[1] = np.clip(pos[1],  _CLAMP_XY[0], _CLAMP_XY[1])
            pos[2] = np.clip(pos[2], -0.5, 0.5)
            completed["right_index"] = pos

        if (self._frame_count % 30 == 0 and
            "right_wrist" in completed and "right_wrist" in neutral):
            n     = np.asarray(neutral["right_wrist"], dtype=np.float32)
            delta = estimated["right_wrist"] - n
            vel   = self._estimator.velocity("right_wrist")
            spd   = float(np.linalg.norm(vel[:2]))
            lead  = 0.0 if spd <= _LEAD_DEAD else \
                    _LEAD_S * min((spd - _LEAD_DEAD) /
                                  (_LEAD_KNEE - _LEAD_DEAD), 1.0)
            lstm_tag = " [LSTM]" if lstm_estimated is not None else ""
            print(f"[debug{lstm_tag}] wrist delta=[ {delta[0]:+.3f}  {delta[1]:+.3f}  "
                  f"{delta[2]:+.3f} ]  vel=[ {vel[0]:+.2f}  {vel[1]:+.2f} ]  "
                  f"speed={spd:.3f}  lead={lead:.3f}s  amp={self.amp_factor}x  "
                  f"completed_y={completed['right_wrist'][1]:.3f}")

        self._last_good = completed
        return completed

    def reset_smoothing(self):
        self._estimator.reset()
        self._frame_count = 0
        self._last_good   = {}
        if self._lstm is not None:
            self._lstm.reset()

    def seed_from_calibration(self, calib_data: dict):
        neutral = calib_data.get("neutral", {})
        if neutral:
            self._estimator.seed_from_neutral(neutral)
            self._frame_count = _WARMUP_FRAMES

    def _lazy_load_lstm(self):
        if self._lstm is not None:
            return

        try:
            from lstm_predictor import LSTMPredictor
        except ImportError as e:
            print(f"[MotionCompleter] Cannot import lstm_predictor: {e}")
            self._use_lstm = False
            return

        lstm_cfg   = self._lstm_cfg
        model_path = lstm_cfg.get("lstm_model_path",
                     lstm_cfg.get("model_path", "lstm_model.onnx"))
        seq_len    = int(lstm_cfg.get("sequence_length", 30))
        blend      = float(lstm_cfg.get("lstm_blend", 0.3))

        try:
            self._lstm = LSTMPredictor(model_path, seq_len=seq_len, blend=blend)
            print(f"[MotionCompleter] LSTM loaded from {model_path!r}  "
                  f"seq_len={seq_len}  blend={blend}")
        except FileNotFoundError as e:
            print(f"[MotionCompleter] LSTM disabled: {e}")
            self._use_lstm = False

    def _lstm_predict(self, estimated: dict) -> dict | None:
        self._lazy_load_lstm()
        if not self._use_lstm or self._lstm is None:
            return None
        return self._lstm.update(estimated)
