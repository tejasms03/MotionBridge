import numpy as np


class JointKalmanFilter:
    N_STATE = 6
    POS = slice(0, 3)
    VEL = slice(3, 6)

    def __init__(self,
                 q_vel:   float,
                 r_cam:   float,
                 r_cam_z: float,
                 r_imu:   float = 0.5,
                 init_p:  float = 0.01,
                 gate_thresh: float = 9.2):

        self.x = np.zeros(self.N_STATE, dtype=np.float64)
        self.P = np.eye(self.N_STATE, dtype=np.float64) * init_p

        self.q_vel       = float(q_vel)
        self.r_cam       = float(r_cam)
        self.r_imu       = float(r_imu)
        self.gate_thresh = float(gate_thresh)

        self.H_cam = np.zeros((3, self.N_STATE), dtype=np.float64)
        self.H_cam[:, self.POS] = np.eye(3)
        self.R_cam = np.diag([r_cam**2, r_cam**2, r_cam_z**2])

        self.H_imu = np.zeros((3, self.N_STATE), dtype=np.float64)
        self.H_imu[:, self.VEL] = np.eye(3)
        self.R_imu = np.eye(3) * (r_imu ** 2)

        self._initialised = False

    def initialise(self, position: np.ndarray):
        self.x[:] = 0.0
        self.x[self.POS] = position
        # start P near measurement noise so initial K ≈ 0.5
        self.P = np.eye(self.N_STATE) * 1e-4
        self.P[self.POS, self.POS] = self.R_cam.copy()
        self.P[self.VEL, self.VEL] = np.eye(3) * 0.1
        # z depth is noisier — start more uncertain
        self.P[2, 2] = float(self.R_cam[2, 2]) * 5.0
        self._initialised = True

    def predict(self, dt: float):
        if not self._initialised or dt <= 0.0:
            return
        F = self._state_transition(dt)
        Q = self._process_noise(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_camera(self, z: np.ndarray):
        if z is None:
            return
        if not self._initialised:
            self.initialise(z)
            return

        y = z - self.H_cam @ self.x
        S = self.H_cam @ self.P @ self.H_cam.T + self.R_cam

        # Gate on x/y only — a bad z reading shouldn't discard a good x/y update
        try:
            S_xy_inv = np.linalg.inv(S[:2, :2])
            d2_xy = float(y[:2].T @ S_xy_inv @ y[:2])
        except np.linalg.LinAlgError:
            d2_xy = 0.0

        if d2_xy > self.gate_thresh:
            return  # outlier — coast on prediction

        self._kalman_update(z, self.H_cam, self.R_cam)

    def update_imu(self, vel_world=None):
        if vel_world is None or not self._initialised:
            return
        self._kalman_update(vel_world, self.H_imu, self.R_imu)

    @property
    def position(self) -> np.ndarray:
        return self.x[self.POS].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.VEL].copy()

    @property
    def initialised(self) -> bool:
        return self._initialised

    def _state_transition(self, dt: float) -> np.ndarray:
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        return np.block([[I3, I3 * dt],
                         [Z3, I3]])

    def _process_noise(self, dt: float) -> np.ndarray:
        s   = self.q_vel ** 2
        dt2 = dt * dt
        I3  = np.eye(3)
        return np.block([[I3 * dt2 * s * 0.25, I3 * dt * s * 0.5],
                         [I3 * dt  * s * 0.5,  I3 * s]])

    def _kalman_update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.N_STATE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form for symmetry


class MultiJointEstimator:
    def __init__(self, cfg: dict):
        self.q_vel     = float(cfg.get("q_vel",     0.3))
        self.r_cam     = float(cfg.get("r_cam",     0.012))
        self.r_cam_z   = float(cfg.get("r_cam_z",   0.20))
        self.r_imu     = float(cfg.get("r_imu",     0.5))
        self.init_p    = float(cfg.get("init_p",    0.01))
        self.gate      = float(cfg.get("gate_thresh", 9.2))
        self._filters: dict[str, JointKalmanFilter] = {}

    def reset(self):
        self._filters.clear()

    def seed_from_neutral(self, neutral: dict):
        for name, pos in neutral.items():
            kf = self._get_filter(name)
            kf.initialise(np.asarray(pos, dtype=np.float64))

    def velocity(self, name: str) -> np.ndarray:
        if name not in self._filters:
            return np.zeros(3, dtype=np.float32)
        return self._filters[name].velocity.astype(np.float32)

    def step(self,
             joints: dict | None,
             dt: float,
             imu:  dict | None = None,
             emg:  dict | None = None) -> dict:

        known = set(self._filters.keys())
        if joints:
            known.update(joints.keys())

        out: dict[str, np.ndarray] = {}
        for name in known:
            kf = self._get_filter(name)
            kf.predict(dt)

            if joints and name in joints:
                kf.update_camera(joints[name].astype(np.float64))

            if imu and name in imu:
                kf.update_imu(imu[name].astype(np.float64))

            if kf.initialised:
                out[name] = kf.position.astype(np.float32)

        return out

    def _get_filter(self, name: str) -> JointKalmanFilter:
        if name not in self._filters:
            self._filters[name] = JointKalmanFilter(
                q_vel=self.q_vel,
                r_cam=self.r_cam,
                r_cam_z=self.r_cam_z,
                r_imu=self.r_imu,
                init_p=self.init_p,
                gate_thresh=self.gate,
            )
        return self._filters[name]
