"""Phase 1: temporal alignment of video frames with IMU telemetry.

Strategy (see plan for justification):
  - accel/gyro: linear interpolation per-axis at each frame's exact timestamp.
    Source IMU (~574Hz) is ~19x oversampled relative to video (30Hz), so linear
    interpolation is accurate and avoids spline overshoot on noisy accel data.
  - orientation: a single quaternion track is integrated once over the full
    native-rate gyro stream (with an optional accel-based complementary
    correction to bound long-horizon drift), then resampled onto frame
    timestamps via spherical interpolation (Slerp). Integrating once at native
    rate and interpolating the *result* avoids compounding resampling error
    that re-integrating from already-resampled gyro would introduce.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from nurvai_pipeline.config import AlignmentConfig


def interp_vec3(t_query: np.ndarray, t_src: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    """Linearly interpolate a (N,3) signal sampled at t_src onto t_query.

    Raises ValueError if any query timestamp falls outside the source range —
    this dataset's video timeline is fully bracketed by the IMU timeline, so
    extrapolation should never be needed; a silent clamp/extrapolate would
    hide a violated assumption if this ever runs on different data.
    """
    if t_query.min() < t_src.min() or t_query.max() > t_src.max():
        raise ValueError(
            "Query timestamps fall outside the IMU sample range "
            f"(query=[{t_query.min()}, {t_query.max()}], "
            f"src=[{t_src.min()}, {t_src.max()}]); refusing to extrapolate."
        )
    out = np.empty((len(t_query), 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(t_query, t_src, v_src[:, axis])
    return out


def integrate_orientation(
    t_ns: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    cfg: AlignmentConfig | None = None,
) -> Rotation:
    """Integrate gyro readings into a native-rate orientation track.

    First-order quaternion update using each sample's own delta-t (IMU
    sampling has real jitter, not a perfectly fixed period). Optionally
    applies a lightweight complementary-filter correction that nudges
    roll/pitch toward the accelerometer-derived gravity direction, bounding
    the drift a pure gyro integration would otherwise accumulate over a
    ~5 minute session. Yaw is left uncorrected (accel can't observe it); a
    full Madgwick/Mahony filter is deliberately out of scope for this budget.
    """
    cfg = cfg or AlignmentConfig()
    n = len(t_ns)
    quats = np.empty((n, 4))  # scipy convention: [x, y, z, w]
    q = Rotation.identity()
    quats[0] = q.as_quat()

    dt = np.diff(t_ns) * 1e-9  # seconds
    world_up = np.array([0.0, 0.0, 1.0])

    for i in range(1, n):
        rotvec = gyro[i - 1] * dt[i - 1]
        q = q * Rotation.from_rotvec(rotvec)

        if cfg.use_accel_correction:
            a = accel[i]
            a_norm = np.linalg.norm(a)
            if a_norm > 1e-6:
                a_unit = a / a_norm
                g_body_pred = q.inv().apply(world_up)
                axis = np.cross(g_body_pred, a_unit)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-9:
                    angle = np.arcsin(np.clip(axis_norm, -1.0, 1.0))
                    correction = (axis / axis_norm) * angle * cfg.complementary_filter_weight
                    q = q * Rotation.from_rotvec(correction)

        quats[i] = q.as_quat()

    return Rotation.from_quat(quats)


def resample_orientation(
    t_query: np.ndarray, t_src: np.ndarray, orientations: Rotation
) -> Rotation:
    """Spherically interpolate a native-rate orientation track onto query timestamps.

    Raises ValueError (via scipy's Slerp) if any query timestamp falls outside
    the source range — the same no-extrapolation guarantee as interp_vec3.
    """
    slerp = Slerp(t_src, orientations)
    return slerp(t_query)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
    return np.stack([w, x, y, z], axis=-1)


def align_imu_to_frames(
    frame_t_ns: np.ndarray,
    imu_t_ns: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    cfg: AlignmentConfig | None = None,
) -> dict[str, np.ndarray]:
    """Produce per-frame interpolated gyro/accel and orientation quaternion (w,x,y,z)."""
    cfg = cfg or AlignmentConfig()

    gyro_per_frame = interp_vec3(frame_t_ns, imu_t_ns, gyro)
    accel_per_frame = interp_vec3(frame_t_ns, imu_t_ns, accel)

    native_orientations = integrate_orientation(imu_t_ns, gyro, accel, cfg)
    frame_orientations = resample_orientation(frame_t_ns, imu_t_ns, native_orientations)
    quat_wxyz = quat_xyzw_to_wxyz(frame_orientations.as_quat())

    return {
        "gyro": gyro_per_frame,
        "accel": accel_per_frame,
        "orientation_quat_wxyz": quat_wxyz,
    }
