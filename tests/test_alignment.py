import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nurvai_pipeline.alignment import (
    align_imu_to_frames,
    integrate_orientation,
    interp_vec3,
    resample_orientation,
)
from nurvai_pipeline.config import AlignmentConfig


def test_interp_vec3_matches_hand_computed_value():
    t_src = np.array([0.0, 1e9, 2e9])  # ns
    v_src = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [20.0, 40.0, 60.0]])
    t_query = np.array([0.5e9])
    result = interp_vec3(t_query, t_src, v_src)
    np.testing.assert_allclose(result, [[5.0, 10.0, 15.0]])


def test_interp_vec3_raises_on_out_of_range_query():
    t_src = np.array([0.0, 1e9])
    v_src = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        interp_vec3(np.array([2e9]), t_src, v_src)


def test_integrate_orientation_matches_analytical_constant_angular_velocity():
    # Constant angular velocity of pi/2 rad/s about Z for 2 seconds -> net 90 degree rotation.
    hz = 100
    duration_s = 2.0
    n = int(duration_s * hz) + 1
    t_ns = np.linspace(0, duration_s * 1e9, n)
    omega_z = np.pi / 2  # rad/s
    gyro = np.tile([0.0, 0.0, omega_z], (n, 1))
    # Gravity-aligned, unchanging accel (no correction needed/expected to fire much).
    accel = np.tile([0.0, 0.0, 9.81], (n, 1))

    cfg = AlignmentConfig(use_accel_correction=False)
    orientations = integrate_orientation(t_ns, gyro, accel, cfg)

    expected_final = Rotation.from_rotvec([0.0, 0.0, omega_z * duration_s])
    actual_final = orientations[-1]

    # Compare via the angle of the relative rotation between expected and actual.
    relative = expected_final.inv() * actual_final
    angle_error_deg = np.degrees(np.linalg.norm(relative.as_rotvec()))
    assert angle_error_deg < 1.0


def test_resample_orientation_raises_on_out_of_range_query():
    t_src = np.array([0.0, 1e9])
    orientations = Rotation.from_quat([[0, 0, 0, 1], [0, 0, 0, 1]])
    with pytest.raises(ValueError):
        resample_orientation(np.array([2e9]), t_src, orientations)


def test_align_imu_to_frames_video_range_within_imu_range_on_real_dataset_shapes():
    # Mimics the real dataset's invariant: IMU range fully brackets frame range.
    imu_t = np.linspace(0, 10e9, 1000)
    frame_t = np.linspace(0.1e9, 9.9e9, 30)
    gyro = np.zeros((1000, 3))
    accel = np.tile([0.0, 0.0, 9.81], (1000, 1))

    result = align_imu_to_frames(frame_t, imu_t, gyro, accel)

    assert result["gyro"].shape == (30, 3)
    assert result["accel"].shape == (30, 3)
    assert result["orientation_quat_wxyz"].shape == (30, 4)
