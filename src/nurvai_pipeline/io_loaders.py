"""Raw CSV loaders for IMU and video-timestamp (vts) data."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_imu(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"timestamp_ns", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"imu.csv missing expected columns: {missing}")
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    return df


def load_vts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"frame_index", "timestamp_ns"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"vts.csv missing expected columns: {missing}")
    df = df.sort_values("frame_index").reset_index(drop=True)
    return df


def imu_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "t": df["timestamp_ns"].to_numpy(dtype=np.float64),
        "gyro": df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float64),
        "accel": df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=np.float64),
    }
