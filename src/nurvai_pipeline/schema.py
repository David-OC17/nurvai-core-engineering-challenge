"""Pydantic models defining the per-frame JSONL document and action-chunk document."""

from typing import Literal

from pydantic import BaseModel


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    w: float
    x: float
    y: float
    z: float


class ImuState(BaseModel):
    gyro: Vec3
    accel: Vec3
    orientation_quat: Quaternion
    energy_z: float | None = None


class HandObservation(BaseModel):
    handedness: Literal["Left", "Right"]
    state: Literal["detected", "held"]
    detection_confidence: float
    frames_since_update: int
    landmarks_px: list[tuple[float, float]]
    landmarks_norm: list[tuple[float, float, float]]
    wrist_velocity: float | None = None


class ActionLabel(BaseModel):
    segment_id: int
    label: Literal["active", "static"]
    activity_score: float | None = None


class FrameDocument(BaseModel):
    frame_index: int
    frame_timestamp_ns: int
    video_time_s: float
    imu: ImuState
    hands: list[HandObservation]
    hands_lost: list[Literal["Left", "Right"]]
    action: ActionLabel


class ActionChunk(BaseModel):
    segment_id: int
    start_frame: int
    end_frame: int
    start_ts_ns: int
    end_ts_ns: int
    label: Literal["active", "static"]
    duration_ms: float
    mean_activity_score: float
    frame_count: int
