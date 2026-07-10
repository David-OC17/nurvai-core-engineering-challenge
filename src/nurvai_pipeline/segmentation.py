"""Phase 3: action segmentation via a fused hand-motion + IMU-energy activity score.

Both signals are normalized session-relatively (robust z-score via median/MAD)
rather than against fixed constants, since absolute sensor/pixel units are
rig- and session-specific. They're fused with max() (an OR over independently
z-scored signals) so a strong signal in either channel alone is enough to
call a frame "active" — a weighted average would let a quiet second channel
dilute a genuinely active one. A minimum-duration hysteresis pass then
removes single-frame label flicker.
"""

from dataclasses import dataclass

import numpy as np

from nurvai_pipeline.config import SegmentationConfig


def compute_wrist_velocities(frames: list[list[dict]], frame_t_ns: np.ndarray) -> None:
    """Mutates each hand observation dict in-place, adding a `wrist_velocity` field.

    Velocity is only computed between consecutive *detected* frames (using
    landmark 0, the wrist) so that "held" frames — whose position is frozen
    stale data, not a real observation — don't fabricate a spurious zero
    velocity that would bias the frame toward "static".
    """
    last_detected: dict[str, tuple[np.ndarray, float]] = {}

    for i, observations in enumerate(frames):
        for obs in observations:
            handedness = obs["handedness"]
            if obs["state"] != "detected":
                obs["wrist_velocity"] = None
                continue

            pos = np.array(obs["landmarks_norm"][0][:2])
            if handedness in last_detected:
                prev_pos, prev_t_ns = last_detected[handedness]
                dt = (frame_t_ns[i] - prev_t_ns) * 1e-9
                obs["wrist_velocity"] = float(np.linalg.norm(pos - prev_pos) / dt) if dt > 0 else None
            else:
                obs["wrist_velocity"] = None
            last_detected[handedness] = (pos, frame_t_ns[i])


def aggregate_hand_velocity(frames: list[list[dict]]) -> np.ndarray:
    """Per-frame mean wrist velocity across currently-tracked hands; NaN if none available."""
    out = np.full(len(frames), np.nan)
    for i, observations in enumerate(frames):
        vals = [o["wrist_velocity"] for o in observations if o.get("wrist_velocity") is not None]
        if vals:
            out[i] = float(np.mean(vals))
    return out


def compute_imu_energy(gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """Per-frame motion energy: gyro magnitude + accel magnitude with the
    session's static gravity bias (approximated by the median accel vector)
    removed, so the signal reflects motion rather than a constant offset."""
    accel_dynamic = accel - np.median(accel, axis=0, keepdims=True)
    return np.linalg.norm(gyro, axis=1) + np.linalg.norm(accel_dynamic, axis=1)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median)) * 1.4826  # consistent estimator for normal data
    if mad < 1e-9:
        return np.zeros_like(x)
    return (x - median) / mad


def fuse_activity_score(hand_velocity: np.ndarray, imu_energy: np.ndarray) -> np.ndarray:
    z_hand = robust_zscore(hand_velocity)
    z_imu = robust_zscore(imu_energy)
    stacked = np.vstack([z_hand, z_imu])
    return np.nanmax(stacked, axis=0)


def _merge_short_segments(segments: list[list], min_segment_frames: int) -> list[list]:
    segments = [list(s) for s in segments]
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for idx, (_label, start, end) in enumerate(segments):
            if end - start < min_segment_frames:
                if idx > 0:
                    segments[idx - 1][2] = end
                else:
                    segments[idx + 1][1] = start
                del segments[idx]
                changed = True
                break

    merged = [segments[0]]
    for label, start, end in segments[1:]:
        if label == merged[-1][0]:
            merged[-1][2] = end
        else:
            merged.append([label, start, end])
    return merged


def label_segments(activity_score: np.ndarray, cfg: SegmentationConfig) -> list[list]:
    """Threshold + hysteresis: returns a list of [label, start_frame, end_frame_exclusive]."""
    candidate = activity_score > cfg.activity_z_threshold

    segments = []
    start = 0
    for i in range(1, len(candidate) + 1):
        if i == len(candidate) or candidate[i] != candidate[start]:
            segments.append([bool(candidate[start]), start, i])
            start = i

    return _merge_short_segments(segments, cfg.min_segment_frames)


@dataclass
class ActionChunkResult:
    segment_id: int
    start_frame: int
    end_frame: int
    label: str


def build_chunks(
    segments: list[list],
    frame_t_ns: np.ndarray,
    activity_score: np.ndarray,
) -> list[dict]:
    chunks = []
    for segment_id, (is_active, start, end) in enumerate(segments):
        end_inclusive = end - 1
        chunks.append(
            {
                "segment_id": segment_id,
                "start_frame": start,
                "end_frame": end_inclusive,
                "start_ts_ns": int(frame_t_ns[start]),
                "end_ts_ns": int(frame_t_ns[end_inclusive]),
                "label": "active" if is_active else "static",
                "duration_ms": float((frame_t_ns[end_inclusive] - frame_t_ns[start]) * 1e-6),
                "mean_activity_score": float(np.nanmean(activity_score[start:end])),
                "frame_count": end - start,
            }
        )
    return chunks


def per_frame_segment_ids(segments: list[list], n_frames: int) -> list[int]:
    ids = np.empty(n_frames, dtype=int)
    for segment_id, (_label, start, end) in enumerate(segments):
        ids[start:end] = segment_id
    return ids.tolist()
