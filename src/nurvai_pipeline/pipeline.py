"""Orchestrates Phases 1-3 into the enriched per-frame JSONL + action-chunks JSONL."""

import json
from pathlib import Path

import numpy as np

from nurvai_pipeline.alignment import align_imu_to_frames
from nurvai_pipeline.config import PipelineConfig
from nurvai_pipeline.hand_tracking import apply_gap_holding, detect_raw_per_frame
from nurvai_pipeline.io_loaders import imu_arrays, load_imu, load_vts
from nurvai_pipeline.schema import (
    ActionLabel,
    FrameDocument,
    HandObservation,
    ImuState,
    Quaternion,
    Vec3,
)
from nurvai_pipeline.segmentation import (
    aggregate_hand_velocity,
    build_chunks,
    compute_imu_energy,
    compute_wrist_velocities,
    fuse_activity_score,
    label_segments,
    per_frame_segment_ids,
)


def run_pipeline(
    video_path: str | Path,
    imu_path: str | Path,
    vts_path: str | Path,
    output_dir: str | Path,
    cfg: PipelineConfig | None = None,
) -> tuple[Path, Path]:
    cfg = cfg or PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_df = load_imu(imu_path)
    vts_df = load_vts(vts_path)
    imu = imu_arrays(imu_df)

    frame_index = vts_df["frame_index"].to_numpy()
    frame_t_ns = vts_df["timestamp_ns"].to_numpy(dtype=np.float64)

    print(f"[1/3] Aligning {len(frame_t_ns)} frames against {len(imu['t'])} IMU samples...")
    aligned = align_imu_to_frames(frame_t_ns, imu["t"], imu["gyro"], imu["accel"], cfg.alignment)

    print("[2/3] Running hand tracking (this may take a while on CPU)...")
    raw_hands = detect_raw_per_frame(video_path, frame_t_ns, cfg.hand_tracking)
    frames_hands: list[list[dict]] = []
    hands_lost_per_frame: list[list[str]] = []
    for observations, lost in apply_gap_holding(raw_hands, cfg.hand_tracking.max_hold_frames):
        frames_hands.append(observations)
        hands_lost_per_frame.append(lost)

    n = len(frames_hands)
    if n != len(frame_t_ns):
        # Defensive truncation: video decode may yield fewer frames than vts.csv rows.
        frame_index = frame_index[:n]
        frame_t_ns = frame_t_ns[:n]
        aligned = {k: v[:n] for k, v in aligned.items()}

    compute_wrist_velocities(frames_hands, frame_t_ns)
    hand_velocity = aggregate_hand_velocity(frames_hands)

    print("[3/3] Segmenting action chunks...")
    imu_energy = compute_imu_energy(aligned["gyro"], aligned["accel"])
    activity_score = fuse_activity_score(hand_velocity, imu_energy)
    segments = label_segments(activity_score, cfg.segmentation)
    chunks = build_chunks(segments, frame_t_ns, activity_score)
    seg_ids = per_frame_segment_ids(segments, n)

    enriched_path = output_dir / "enriched.jsonl"
    with open(enriched_path, "w") as f:
        for i in range(n):
            gyro = aligned["gyro"][i]
            accel = aligned["accel"][i]
            quat = aligned["orientation_quat_wxyz"][i]

            hands = [
                HandObservation(
                    handedness=obs["handedness"],
                    state=obs["state"],
                    detection_confidence=obs["detection_confidence"],
                    frames_since_update=obs["frames_since_update"],
                    landmarks_px=obs["landmarks_px"],
                    landmarks_norm=obs["landmarks_norm"],
                    wrist_velocity=obs.get("wrist_velocity"),
                )
                for obs in frames_hands[i]
            ]

            doc = FrameDocument(
                frame_index=int(frame_index[i]),
                frame_timestamp_ns=int(frame_t_ns[i]),
                video_time_s=float(frame_t_ns[i] - frame_t_ns[0]) / 1e9,
                imu=ImuState(
                    gyro=Vec3(x=gyro[0], y=gyro[1], z=gyro[2]),
                    accel=Vec3(x=accel[0], y=accel[1], z=accel[2]),
                    orientation_quat=Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3]),
                    energy_z=float(imu_energy[i]),
                ),
                hands=hands,
                hands_lost=hands_lost_per_frame[i],
                action=ActionLabel(
                    segment_id=int(seg_ids[i]),
                    label="active" if segments[seg_ids[i]][0] else "static",
                    activity_score=float(activity_score[i]) if not np.isnan(activity_score[i]) else None,
                ),
            )
            f.write(doc.model_dump_json() + "\n")

    chunks_path = output_dir / "action_chunks.jsonl"
    with open(chunks_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Wrote {n} frames to {enriched_path}")
    print(f"Wrote {len(chunks)} action chunks to {chunks_path}")
    return enriched_path, chunks_path
