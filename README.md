# Nurvai Core Engineering Challenge — Embodied Data Alchemist

Post-processing pipeline that aligns egocentric video with IMU telemetry,
extracts hand keypoints, and segments the session into "active"/"static"
action chunks, producing an enriched per-frame `.jsonl` dataset and a QA
verification video.

## Host setup (one-time, required for GPU passthrough)

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # sanity check
```

## Build & run

```bash
docker build -t nurvai-pipeline .

docker run --rm --gpus all \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/out:/out" \
  nurvai-pipeline run --video /data/video.mp4 --imu /data/imu.csv --vts /data/vts.csv --output-dir /out

docker run --rm --gpus all \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/out:/out" \
  nurvai-pipeline qa --video /data/video.mp4 --jsonl /out/enriched.jsonl --output /out/output_qa.mp4
```

Outputs land in `out/`: `enriched.jsonl`, `action_chunks.jsonl`, `output_qa.mp4`.

## Local development (no Docker)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
nurvai-pipeline run --video data/video.mp4 --imu data/imu.csv --vts data/vts.csv --output-dir out
nurvai-pipeline qa --video data/video.mp4 --jsonl out/enriched.jsonl --output out/output_qa.mp4
```

## Design notes & trade-offs

**Phase 1 — alignment.** `vts.csv` already maps `frame_index → timestamp_ns`
in the same clock domain as `imu.csv` (verified: the IMU range fully brackets
the video timeline, no extrapolation ever needed for this dataset — enforced
defensively with a `ValueError` guard rather than silently assumed). Accel/gyro
are linearly interpolated per axis (IMU is ~19x oversampled relative to video,
so linear interpolation is accurate and safer than cubic spline, which can
overshoot on noisy accel with no smoothness guarantee to justify it).
Orientation is integrated once, at native IMU rate, into a continuous
quaternion track (first-order update using each sample's own delta-t, since
real sampling has jitter), then resampled onto frame timestamps via Slerp —
integrating once and interpolating the result avoids compounding error versus
re-integrating from already-resampled gyro. A lightweight complementary filter
nudges roll/pitch toward the accelerometer-derived gravity direction to bound
long-horizon gyro drift over the ~5 minute session (config: `use_accel_correction`).
A full Madgwick/Mahony filter was considered but is out of scope for this
budget — noted here rather than silently skipped.

**Phase 2 — hand tracking.** MediaPipe Hand Landmarker (Tasks API),
`running_mode=VIDEO` (uses timestamp continuity for tracking, unlike per-frame
`IMAGE` mode). Robustness to occlusion/lighting is handled with an explicit
per-hand-slot state machine: `detected` (fresh), `held` (missing but within
`max_hold_frames`, ~333ms — holds last-known position rather than fabricating
motion), `lost` (missing beyond that window — landmarks omitted, not invented,
flagged via `hands_lost`). MediaPipe's Python GPU delegate on Linux is
inconsistent for mainline pip wheels; since this is an offline batch job with
no real-time constraint, hand tracking runs on CPU delegate (fast enough at
30fps offline) — the CUDA base image/GPU passthrough is still provided per
the deliverable spec and leaves headroom for other work, but effort wasn't
spent fighting MediaPipe's GPU delegate specifically.

**Phase 3 — segmentation.** Fuses hand-keypoint motion (wrist velocity,
computed only between genuine `detected` frames so `held` frames don't read
as spurious zero-velocity) with IMU motion energy. Both signals are
normalized session-relatively via a robust z-score (median/MAD) rather than
fixed constants, since absolute sensor/pixel units are rig- and
session-specific. Fusion is `max(z_hand, z_imu)` (an OR over independently
z-scored signals) rather than a weighted average, so a strong signal in
either channel is enough to call a frame active — a weighted sum would let a
quiet channel dilute a genuinely active one. If both hands are lost, the score
falls back to IMU-only rather than treating the frame as motionless. A
threshold + minimum-duration hysteresis pass removes single-frame flicker
between labels.

**Explicitly out of scope** (documented, not silently skipped): cross-checking
IMU/video clock offset via optical flow (the data itself resolves this — no
drift correction needed); pixel-level accuracy validation of MediaPipe's
pretrained hand model (trusted as-is; only the plumbing/robustness logic
around it is tested); a full strapdown INS/Madgwick orientation filter.

## Repository layout

```
src/nurvai_pipeline/   pipeline package (alignment, hand_tracking, segmentation, schema, cli, qa_render)
tests/                 unit tests + an end-to-end smoke test on a short real-data slice
notebooks/             exploratory analysis only — not part of the CLI/Docker execution path
data/                  input files (video.mp4, imu.csv, vts.csv)
```
