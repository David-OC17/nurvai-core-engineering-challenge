import json
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SLICE_FRAMES = 60  # ~2s at 30fps, small enough to run fast in CI/local


def _make_data_slice(tmp_path: Path) -> tuple[Path, Path, Path]:
    vts_df = pd.read_csv(DATA_DIR / "vts.csv").sort_values("frame_index").reset_index(drop=True)
    vts_slice = vts_df.iloc[:SLICE_FRAMES]
    last_ts = int(vts_slice["timestamp_ns"].iloc[-1])

    imu_df = pd.read_csv(DATA_DIR / "imu.csv").sort_values("timestamp_ns").reset_index(drop=True)
    # Keep IMU samples covering the slice range with margin on both sides.
    imu_slice = imu_df[imu_df["timestamp_ns"] <= last_ts + int(1e8)].reset_index(drop=True)

    vts_path = tmp_path / "vts_slice.csv"
    imu_path = tmp_path / "imu_slice.csv"
    vts_slice.to_csv(vts_path, index=False)
    imu_slice.to_csv(imu_path, index=False)

    src_video = cv2.VideoCapture(str(DATA_DIR / "video.mp4"))
    fps = src_video.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_path = tmp_path / "video_slice.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for _ in range(SLICE_FRAMES):
        ok, frame = src_video.read()
        if not ok:
            break
        writer.write(frame)
    src_video.release()
    writer.release()

    return video_path, imu_path, vts_path


@pytest.mark.skipif(not (DATA_DIR / "video.mp4").exists(), reason="requires real challenge data")
def test_cli_run_end_to_end_on_data_slice(tmp_path):
    video_path, imu_path, vts_path = _make_data_slice(tmp_path)
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nurvai_pipeline.cli",
            "run",
            "--video",
            str(video_path),
            "--imu",
            str(imu_path),
            "--vts",
            str(vts_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    enriched_path = output_dir / "enriched.jsonl"
    chunks_path = output_dir / "action_chunks.jsonl"
    assert enriched_path.exists()
    assert chunks_path.exists()

    lines = enriched_path.read_text().splitlines()
    assert len(lines) == SLICE_FRAMES

    seen_frame_indices = set()
    for line in lines:
        doc = json.loads(line)
        assert "frame_index" in doc
        assert "imu" in doc
        assert "action" in doc
        seen_frame_indices.add(doc["frame_index"])
    assert len(seen_frame_indices) == SLICE_FRAMES

    chunk_lines = [json.loads(l) for l in chunks_path.read_text().splitlines()]
    total_chunk_frames = sum(c["frame_count"] for c in chunk_lines)
    assert total_chunk_frames == SLICE_FRAMES

    # Chunk frame ranges should be contiguous and non-overlapping.
    sorted_chunks = sorted(chunk_lines, key=lambda c: c["start_frame"])
    for prev, curr in zip(sorted_chunks, sorted_chunks[1:]):
        assert curr["start_frame"] == prev["end_frame"] + 1
