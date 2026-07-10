"""QA utility: renders keypoints + synchronized telemetry over the original video."""

import json
from pathlib import Path

import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]

STATE_COLOR = {
    "detected": (0, 255, 0),   # green
    "held": (0, 200, 255),     # amber
}

HANDEDNESS_POINT_COLOR = {
    "Left": (255, 100, 0),
    "Right": (0, 100, 255),
}


def _load_frame_docs(jsonl_path: str | Path) -> dict[int, dict]:
    docs = {}
    with open(jsonl_path) as f:
        for line in f:
            doc = json.loads(line)
            docs[doc["frame_index"]] = doc
    return docs


def render_qa_video(
    video_path: str | Path,
    jsonl_path: str | Path,
    output_path: str | Path,
) -> Path:
    docs = _load_frame_docs(jsonl_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            doc = docs.get(frame_idx)
            if doc is not None:
                _draw_frame(frame, doc)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return Path(output_path)


def _draw_frame(frame, doc: dict) -> None:
    for hand in doc["hands"]:
        color = STATE_COLOR.get(hand["state"], (255, 255, 255))
        point_color = HANDEDNESS_POINT_COLOR.get(hand["handedness"], (255, 255, 255))
        pts = [(int(x), int(y)) for x, y in hand["landmarks_px"]]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(frame, p, 3, point_color, -1)

        label_pos = pts[0]
        cv2.putText(
            frame,
            f"{hand['handedness']} ({hand['state']})",
            (label_pos[0], label_pos[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            point_color,
            1,
            cv2.LINE_AA,
        )

    imu = doc["imu"]
    action = doc["action"]
    lines = [
        f"frame {doc['frame_index']}  t={doc['video_time_s']:.2f}s",
        f"gyro=({imu['gyro']['x']:.2f},{imu['gyro']['y']:.2f},{imu['gyro']['z']:.2f})",
        f"accel=({imu['accel']['x']:.2f},{imu['accel']['y']:.2f},{imu['accel']['z']:.2f})",
        f"action={action['label']} (score={action.get('activity_score') or 0:.2f}) seg={action['segment_id']}",
    ]
    if doc.get("hands_lost"):
        lines.append(f"hands_lost={doc['hands_lost']}")

    for i, line in enumerate(lines):
        y = 25 + i * 20
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
