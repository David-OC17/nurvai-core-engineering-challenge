"""Phase 2: ego-hand tracking via MediaPipe Hand Landmarker, with an explicit
detected/held/lost state machine for robustness to occlusion and lighting changes.

MediaPipe's Python GPU delegate on Linux is inconsistent for mainline pip
wheels; since this is an offline batch job (no real-time constraint), hand
tracking deliberately runs on CPU delegate, which is fast enough at 30fps
offline. See README for the full trade-off discussion.
"""

import urllib.request
from pathlib import Path
from typing import Iterator, Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from nurvai_pipeline.config import HandTrackingConfig

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "nurvai_pipeline" / "hand_landmarker.task"

Handedness = Literal["Left", "Right"]


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _build_landmarker(cfg: HandTrackingConfig, model_path: Path) -> mp_vision.HandLandmarker:
    base_options = mp_python.BaseOptions(
        model_asset_path=str(model_path),
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=cfg.num_hands,
        min_hand_detection_confidence=cfg.min_hand_detection_confidence,
        min_hand_presence_confidence=cfg.min_hand_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_raw_per_frame(
    video_path: str | Path,
    frame_timestamps_ns: np.ndarray,
    cfg: HandTrackingConfig | None = None,
    model_path: Path | None = None,
) -> Iterator[dict[Handedness, dict]]:
    """Run the MediaPipe HandLandmarker over every frame in VIDEO mode.

    Yields, per frame, a dict keyed by handedness ("Left"/"Right") for
    however many hands (0-2) were detected that frame, each holding raw
    pixel/normalized landmarks and the detection confidence.
    """
    cfg = cfg or HandTrackingConfig()
    model_path = ensure_model(model_path or DEFAULT_MODEL_PATH)
    landmarker = _build_landmarker(cfg, model_path)

    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = 0
        n_frames = len(frame_timestamps_ns)
        while frame_idx < n_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_timestamps_ns[frame_idx] / 1e6)
            h, w = frame_bgr.shape[:2]
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_hands: dict[Handedness, dict] = {}
            for hand_landmarks, world_landmarks, handedness in zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness
            ):
                label: Handedness = handedness[0].category_name  # "Left" | "Right"
                confidence = float(handedness[0].score)
                landmarks_px = [(lm.x * w, lm.y * h) for lm in hand_landmarks]
                landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                frame_hands[label] = {
                    "landmarks_px": landmarks_px,
                    "landmarks_norm": landmarks_norm,
                    "detection_confidence": confidence,
                }
            yield frame_hands
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()


def apply_gap_holding(
    per_frame_raw: Iterator[dict[Handedness, dict]],
    max_hold_frames: int,
) -> Iterator[tuple[list[dict], list[Handedness]]]:
    """Apply the detected/held/lost state machine independently per hand slot.

    Yields, per frame, a tuple of (hand_observations, hands_lost) where
    hand_observations is a list of 0-2 dicts (one per currently detected-or-
    held hand) and hands_lost flags any slot that was previously tracked but
    has now exceeded the hold window without a fresh detection.
    """
    last_seen: dict[Handedness, dict] = {}
    frames_since_update: dict[Handedness, int] = {"Left": 0, "Right": 0}
    ever_detected: dict[Handedness, bool] = {"Left": False, "Right": False}

    for frame_hands in per_frame_raw:
        observations: list[dict] = []
        hands_lost: list[Handedness] = []

        for handedness in ("Left", "Right"):
            if handedness in frame_hands:
                last_seen[handedness] = frame_hands[handedness]
                frames_since_update[handedness] = 0
                ever_detected[handedness] = True
                observations.append(
                    {
                        "handedness": handedness,
                        "state": "detected",
                        "frames_since_update": 0,
                        **frame_hands[handedness],
                    }
                )
            elif ever_detected[handedness]:
                frames_since_update[handedness] += 1
                if frames_since_update[handedness] <= max_hold_frames:
                    observations.append(
                        {
                            "handedness": handedness,
                            "state": "held",
                            "frames_since_update": frames_since_update[handedness],
                            **last_seen[handedness],
                        }
                    )
                else:
                    hands_lost.append(handedness)

        yield observations, hands_lost
