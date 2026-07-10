"""Tunable thresholds and constants for the pipeline, overridable via CLI flags."""

from dataclasses import dataclass, field


@dataclass
class HandTrackingConfig:
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_hands: int = 2
    max_hold_frames: int = 10  # ~333ms at 30fps: short-gap hold policy before declaring "lost"


@dataclass
class AlignmentConfig:
    use_accel_correction: bool = True
    complementary_filter_weight: float = 0.03  # weight given to accel-derived tilt correction


@dataclass
class SegmentationConfig:
    activity_z_threshold: float = 1.0
    min_segment_frames: int = 15  # ~0.5s at 30fps: hysteresis / debounce window


@dataclass
class PipelineConfig:
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
