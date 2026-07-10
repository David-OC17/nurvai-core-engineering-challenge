import numpy as np

from nurvai_pipeline.config import SegmentationConfig
from nurvai_pipeline.segmentation import (
    build_chunks,
    fuse_activity_score,
    label_segments,
    per_frame_segment_ids,
    robust_zscore,
)


def test_robust_zscore_constant_signal_is_zero():
    x = np.full(10, 5.0)
    z = robust_zscore(x)
    np.testing.assert_allclose(z, np.zeros(10))


def test_fuse_activity_score_detects_injected_spike():
    n = 60
    hand_velocity = np.random.default_rng(0).normal(0, 0.01, n)
    imu_energy = np.random.default_rng(1).normal(0, 0.01, n)
    spike = slice(20, 30)
    hand_velocity[spike] += 5.0  # clear spike well above baseline noise

    score = fuse_activity_score(hand_velocity, imu_energy)

    assert np.all(score[spike] > 1.0)
    baseline_mask = np.ones(n, dtype=bool)
    baseline_mask[spike] = False
    assert np.mean(score[baseline_mask] > 1.0) < 0.2  # baseline mostly below threshold


def test_label_segments_absorbs_single_frame_flicker():
    cfg = SegmentationConfig(activity_z_threshold=1.0, min_segment_frames=5)
    score = np.zeros(30)
    score[10] = 5.0  # single-frame flicker, should be absorbed into surrounding "static" run
    segments = label_segments(score, cfg)

    # Expect exactly one segment spanning the whole range, labeled static (flicker absorbed).
    assert len(segments) == 1
    label, start, end = segments[0]
    assert label is False
    assert (start, end) == (0, 30)


def test_label_segments_keeps_sustained_activity():
    cfg = SegmentationConfig(activity_z_threshold=1.0, min_segment_frames=5)
    score = np.zeros(30)
    score[10:20] = 5.0  # sustained activity, well above min_segment_frames
    segments = label_segments(score, cfg)

    labels = [label for label, _s, _e in segments]
    assert True in labels
    active_segment = next(s for s in segments if s[0] is True)
    assert active_segment[1] == 10 and active_segment[2] == 20


def test_build_chunks_and_segment_ids_are_consistent():
    cfg = SegmentationConfig(activity_z_threshold=1.0, min_segment_frames=5)
    score = np.zeros(20)
    score[5:15] = 3.0
    segments = label_segments(score, cfg)
    frame_t_ns = np.arange(20) * int(1e9 / 30)

    chunks = build_chunks(segments, frame_t_ns, score)
    seg_ids = per_frame_segment_ids(segments, 20)

    total_frames = sum(c["frame_count"] for c in chunks)
    assert total_frames == 20
    assert len(seg_ids) == 20
    # segment ids should be non-decreasing and contiguous
    assert seg_ids == sorted(seg_ids)
