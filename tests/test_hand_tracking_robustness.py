from nurvai_pipeline.hand_tracking import apply_gap_holding


def _raw_hand(x=0.5, y=0.5):
    return {
        "landmarks_px": [(x * 100, y * 100)] * 21,
        "landmarks_norm": [(x, y, 0.0)] * 21,
        "detection_confidence": 0.9,
    }


def test_short_gap_is_held_not_lost():
    max_hold = 3
    frames_raw = [
        {"Right": _raw_hand()},  # detected
        {},  # gap frame 1
        {},  # gap frame 2
        {"Right": _raw_hand()},  # re-detected within hold window
    ]

    results = list(apply_gap_holding(iter(frames_raw), max_hold_frames=max_hold))

    states = [
        next((o["state"] for o in obs if o["handedness"] == "Right"), None)
        for obs, _lost in results
    ]
    assert states == ["detected", "held", "held", "detected"]
    lost_flags = [lost for _obs, lost in results]
    assert all("Right" not in lost for lost in lost_flags)


def test_long_gap_transitions_to_lost():
    max_hold = 2
    frames_raw = [
        {"Left": _raw_hand()},  # detected
        {},  # held (1)
        {},  # held (2) == max_hold
        {},  # lost (3 > max_hold)
        {},  # still lost
    ]

    results = list(apply_gap_holding(iter(frames_raw), max_hold_frames=max_hold))

    states = [
        next((o["state"] for o in obs if o["handedness"] == "Left"), None)
        for obs, _lost in results
    ]
    assert states == ["detected", "held", "held", None, None]

    lost_flags = [lost for _obs, lost in results]
    assert lost_flags[3] == ["Left"]
    assert lost_flags[4] == ["Left"]
    assert lost_flags[0] == []


def test_hand_never_seen_is_not_flagged_as_lost():
    frames_raw = [{}, {}, {}]
    results = list(apply_gap_holding(iter(frames_raw), max_hold_frames=5))
    for obs, lost in results:
        assert obs == []
        assert lost == []


def test_recovery_after_lost_resets_to_detected():
    max_hold = 1
    frames_raw = [
        {"Right": _raw_hand()},
        {},  # held
        {},  # lost
        {"Right": _raw_hand()},  # re-detected
    ]
    results = list(apply_gap_holding(iter(frames_raw), max_hold_frames=max_hold))
    states = [
        next((o["state"] for o in obs if o["handedness"] == "Right"), None)
        for obs, _lost in results
    ]
    assert states == ["detected", "held", None, "detected"]
