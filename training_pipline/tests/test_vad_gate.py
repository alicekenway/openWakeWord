"""Regression tests for optional CTC-WAC VAD gating."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wuw_training.stages.summary import _keyword_threshold_metrics, _metrics  # noqa: E402
from wuw_training.vad import VAD_FRAME_SAMPLES, VadScorer  # noqa: E402


class _FakeVadModel:
    def __init__(self) -> None:
        self.reset_count = 0
        self.frames: list[np.ndarray] = []

    def reset_states(self) -> None:
        self.reset_count += 1

    def predict(self, frame: np.ndarray, *, frame_size: int) -> float:
        assert frame_size == VAD_FRAME_SAMPLES
        self.frames.append(frame.copy())
        return len(self.frames) / 10.0


def test_vad_scores_30ms_frames_and_resets_each_clip() -> None:
    scorer = VadScorer.__new__(VadScorer)
    scorer.model = _FakeVadModel()

    scores = scorer.frame_scores(np.ones(VAD_FRAME_SAMPLES + 1, dtype=np.float32))

    assert scores.tolist() == pytest.approx([0.1, 0.2])
    assert scorer.model.reset_count == 1
    assert len(scorer.model.frames) == 2
    assert scorer.model.frames[-1].shape == (VAD_FRAME_SAMPLES,)
    assert np.count_nonzero(scorer.model.frames[-1]) == 1


def test_vad_interval_uses_overlap_and_clips_padding() -> None:
    scores = np.asarray([0.1, 0.2, 0.9, 0.3], dtype=np.float32)

    score, start, end = VadScorer.interval_score(
        scores,
        start_seconds=0.06,
        end_seconds=0.09,
        audio_seconds=0.12,
        padding_ms=100,
    )

    assert score == pytest.approx(0.9)
    assert start == 0.0
    assert end == pytest.approx(0.12)


def test_summary_applies_vad_to_global_and_keyword_thresholds() -> None:
    records = [
        {
            "duration_seconds": 4.0,
            "audio_window_count": 2,
            "stage1_candidates": [
                {
                    "keyword_id": "wake",
                    "score": 0.9,
                    "stage2_threshold": 0.5,
                    "end_time": 1.0,
                    "audio_window_index": 0,
                    "vad_passed": False,
                },
                {
                    "keyword_id": "wake",
                    "score": 0.8,
                    "stage2_threshold": 0.5,
                    "end_time": 3.0,
                    "audio_window_index": 1,
                    "vad_passed": True,
                },
            ],
        }
    ]

    global_metrics = _metrics(records, 0, 0.5, 1.0, 0)
    keyword_metrics = _keyword_threshold_metrics(records, 0, 1.0, 0)

    assert global_metrics["false_accept_events"] == 1
    assert global_metrics["false_accept_crops"] == 1
    assert keyword_metrics["false_accept_events"] == 1
    assert keyword_metrics["false_accept_crops"] == 1


def test_summary_treats_missing_vad_field_as_legacy_pass() -> None:
    records = [
        {
            "duration_seconds": 2.0,
            "audio_window_count": 1,
            "stage1_candidates": [
                {"score": 0.8, "end_time": 1.0, "audio_window_index": 0}
            ],
        }
    ]

    metrics = _metrics(records, 0, 0.5, 1.0, 0)

    assert metrics["false_accept_events"] == 1
    assert metrics["false_accept_crops"] == 1
