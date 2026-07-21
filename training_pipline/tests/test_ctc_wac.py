"""Focused tests for the opt-in frozen-CTC + WAC pipeline."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wuw_training.artifacts import write_json, write_jsonl  # noqa: E402
from wuw_training.ctc_wac import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    BoundedCtcKeywordScorer,
    CtcWacFeatureBlock,
    Keyword,
    Stage1Contract,
    StreamingCtcStage1,
    best_ctc_candidate,
    ctc_candidate_token_alignments,
    ctc_keyword_alignment_trace,
    ctc_keyword_alignment_traces,
    ctc_keyword_score_trace,
    feature_bundle_paths,
    feature_bundle_valid,
    keyword_token_fingerprint,
    load_keywords,
    make_ctc_wac_model,
    rank_keyword_scores,
)
from wuw_training.runner import PipelineRunner  # noqa: E402
from wuw_training.config import load_ini_config  # noqa: E402
from wuw_training.legacy import get_legacy_module  # noqa: E402
from wuw_training.stages.train import FeatureBlock  # noqa: E402
from wuw_training.stages.feature import _merge_ctc_wac_features  # noqa: E402
from wuw_training.stages.testing import _ctc_wac_record  # noqa: E402
import wuw_training.ctc_wac as ctc_wac_module  # noqa: E402


def _keyword_file(root: Path) -> Path:
    path = root / "keywords.json"
    write_json(
        path,
        {
            "keywords": [
                {"id": "wake_a", "display_text": "wake a", "token_ids": [1, 2], "threshold": -2.0},
                {"id": "wake_b", "display_text": "wake b", "token_ids": [2, 1], "threshold": -2.0},
            ]
        },
    )
    return path


def _write_bundle(
    path: Path,
    *,
    label: int,
    keywords_path: Path,
    seed: int,
    debug_alignment: bool = False,
) -> None:
    keywords = load_keywords(keywords_path)
    rng = np.random.default_rng(seed)
    lengths = np.asarray([3, 6, 4, 5], dtype=np.int32)
    offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])
    features = rng.standard_normal((int(offsets[-1]), 5), dtype=np.float32)
    # Three rows pass stage 1. The final row must remain in the artifact but be
    # dropped only by the CTC-WAC train loader.
    scores = np.asarray(
        [[-1.0, -4.0], [-4.0, -1.1], [-1.2, -3.0], [-4.0, -3.0]], dtype=np.float32
    )
    top, margin, winner = rank_keyword_scores(scores)
    onehot = np.zeros_like(scores)
    onehot[np.arange(scores.shape[0]), winner] = 1.0
    paths = feature_bundle_paths(path)
    np.save(paths.features, features)
    np.save(paths.offsets, offsets)
    np.save(paths.lengths, lengths)
    np.save(paths.all_scores, scores)
    np.save(paths.top_score, top.reshape(-1, 1))
    np.save(paths.margin, margin.reshape(-1, 1))
    np.save(paths.winner_onehot, onehot)
    expected_keyword_ids = (
        ["wake_a", "wake_b", "wake_a", "wake_b"] if label == 1 else [None] * len(lengths)
    )
    write_jsonl(
        paths.rows,
        [
            {
                "row": index,
                "label": label,
                "expected_keyword_id": expected_keyword_ids[index],
            }
            for index in range(lengths.shape[0])
        ],
    )
    if debug_alignment:
        write_jsonl(
            paths.debug_alignments,
            [
                {
                    "schema_version": 1,
                    "source_index": index,
                    "status": "ok",
                    "candidate": {"tokens": []},
                }
                for index in range(lengths.shape[0])
            ],
        )
    write_json(
        paths.summary,
        {
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "feature_count": int(lengths.shape[0]),
            "feature_storage_shape": list(features.shape),
            "feature_dim": 5,
            "keyword_count": len(keywords),
            "keyword_ids": [item.id for item in keywords],
            "keyword_token_fingerprint": keyword_token_fingerprint(keywords),
            "input_count": int(lengths.shape[0]),
            "input_duration_seconds": 3600.0,
            "invalid_alignment_rows": 0,
            "expected_keyword_counts": {
                "wake_a": 2 if label == 1 else 0,
                "wake_b": 2 if label == 1 else 0,
            },
            "expected_keyword_invalid_alignment_counts": {"wake_a": 0, "wake_b": 0},
            "debug_alignment_enabled": debug_alignment,
            "debug_alignment_jsonl": str(paths.debug_alignments) if debug_alignment else None,
            "debug_alignment_rows": int(lengths.shape[0]) if debug_alignment else 0,
            "error_count": 0,
        },
    )


def test_audio_to_fbank_matches_wenet_pcm_scaling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract_path = tmp_path / "contract.json"
    write_json(
        contract_path,
        {
            "schema_version": 2,
            "sample_rate": 16000,
            "fbank": {
                "num_mel_bins": 3,
                "frame_length_ms": 25.0,
                "frame_shift_ms": 10.0,
                "dither": 0.0,
            },
            "chunk_frames": 7,
            "inputs": {"features": "chunk"},
            "outputs": {"encoder": "encoder_out", "ctc_log_probs": "ctc_log_probs"},
        },
    )
    contract = Stage1Contract.from_json(contract_path)
    captured: dict[str, torch.Tensor] = {}

    class FakeKaldi:
        @staticmethod
        def fbank(waveform: torch.Tensor, **_kwargs: object) -> torch.Tensor:
            captured["waveform"] = waveform.clone()
            return torch.ones((2, 3), dtype=torch.float32)

    fake_torchaudio = SimpleNamespace(compliance=SimpleNamespace(kaldi=FakeKaldi()))
    monkeypatch.setattr(ctc_wac_module, "_torchaudio", lambda: (torch, fake_torchaudio))

    result = ctc_wac_module.audio_to_fbank(
        np.asarray([-1.0, 0.5, 1.0], dtype=np.float32), contract
    )

    assert contract.waveform_scale == float(1 << 15)
    assert torch.equal(
        captured["waveform"],
        torch.tensor([[-32768.0, 16384.0, 32768.0]], dtype=torch.float32),
    )
    assert result.shape == (2, 3)


def test_feature_bundle_rejects_a_stale_stage1_contract(
    tmp_path: Path,
) -> None:
    keywords_path = _keyword_file(tmp_path)
    features = tmp_path / "features.npy"
    _write_bundle(features, label=1, keywords_path=keywords_path, seed=3)

    assert feature_bundle_valid(features)
    assert not feature_bundle_valid(
        features,
        expected_stage1_contract_fingerprint="new-contract",
    )

    paths = feature_bundle_paths(features)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    summary["stage1_contract_fingerprint"] = "new-contract"
    write_json(paths.summary, summary)
    assert feature_bundle_valid(
        features,
        expected_stage1_contract_fingerprint="new-contract",
    )


def test_onnx_threads_follow_slurm_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert ctc_wac_module._onnx_intra_op_threads() == 2

    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert ctc_wac_module._onnx_intra_op_threads() == 8


def test_ctc_viterbi_trace_prefers_the_right_token_order() -> None:
    probabilities = np.asarray(
        [
            [0.05, 0.90, 0.05],  # token 1
            [0.75, 0.10, 0.15],  # blank separates the labels
            [0.05, 0.05, 0.90],  # token 2
        ],
        dtype=np.float32,
    )
    log_probs = np.log(probabilities)
    right = ctc_keyword_score_trace(log_probs, [1, 2], blank_id=0)
    wrong = ctc_keyword_score_trace(log_probs, [2, 1], blank_id=0)
    repeated = ctc_keyword_score_trace(log_probs[[0, 1, 0]], [1, 1], blank_id=0)
    assert np.isfinite(right).all()
    assert right[-1] > wrong[-1]
    assert np.isfinite(repeated[-1])


def test_ctc_alignment_trace_reports_token_boundaries_not_trailing_blanks() -> None:
    probabilities = np.asarray(
        [
            [0.02, 0.96, 0.02],  # token 1
            [0.95, 0.03, 0.02],  # blank
            [0.02, 0.02, 0.96],  # token 2
            [0.99, 0.005, 0.005],  # trailing blank
        ],
        dtype=np.float32,
    )
    trace = ctc_keyword_alignment_trace(np.log(probabilities), [1, 2], blank_id=0)
    assert trace.starts[2] == 0
    assert trace.ends[2] == 2
    assert trace.ends[3] == 2


def test_ctc_candidate_token_alignments_report_frame_spans_and_normalized_scores() -> None:
    probabilities = np.asarray(
        [
            [0.02, 0.90, 0.08],  # token 1
            [0.05, 0.80, 0.15],  # token 1
            [0.90, 0.05, 0.05],  # inter-token blank
            [0.10, 0.20, 0.70],  # token 2
            [0.05, 0.15, 0.80],  # token 2
            [0.98, 0.01, 0.01],  # trailing blank outside the non-blank candidate range
        ],
        dtype=np.float32,
    )
    log_probs = np.log(probabilities)
    alignments = ctc_candidate_token_alignments(
        log_probs,
        [1, 2],
        candidate_start_frame=0,
        candidate_end_frame=4,
        blank_id=0,
    )

    assert [(item.token_id, item.start_frame, item.end_frame, item.frame_count) for item in alignments] == [
        (1, 0, 1, 2),
        (2, 3, 4, 2),
    ]
    assert alignments[0].normalized_score == pytest.approx((log_probs[0, 1] + log_probs[1, 1]) / 2)
    assert alignments[1].normalized_score == pytest.approx((log_probs[3, 2] + log_probs[4, 2]) / 2)


def test_ctc_candidate_token_alignments_keep_repeated_tokens_separate() -> None:
    probabilities = np.asarray(
        [
            [0.02, 0.96],  # first token 1
            [0.98, 0.02],  # required blank between repeated token IDs
            [0.02, 0.96],  # second token 1
        ],
        dtype=np.float32,
    )
    alignments = ctc_candidate_token_alignments(
        np.log(probabilities),
        [1, 1],
        candidate_start_frame=0,
        candidate_end_frame=2,
        blank_id=0,
    )

    assert [(item.token_index, item.start_frame, item.end_frame) for item in alignments] == [
        (0, 0, 0),
        (1, 2, 2),
    ]


def test_bounded_ctc_trace_matches_an_exact_rolling_reference() -> None:
    """A finite horizon must discard old prefixes without resetting CTC state."""

    rng = np.random.default_rng(19)
    probabilities = rng.random((11, 4), dtype=np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    log_probs = np.log(probabilities)
    tokens = [1, 2]
    horizon = 4

    bounded = ctc_keyword_alignment_trace(
        log_probs,
        tokens,
        blank_id=0,
        max_search_frames=horizon,
    )
    incremental = BoundedCtcKeywordScorer(tokens, blank_id=0, max_search_frames=horizon)
    for frame_index in range(log_probs.shape[0]):
        start = max(0, frame_index - horizon + 1)
        reference = ctc_keyword_alignment_trace(log_probs[start:frame_index + 1], tokens, blank_id=0)
        score, candidate_start, candidate_end = incremental.push(log_probs[frame_index])
        expected_start = -1 if reference.starts[-1] < 0 else int(reference.starts[-1] + start)
        expected_end = -1 if reference.ends[-1] < 0 else int(reference.ends[-1] + start)
        assert np.isclose(bounded.scores[frame_index], reference.scores[-1])
        assert (int(bounded.starts[frame_index]), int(bounded.ends[frame_index])) == (
            expected_start,
            expected_end,
        )
        assert np.isclose(score, reference.scores[-1])
        assert (candidate_start, candidate_end) == (expected_start, expected_end)
        if candidate_start >= 0:
            assert candidate_start >= start


def test_best_ctc_candidate_obeys_the_configured_search_horizon() -> None:
    probabilities = np.asarray(
        [
            [0.02, 0.96, 0.02],  # token 1, frame 0
            [0.96, 0.02, 0.02],  # blank
            [0.02, 0.02, 0.96],  # token 2, frame 2
            [0.98, 0.01, 0.01],  # trailing blank
            [0.98, 0.01, 0.01],
            [0.98, 0.01, 0.01],
        ],
        dtype=np.float32,
    )
    keyword = Keyword("wake", "wake", (1, 2), -3.0)
    candidate = best_ctc_candidate(
        np.log(probabilities),
        [keyword],
        blank_id=0,
        max_search_frames=3,
    )
    assert candidate.start_frame >= candidate.frame - 2
    traces, starts, _ends = ctc_keyword_alignment_traces(
        np.log(probabilities),
        [keyword],
        blank_id=0,
        max_search_frames=3,
    )
    assert candidate.start_frame == int(starts[candidate.frame, candidate.keyword_index])
    assert candidate.top_score == pytest.approx(float(traces[candidate.frame, candidate.keyword_index]))


def test_ctc_context_augmentation_uses_variable_real_background_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torchaudio")
    legacy = get_legacy_module()
    source_path = tmp_path / "source.wav"
    noise_path = tmp_path / "background.wav"
    source = torch.full((48_000,), 0.1, dtype=torch.float32)  # 3 seconds
    long_source = torch.full((80_000,), 0.1, dtype=torch.float32)
    monkeypatch.setattr(
        legacy,
        "load_audio_float",
        lambda path, sr: long_source.clone() if Path(path).name == "near_limit.wav" else source.clone(),
    )

    signal, active, context = legacy._ctc_context_signal(
        source_path,
        target_samples=81_920,  # 2 * 2.56 seconds at 16 kHz
        long_audio_mode="filter",
        rng=random.Random(11),
        sample_rate=16_000,
        leading_context_range_samples=(16_000, 32_000),
    )
    assert 16_000 <= context <= 32_000
    assert signal.numel() == source.numel() + context
    assert torch.equal(signal[context:], active)

    capped_signal, _capped_active, capped_context = legacy._ctc_context_signal(
        tmp_path / "near_limit.wav",
        target_samples=81_920,
        long_audio_mode="filter",
        rng=random.Random(11),
        sample_rate=16_000,
        leading_context_range_samples=(16_000, 32_000),
    )
    assert capped_context == 1_920
    assert capped_signal.numel() == 81_920

    requested_background_lengths: list[int] = []
    saved_lengths: list[int] = []
    monkeypatch.setattr(
        legacy,
        "_ctc_background_window",
        lambda _path, *, target_samples, rng, sample_rate: (
            requested_background_lengths.append(target_samples)
            or torch.ones(target_samples, dtype=torch.float32)
        ),
    )
    monkeypatch.setattr(
        legacy,
        "save_wav",
        lambda _path, values, sr: saved_lengths.append(int(values.numel())),
    )
    legacy._init_augment_worker(
        {
            "output_dir": str(tmp_path / "augmented"),
            "noise_paths": [str(noise_path)],
            "target_samples": 81_920,
            "snr_low": 0.0,
            "snr_high": 0.0,
            "artificial_prob": 0.0,
            "random_gain_db": 0.0,
            "sample_rate": 16_000,
            "placement": "end",
            "ctc_context": True,
            "long_audio_mode": "filter",
            "leading_context_range_samples": (16_000, 32_000),
            "seed": 29,
            "overwrite": True,
        }
    )
    result = legacy._augment_audio_worker((0, 0, {"path": str(source_path)}))
    record = result["record"]
    assert record is not None
    assert record["ctc_source_samples"] == source.numel()
    assert 16_000 <= record["ctc_leading_context_samples"] <= 32_000
    assert record["ctc_window_samples"] == source.numel() + record["ctc_leading_context_samples"]
    assert requested_background_lengths == [record["ctc_window_samples"]]
    assert saved_lengths == [record["ctc_window_samples"]]


def test_ctc_background_window_decodes_only_the_requested_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torchaudio")
    legacy = get_legacy_module()
    background = tmp_path / "hour_long_background.wav"
    target_samples = 81_920
    source_sample_rate = 48_000
    source_total_frames = source_sample_rate * 3600
    info_calls: list[str] = []
    load_calls: list[dict[str, int]] = []

    def fake_info(path: str) -> SimpleNamespace:
        info_calls.append(path)
        return SimpleNamespace(sample_rate=source_sample_rate, num_frames=source_total_frames)

    def fake_load(path: str, *, frame_offset: int, num_frames: int) -> tuple[torch.Tensor, int]:
        load_calls.append({"frame_offset": frame_offset, "num_frames": num_frames})
        return torch.ones((1, num_frames), dtype=torch.float32), source_sample_rate

    monkeypatch.setattr(legacy.torchaudio, "info", fake_info)
    monkeypatch.setattr(legacy.torchaudio, "load", fake_load)
    monkeypatch.setattr(
        legacy,
        "waveform_to_float",
        lambda waveform, decoded_sample_rate, sr: torch.ones(target_samples + 1, dtype=torch.float32),
    )
    monkeypatch.setattr(
        legacy,
        "load_audio_float",
        lambda *_args, **_kwargs: pytest.fail("full background recording must not be decoded"),
    )
    legacy._CTC_BACKGROUND_INFO_CACHE.clear()

    first = legacy._ctc_background_window(
        background,
        target_samples=target_samples,
        rng=random.Random(7),
        sample_rate=16_000,
    )
    second = legacy._ctc_background_window(
        background,
        target_samples=target_samples,
        rng=random.Random(8),
        sample_rate=16_000,
    )

    expected_source_frames = math.ceil(target_samples * source_sample_rate / 16_000) + 2
    assert first.shape == second.shape == (target_samples,)
    assert info_calls == [str(background)]
    assert len(load_calls) == 2
    assert all(call["num_frames"] == expected_source_frames for call in load_calls)
    assert all(
        0 <= call["frame_offset"] <= source_total_frames - expected_source_frames
        for call in load_calls
    )


def test_feature_bundle_crops_variable_length_ctc_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    keywords_path = _keyword_file(tmp_path)
    contract_path = tmp_path / "contract.json"
    write_json(
        contract_path,
        {
            "schema_version": 2,
            "sample_rate": 16000,
            "fbank": {"num_mel_bins": 80, "frame_length_ms": 25.0, "frame_shift_ms": 10.0, "dither": 0.0},
            "chunk_frames": 7,
            "chunk_stride_frames": 7,
            "minimum_input_frames": 1,
            "pad_final_chunk": False,
            "inputs": {"features": "chunk"},
            "outputs": {"encoder": "encoder_out", "ctc_log_probs": "ctc_log_probs"},
            "ctc_output_is_log_probs": True,
            "encoder_frame_shift_ms": 40.0,
            "encoder_output_size": 3,
            "vocab_size": 3,
        },
    )

    class FakeStage1:
        providers = ["fake"]

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def infer_fbank(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            frames = int(values[0, 0])
            encoder = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
            probabilities = np.full((frames, 3), 0.005, dtype=np.float32)
            probabilities[:, 0] = 0.99
            probabilities[0] = [0.025, 0.95, 0.025]
            probabilities[-1] = [0.025, 0.025, 0.95]
            return encoder, np.log(probabilities)

    monkeypatch.setattr(ctc_wac_module, "StreamingCtcStage1", FakeStage1)
    monkeypatch.setattr(
        ctc_wac_module,
        "load_audio",
        lambda path, _sample_rate: np.zeros(5 if path.name == "one.wav" else 7, dtype=np.float32),
    )
    monkeypatch.setattr(
        ctc_wac_module,
        "audio_to_fbank",
        lambda audio, _contract: np.asarray([[audio.size]], dtype=np.float32),
    )
    output = tmp_path / "bundle.npy"
    arguments = {
        "records": [
            {"id": "one", "path": str(tmp_path / "one.wav"), "label": 1, "text": "wake a"},
            {"id": "two", "path": str(tmp_path / "two.wav"), "label": 1, "text": "wake b"},
        ],
        "output_file": output,
        "model_path": tmp_path / "stage1.onnx",
        "contract_path": contract_path,
        "keywords_path": keywords_path,
        "candidate_pre_margin_frames": 1,
    }
    summary = ctc_wac_module.generate_ctc_wac_feature_bundle(**arguments)
    paths = feature_bundle_paths(output)
    lengths = np.load(paths.lengths)
    offsets = np.load(paths.offsets)
    assert summary["feature_count"] == 2
    assert summary["expected_keyword_counts"] == {"wake_a": 1, "wake_b": 1}
    assert summary["expected_keyword_invalid_alignment_counts"] == {"wake_a": 0, "wake_b": 0}
    assert lengths.tolist() == [5, 7]
    assert offsets.tolist() == [0, 5, 12]
    assert np.load(paths.features).shape == (12, 3)
    rows = [json.loads(line) for line in paths.rows.read_text(encoding="utf-8").splitlines()]
    assert [row["expected_keyword_id"] for row in rows] == ["wake_a", "wake_b"]
    assert summary["debug_alignment_enabled"] is False
    assert not paths.debug_alignments.exists()
    assert not feature_bundle_valid(output, require_debug_alignments=True)

    debug_summary = ctc_wac_module.generate_ctc_wac_feature_bundle(
        **arguments,
        overwrite=True,
        debug_alignments=True,
    )
    debug_rows = [json.loads(line) for line in paths.debug_alignments.read_text(encoding="utf-8").splitlines()]
    assert debug_summary["debug_alignment_enabled"] is True
    assert debug_summary["debug_alignment_rows"] == 2
    assert feature_bundle_valid(output, require_debug_alignments=True)
    assert [item["status"] for item in debug_rows] == ["ok", "ok"]
    token_rows = debug_rows[0]["candidate"]["tokens"]
    assert [item["token_id"] for item in token_rows] == [1, 2]
    assert all(item["start_frame"] <= item["end_frame"] for item in token_rows)
    assert all(item["normalized_score"] <= 0.0 for item in token_rows)

    normal_summary = ctc_wac_module.generate_ctc_wac_feature_bundle(**arguments)
    assert normal_summary["debug_alignment_enabled"] is False
    assert not paths.debug_alignments.exists()
    assert not feature_bundle_valid(output, require_debug_alignments=True)


def test_masked_wac_pooling_ignores_tail_padding() -> None:
    model = make_ctc_wac_model(
        feature_dim=3,
        keyword_count=2,
        model_config={"frame_hidden": 4, "frame_layers": 1, "head_hidden": 4, "dropout": 0.0},
    ).eval()
    features = torch.zeros((1, 4, 3), dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    scalar = torch.zeros((1, 1), dtype=torch.float32)
    winner = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    with torch.inference_mode():
        first = model(features, mask, scalar, scalar, winner)
        features[:, 2:] = 1000.0
        second = model(features, mask, scalar, scalar, winner)
    assert torch.allclose(first, second, rtol=1e-6, atol=1e-6)


def test_slurm_merge_rebuilds_ragged_offsets(tmp_path: Path) -> None:
    keywords_path = _keyword_file(tmp_path)
    first = tmp_path / "shard_a.npy"
    second = tmp_path / "shard_b.npy"
    output = tmp_path / "merged.npy"
    _write_bundle(first, label=1, keywords_path=keywords_path, seed=3, debug_alignment=True)
    _write_bundle(second, label=1, keywords_path=keywords_path, seed=4, debug_alignment=True)
    ctx = SimpleNamespace(
        step="feature.positive_train",
        section={"output_file": str(output), "label": "1", "split": "train", "debug_alignment": "yes"},
        config=SimpleNamespace(resolve_path=lambda value: Path(value).resolve()),
    )
    result = _merge_ctc_wac_features(
        ctx,
        [{"output_file": str(first)}, {"output_file": str(second)}],
    )
    paths = feature_bundle_paths(output)
    lengths = np.load(paths.lengths)
    offsets = np.load(paths.offsets)
    rows = [json.loads(line) for line in paths.rows.read_text(encoding="utf-8").splitlines()]
    assert feature_bundle_valid(output, require_debug_alignments=True)
    assert result["feature_count"] == 8
    assert lengths.tolist() == [3, 6, 4, 5, 3, 6, 4, 5]
    assert offsets.tolist() == [0, 3, 9, 13, 18, 21, 27, 31, 36]
    assert [row["row"] for row in rows] == list(range(8))
    debug_rows = [json.loads(line) for line in paths.debug_alignments.read_text(encoding="utf-8").splitlines()]
    assert len(debug_rows) == 8
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert summary["expected_keyword_counts"] == {"wake_a": 4, "wake_b": 4}
    assert summary["expected_keyword_invalid_alignment_counts"] == {"wake_a": 0, "wake_b": 0}


def test_generated_contract_fields_and_first_chunk_attention_mask(tmp_path: Path) -> None:
    contract_path = tmp_path / "stage1.contract.json"
    write_json(
        contract_path,
        {
            "sample_rate": 16000,
            "fbank": {"num_mel_bins": 80, "dither": 0.0},
            "chunk_frames": 67,
            "chunk_stride_frames": 64,
            "minimum_input_frames": 7,
            "pad_final_chunk": False,
            "initial_offset": 64,
            "inputs": {"features": "chunk", "offset": "offset"},
            "outputs": {"encoder": "encoder_out", "ctc_log_probs": "ctc_log_probs"},
            "ctc_output_is_log_probs": True,
            "attention_mask": {"input": "att_mask", "cache_frames": 64, "chunk_frames": 16},
        },
    )
    contract = Stage1Contract.from_json(contract_path)
    assert contract.chunk_frames == 67
    assert contract.chunk_stride_frames == 64
    assert contract.minimum_input_frames == 7
    assert contract.pad_final_chunk is False

    # This small unit test does not need ONNX Runtime.  It checks the mask
    # policy used by WeNet's own fixed-cache ONNX simulation.
    runner = StreamingCtcStage1.__new__(StreamingCtcStage1)
    runner.contract = contract
    runner._offset = contract.initial_offset
    runner._chunks_run = 0
    first = runner._streaming_attention_mask()
    assert not first[..., :64].any()
    assert first[..., 64:].all()
    runner._chunks_run = 1
    assert runner._streaming_attention_mask().all()


def test_bundle_keeps_all_rows_but_loader_applies_stage1_threshold(tmp_path: Path) -> None:
    keywords_path = _keyword_file(tmp_path)
    bundle = tmp_path / "features.npy"
    _write_bundle(bundle, label=1, keywords_path=keywords_path, seed=7)
    block = FeatureBlock("feature.positive", bundle, 1, "train", (6, 5), 4)
    loaded = CtcWacFeatureBlock.from_feature_block(block, load_keywords(keywords_path))
    assert loaded.input_count == 4
    assert loaded.retained_count == 3
    assert loaded.filtering_summary()["dropped_rows"] == 1
    # Thresholds may change without invalidating frozen CTC encoder features:
    # only id/token mapping must stay the same.
    changed = tmp_path / "keywords_changed_threshold.json"
    write_json(
        changed,
        {
            "keywords": [
                {"id": "wake_a", "display_text": "wake a", "token_ids": [1, 2], "threshold": -0.9},
                {"id": "wake_b", "display_text": "wake b", "token_ids": [2, 1], "threshold": -0.9},
            ]
        },
    )
    changed_loaded = CtcWacFeatureBlock.from_feature_block(block, load_keywords(changed))
    assert changed_loaded.retained_count == 0


def test_cascade_record_feeds_all_masked_wac_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    keywords_path = _keyword_file(tmp_path)
    keywords = load_keywords(keywords_path)

    class FakeStage1:
        contract = SimpleNamespace(sample_rate=16000, blank_id=0, encoder_frame_shift_ms=40.0)

        def infer_fbank(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            encoder = np.arange(20, dtype=np.float32).reshape(4, 5)
            probabilities = np.asarray(
                [[0.05, 0.90, 0.05], [0.80, 0.10, 0.10], [0.05, 0.05, 0.90], [0.8, 0.1, 0.1]],
                dtype=np.float32,
            )
            return encoder, np.log(probabilities)

    class FakeStage2:
        def __init__(self) -> None:
            self.feed: dict[str, np.ndarray] | None = None

        def run(self, _outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.feed = feed
            return [np.asarray([[0.75]], dtype=np.float32)]

    monkeypatch.setattr("wuw_training.stages.testing.load_audio", lambda path, sample_rate: np.zeros(32000, dtype=np.float32))
    monkeypatch.setattr(
        "wuw_training.stages.testing.audio_to_fbank",
        lambda audio, contract: np.zeros((4, 80), dtype=np.float32),
    )
    stage2 = FakeStage2()
    detail = _ctc_wac_record(
        record={"id": "one", "path": str(tmp_path / "audio.wav"), "text": "wake a"},
        stage1=FakeStage1(),  # type: ignore[arg-type]
        keywords=keywords,
        stage2=stage2,
        feature_dim=5,
        expected_label=1,
    )
    assert detail["stage1_candidate_count"] >= 1
    assert stage2.feed is not None
    assert set(stage2.feed) == {"encoder_features", "frame_mask", "top_score", "margin", "winner_onehot"}
    assert stage2.feed["encoder_features"].shape[0] == 1
    assert stage2.feed["encoder_features"].shape[2] == 5
    assert np.all(stage2.feed["frame_mask"] == 1.0)
    assert stage2.feed["winner_onehot"].shape == (1, 2)


def test_ctc_wac_train_and_export_masked_dynamic_onnx(tmp_path: Path) -> None:
    has_onnx = importlib.util.find_spec("onnx") is not None
    has_ort = importlib.util.find_spec("onnxruntime") is not None
    if has_ort:
        import onnxruntime as ort
    else:
        ort = None
    keywords_path = _keyword_file(tmp_path)
    names = {
        "positive_train": (1, "train", 11),
        "negative_train": (0, "train", 12),
        "positive_dev": (1, "dev", 13),
        "negative_dev": (0, "dev", 14),
    }
    paths: dict[str, Path] = {}
    for name, (label, _split, seed) in names.items():
        path = tmp_path / f"{name}.npy"
        _write_bundle(path, label=label, keywords_path=keywords_path, seed=seed)
        paths[name] = path
    config = tmp_path / "pipeline.ini"
    config.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}
model_name = test_ctc_wac
sample_rate = 16000
seed = 7

[steps]
steps = {"train, export" if has_onnx else "train"}

[feature.positive_train]
output_file = {paths['positive_train']}
label = 1
split = train

[feature.negative_train]
output_file = {paths['negative_train']}
label = 0
split = train

[feature.positive_dev]
output_file = {paths['positive_dev']}
label = 1
split = dev

[feature.negative_dev]
output_file = {paths['negative_dev']}
label = 0
split = dev

[train]
structure = ctc_wac
keywords = {keywords_path}
train = feature.positive_train, feature.negative_train
dev = feature.positive_dev, feature.negative_dev
batch.feature.positive_train = 2
batch.feature.negative_train = 2
steps = 1
phase_step_ratios = [1.0]
phase_learning_rates = [0.001]
wac.frame_hidden = 8
wac.frame_layers = 1
wac.head_hidden = 8
wac.dropout = 0.0
max_negative_weight = 1
checkpoint_interval_steps = 1
keep_checkpoints = 1
log_interval_steps = 1
resume = yes
model_checkpoint_dir = ${{main:experiment_dir}}/checkpoints
output_model = ${{main:experiment_dir}}/model.pt
output_summary = ${{main:experiment_dir}}/training.json

[export]
input_model = ${{train:output_model}}
output_model = ${{main:experiment_dir}}/model.onnx
verify = yes
opset_version = 13
""",
        encoding="utf-8",
    )
    result = PipelineRunner(load_ini_config(config)).run()
    assert result["steps"]["train"]["result"]["structure"] == "ctc_wac"
    if not has_onnx or ort is None:
        return
    session = ort.InferenceSession(str(tmp_path / "experiment" / "model.onnx"), providers=["CPUExecutionProvider"])
    assert {item.name for item in session.get_inputs()} == {
        "encoder_features",
        "frame_mask",
        "top_score",
        "margin",
        "winner_onehot",
    }
    output = session.run(
        None,
        {
            "encoder_features": np.zeros((1, 4, 5), dtype=np.float32),
            "frame_mask": np.ones((1, 4), dtype=np.float32),
            "top_score": np.zeros((1, 1), dtype=np.float32),
            "margin": np.zeros((1, 1), dtype=np.float32),
            "winner_onehot": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
    )[0]
    assert output.shape == (1, 1)
    assert 0.0 <= float(output[0, 0]) <= 1.0


def test_stage1_report_summarizes_ragged_candidate_scores(tmp_path: Path) -> None:
    keywords_path = _keyword_file(tmp_path)
    positive = tmp_path / "positive.npy"
    negative = tmp_path / "negative.npy"
    _write_bundle(positive, label=1, keywords_path=keywords_path, seed=1)
    _write_bundle(negative, label=0, keywords_path=keywords_path, seed=2)
    config = tmp_path / "report.ini"
    config.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}

[steps]
steps = stage1_report

[feature.positive]
output_file = {positive}
label = 1
split = train

[feature.negative]
output_file = {negative}
label = 0
split = train

[stage1_report]
features = feature.positive, feature.negative
output_json = ${{main:experiment_dir}}/report.json
output_report = ${{main:experiment_dir}}/report.md
threshold_start = -5
threshold_stop = 0
threshold_step = 1
""",
        encoding="utf-8",
    )
    PipelineRunner(load_ini_config(config)).run()
    payload = json.loads((tmp_path / "experiment" / "report.json").read_text(encoding="utf-8"))
    assert payload["report_schema"] == 2
    positive_table = payload["blocks"][0]["threshold_table"]
    negative_table = payload["blocks"][1]["threshold_table"]
    positive_at_minus_two = next(item for item in positive_table if item["threshold"] == -2.0)
    negative_at_minus_two = next(item for item in negative_table if item["threshold"] == -2.0)
    assert positive_at_minus_two["keywords"]["wake_a"]["accuracy"] == 1.0
    assert positive_at_minus_two["keywords"]["wake_a"]["false_rejection_rate"] == 0.0
    assert positive_at_minus_two["keywords"]["wake_b"]["accuracy"] == 0.5
    assert positive_at_minus_two["keywords"]["wake_b"]["false_rejection_rate"] == 0.5
    assert negative_at_minus_two["keywords"]["wake_a"]["false_accepts_per_hour"] == 2.0
    assert negative_at_minus_two["keywords"]["wake_b"]["false_accepts_per_hour"] == 1.0
    assert negative_at_minus_two["keywords"]["wake_a"]["false_accept_rate"] == 0.5
    assert negative_at_minus_two["keywords"]["wake_b"]["false_accept_rate"] == 0.25
    markdown = (tmp_path / "experiment" / "report.md").read_text(encoding="utf-8")
    assert "Acc / FR" in markdown
    assert "FA/h" in markdown
    assert "FA rate" in markdown
    assert "quantile" not in markdown.lower()
