"""Independent ONNX test sets with per-set Markdown threshold reports."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..artifacts import normalise_manifest_inputs, read_jsonl, write_jsonl
from ..config import ConfigurationError, parse_json
from ..ctc_wac import (
    Stage1Contract,
    StreamingCtcStage1,
    audio_to_fbank,
    ctc_keyword_alignment_traces,
    ctc_keyword_vs_filler_score,
    load_audio,
    load_keywords,
    load_stage2_onnx,
    rank_keyword_scores,
    winner_onehot,
)
from ..legacy import get_legacy_module
from .common import integer, number, optional_integer, optional_number, require, stage_work_path


def _inputs(ctx: Any):
    from ..artifacts import parse_manifest_inputs

    return parse_manifest_inputs(ctx.config, ctx.step)


def _output_dir(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_dir", ctx.step))


def _output_report(ctx: Any) -> Path:
    configured = ctx.section.get("output_report")
    return ctx.config.resolve_path(configured) if configured else _output_dir(ctx) / "threshold_summary.md"


def _details_path(ctx: Any) -> Path:
    return _output_dir(ctx) / "eval_details.jsonl"


def _model(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "model", ctx.step))


def _model_dir(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "model_dir", ctx.step))


def _structure(ctx: Any) -> str:
    value = ctx.section.get("structure", "openwakeword").strip().lower()
    if value not in {"openwakeword", "ctc_wac"}:
        raise ConfigurationError(f"[{ctx.step}] structure must be openwakeword or ctc_wac")
    return value


def _stage1_model(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "stage1_model", ctx.step))


def _stage1_contract(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "stage1_contract", ctx.step))


def _keywords(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "keywords", ctx.step))


def _ctc_wac_stage1_gate_score(ctx: Any) -> str:
    """Return the score definition used by the user-facing Stage-1 gate."""

    value = ctx.section.get("stage1_gate_score", "normalized_ctc_score").strip().lower()
    allowed = {"normalized_ctc_score", "normalized_confidence"}
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ConfigurationError(f"[{ctx.step}] stage1_gate_score must be one of {choices}, got {value!r}")
    return value


def _ctc_proposal_score_floor(ctx: Any) -> float:
    """Return the internal CTC floor used only to avoid excessive beam work."""

    value = number(ctx.section, "ctc_proposal_score_floor", ctx.step, -8.0)
    if not math.isfinite(value):
        raise ConfigurationError(f"[{ctx.step}] ctc_proposal_score_floor must be finite")
    return value


def _ctc_competitor_options(ctx: Any) -> tuple[int, int | None]:
    beam_size = integer(ctx.section, "competitor_beam_size", ctx.step, 16)
    if beam_size < 2:
        raise ConfigurationError(f"[{ctx.step}] competitor_beam_size must be >= 2")
    token_prune = optional_integer(ctx.section, "competitor_token_prune", ctx.step)
    if token_prune is None:
        token_prune = 8
    if token_prune < 1:
        raise ConfigurationError(f"[{ctx.step}] competitor_token_prune must be >= 1")
    return beam_size, token_prune


def _ctc_max_search_frames(ctx: Any, contract: Stage1Contract) -> int | None:
    """Return an optional bounded CTC search horizon in encoder frames."""

    if "window_count" not in ctx.section:
        return None
    count = integer(ctx.section, "window_count", ctx.step)
    if count < 1:
        raise ConfigurationError(f"[{ctx.step}] window_count must be >= 1")
    main = ctx.config.section("main")
    default_seconds = number(
        main,
        "ctc_context_seconds",
        "main",
        number(main, "clip_seconds", "main", 2.0),
    )
    seconds = number(ctx.section, "window_seconds", ctx.step, default_seconds)
    if seconds <= 0:
        raise ConfigurationError(f"[{ctx.step}] window_seconds must be > 0")
    return max(1, int(math.ceil(seconds * count * 1000.0 / contract.encoder_frame_shift_ms)))


def _thresholds(ctx: Any) -> list[Decimal]:
    range_text = ctx.section.get("threshold_range")
    if range_text is not None:
        values = parse_json(range_text, f"[{ctx.step}] threshold_range", list)
        if len(values) != 2:
            raise ConfigurationError(f"[{ctx.step}] threshold_range must be [start, stop]")
        start_text, stop_text = (str(value) for value in values)
    else:
        start_text = require(ctx.section, "threshold_start", ctx.step)
        stop_text = require(ctx.section, "threshold_stop", ctx.step)
    try:
        start = Decimal(start_text)
        stop = Decimal(stop_text)
        increment = Decimal(require(ctx.section, "threshold_step", ctx.step))
    except InvalidOperation as exc:
        raise ConfigurationError(f"[{ctx.step}] threshold range and step must be decimal numbers") from exc
    if start < 0 or stop > 1 or start > stop or increment <= 0:
        raise ConfigurationError(
            f"[{ctx.step}] threshold range must satisfy 0 <= start <= stop <= 1 and threshold_step > 0"
        )
    thresholds: list[Decimal] = []
    current = start
    while current <= stop:
        thresholds.append(current)
        current += increment
    if thresholds[-1] != stop:
        raise ConfigurationError(f"[{ctx.step}] threshold_step must reach the end of threshold_range exactly")
    return thresholds


def validate(ctx: Any) -> None:
    expected_label = integer(ctx.section, "expected_label", ctx.step)
    if expected_label not in {0, 1}:
        raise ConfigurationError(f"[{ctx.step}] expected_label must be 0 or 1")
    _thresholds(ctx)
    for item in _inputs(ctx):
        if item.audio_base_dir and not item.audio_base_dir.is_dir():
            raise ConfigurationError(f"[{ctx.step}] audio_base_dir does not exist: {item.audio_base_dir}")
    if _structure(ctx) == "ctc_wac":
        for name, path in (
            ("model", _model(ctx)),
            ("stage1_model", _stage1_model(ctx)),
            ("stage1_contract", _stage1_contract(ctx)),
            ("keywords", _keywords(ctx)),
        ):
            if not path.is_file():
                raise ConfigurationError(f"[{ctx.step}] {name} does not exist: {path}")
        contract = Stage1Contract.from_json(_stage1_contract(ctx))
        load_keywords(_keywords(ctx))
        _ctc_wac_stage1_gate_score(ctx)
        _ctc_proposal_score_floor(ctx)
        _ctc_competitor_options(ctx)
        if number(ctx.config.section("main"), "sample_rate", "main", contract.sample_rate) != contract.sample_rate:
            raise ConfigurationError(
                f"[{ctx.step}] [main] sample_rate must match stage-1 contract sample_rate={contract.sample_rate}"
            )
        device = ctx.section.get("stage1_device", "cpu").lower()
        if device not in {"auto", "cpu", "gpu"}:
            raise ConfigurationError(f"[{ctx.step}] stage1_device must be auto, cpu, or gpu")
        if number(ctx.section, "debounce_seconds", ctx.step, 1.0) < 0:
            raise ConfigurationError(f"[{ctx.step}] debounce_seconds must be >= 0")
        if integer(ctx.section, "candidate_pre_margin_frames", ctx.step, 3) < 0:
            raise ConfigurationError(f"[{ctx.step}] candidate_pre_margin_frames must be >= 0")
        if integer(ctx.section, "candidate_post_margin_frames", ctx.step, 0) < 0:
            raise ConfigurationError(f"[{ctx.step}] candidate_post_margin_frames must be >= 0")
        _ctc_max_search_frames(ctx, contract)
        _output_report(ctx)
        return
    model_dir = _model_dir(ctx)
    if model_dir.exists() and not model_dir.is_dir():
        raise ConfigurationError(f"[{ctx.step}] model_dir is not a directory: {model_dir}")
    if number(ctx.section, "debounce_seconds", ctx.step, 1.0) < 0:
        raise ConfigurationError(f"[{ctx.step}] debounce_seconds must be >= 0")
    if integer(ctx.section, "chunk_size", ctx.step, 1280) < 1:
        raise ConfigurationError(f"[{ctx.step}] chunk_size must be >= 1")
    _output_report(ctx)


def input_paths(ctx: Any) -> list[Path]:
    if _structure(ctx) == "ctc_wac":
        return [
            *(item.path for item in _inputs(ctx)),
            _model(ctx),
            _stage1_model(ctx),
            _stage1_contract(ctx),
            _keywords(ctx),
        ]
    model_dir = _model_dir(ctx)
    return [
        *(item.path for item in _inputs(ctx)),
        _model(ctx),
        model_dir / "melspectrogram.onnx",
        model_dir / "embedding_model.onnx",
    ]


def output_paths(ctx: Any) -> list[Path]:
    return [_output_report(ctx), _details_path(ctx)]


def validate_outputs(ctx: Any) -> bool:
    report, details = output_paths(ctx)
    if not report.is_file() or not details.is_file():
        return False
    try:
        text = report.read_text(encoding="utf-8")
        return "| Threshold |" in text and "FA/hour" in text and "FA rate" in text and "FR rate" in text
    except OSError:
        return False


def _event_count(windows: list[dict[str, Any]], threshold: float, debounce_seconds: float) -> int:
    previous = -float("inf")
    count = 0
    for window in windows:
        try:
            score = float(window["score"])
            event_time = float(window["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if score >= threshold and event_time - previous >= debounce_seconds:
            count += 1
            previous = event_time
    return count


def _metric_rows(
    accumulators: list[dict[str, Any]],
    *,
    expected_label: int,
    evaluated: int,
    evaluated_seconds: float,
) -> list[dict[str, Any]]:
    hours = evaluated_seconds / 3600.0
    rows: list[dict[str, Any]] = []
    for values in accumulators:
        row = {
            "threshold": values["threshold"],
            "false_accept_events": values["false_accept_events"] if expected_label == 0 else None,
            "false_accept_clips": values["false_accept_clips"] if expected_label == 0 else None,
            "false_accepts_per_hour": (
                values["false_accept_events"] / hours if expected_label == 0 and hours else None
            ),
            "false_accept_rate": (
                values["false_accept_clips"] / evaluated if expected_label == 0 and evaluated else None
            ),
            "false_rejects": values["false_rejects"] if expected_label == 1 else None,
            "false_reject_rate": (
                values["false_rejects"] / evaluated if expected_label == 1 and evaluated else None
            ),
        }
        rows.append(row)
    return rows


def _format_metric(value: Any, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _markdown_report(
    *,
    step: str,
    model: Path,
    input_manifests: list[Path],
    expected_label: int,
    debounce_seconds: float,
    requested: int,
    evaluated: int,
    evaluated_seconds: float,
    errors: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    label = "positive" if expected_label == 1 else "negative"
    lines = [
        f"# Threshold Evaluation: {step}",
        "",
        f"- Model: `{model}`",
        f"- Expected label: `{label}`",
        f"- Input manifest(s): `{', '.join(str(path) for path in input_manifests)}`",
        f"- Debounce: `{debounce_seconds}` seconds",
        f"- Clips requested/evaluated: `{requested}` / `{evaluated}`",
        f"- Evaluated duration: `{evaluated_seconds / 3600.0:.6f}` hours",
        f"- Evaluation errors: `{len(errors)}`",
        "",
        "| Threshold | FA events | FA clips | FA/hour | FA rate | False rejects | FR rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        fa_events = "n/a" if row["false_accept_events"] is None else str(row["false_accept_events"])
        fa_clips = "n/a" if row["false_accept_clips"] is None else str(row["false_accept_clips"])
        false_rejects = "n/a" if row["false_rejects"] is None else str(row["false_rejects"])
        lines.append(
            f"| {row['threshold']:.6g} | {fa_events} | {fa_clips} | "
            f"{_format_metric(row['false_accepts_per_hour'])} | "
            f"{_format_metric(row['false_accept_rate'])} | {false_rejects} | "
            f"{_format_metric(row['false_reject_rate'])} |"
        )
    if errors:
        lines.extend(["", "## Evaluation errors", ""])
        for error in errors[:50]:
            lines.append(f"- index `{error.get('index')}` path `{error.get('path')}`: `{error.get('error')}`")
    return "\n".join(lines).rstrip() + "\n"


def _ctc_wac_stage2_shape(shapes: dict[str, tuple[int | None, ...]]) -> tuple[int, int]:
    encoder = shapes["encoder_features"]
    mask = shapes["frame_mask"]
    winner = shapes["winner_onehot"]
    if len(encoder) != 3 or encoder[2] is None:
        raise ConfigurationError(
            "CTC-WAC stage-2 ONNX must have encoder_features shaped [batch, frames, feature_dim]"
        )
    if len(mask) != 2 or (encoder[1] is not None and mask[1] is not None and encoder[1] != mask[1]):
        raise ConfigurationError("CTC-WAC stage-2 ONNX must have frame_mask shaped [batch, frames]")
    if len(winner) != 2 or winner[1] is None:
        raise ConfigurationError("CTC-WAC stage-2 ONNX must have winner_onehot shaped [batch, keyword_count]")
    return int(encoder[2]), int(winner[1])


def _ctc_wac_candidate_features(
    encoder: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    pre_margin_frames: int,
    post_margin_frames: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Crop the exact stage-1 candidate region and return its all-one mask."""

    crop_start = max(0, int(start_frame) - int(pre_margin_frames))
    crop_end = min(int(encoder.shape[0]), int(end_frame) + 1 + int(post_margin_frames))
    if crop_start >= crop_end:
        raise RuntimeError("CTC candidate crop is empty")
    values = np.asarray(encoder[crop_start:crop_end], dtype=np.float32)[np.newaxis, ...]
    mask = np.ones((1, values.shape[1]), dtype=np.float32)
    return values, mask, crop_start, crop_end


def _ctc_wac_record(
    *,
    record: dict[str, Any],
    stage1: StreamingCtcStage1,
    keywords: list[Any],
    stage2: Any,
    feature_dim: int,
    expected_label: int,
    candidate_pre_margin_frames: int = 3,
    candidate_post_margin_frames: int = 0,
    max_search_frames: int | None = None,
    stage1_gate_score: str = "normalized_ctc_score",
    ctc_proposal_score_floor: float = -8.0,
    competitor_beam_size: int = 16,
    competitor_token_prune: int | None = 8,
) -> dict[str, Any]:
    """Run both stages on one full audio record and retain candidate details."""

    path = Path(str(record["path"]))
    started = time.perf_counter()
    audio = load_audio(path, stage1.contract.sample_rate)
    duration = audio.size / float(stage1.contract.sample_rate)
    fbank = audio_to_fbank(audio, stage1.contract)
    encoder, ctc = stage1.infer_fbank(fbank)
    if encoder.shape[1] != feature_dim:
        raise RuntimeError(
            f"Stage-1 encoder has {encoder.shape[1]} dimensions but stage-2 expects {feature_dim}"
        )
    expected_vocab = getattr(stage1.contract, "vocab_size", None)
    if expected_vocab is not None and ctc.shape[1] != int(expected_vocab):
        raise RuntimeError(
            f"Stage-1 CTC vocabulary has {ctc.shape[1]} outputs but contract expects {expected_vocab}"
        )
    score_traces, start_traces, end_traces = ctc_keyword_alignment_traces(
        ctc,
        keywords,
        blank_id=stage1.contract.blank_id,
        max_search_frames=max_search_frames,
    )
    if score_traces.shape[1] == 0:
        raise RuntimeError("Stage-1 CTC scorer returned no keyword scores")
    if stage1_gate_score not in {"normalized_ctc_score", "normalized_confidence"}:
        raise ValueError(f"Unsupported Stage-1 gate score {stage1_gate_score!r}")
    if competitor_beam_size < 2:
        raise ValueError("competitor_beam_size must be >= 2")
    if competitor_token_prune is not None and competitor_token_prune < 1:
        raise ValueError("competitor_token_prune must be >= 1 when configured")
    thresholds = np.asarray([item.threshold for item in keywords], dtype=np.float32)
    previous_above = np.zeros(len(keywords), dtype=bool)
    # A confidence comparison is calculated only once after a candidate's
    # final non-blank token has been followed by a frame.  This avoids running
    # a beam search repeatedly while a token is still being extended.
    cached_keys: list[tuple[int, int, int] | None] = [None] * len(keywords)
    cached_comparisons: list[Any | None] = [None] * len(keywords)
    emitted_keys: list[tuple[int, int, int] | None] = [None] * len(keywords)
    candidates: list[dict[str, Any]] = []
    frame_count = int(score_traces.shape[0])
    # Add one synthetic end-of-stream iteration. During ordinary streaming we
    # wait for a frame after the final non-blank keyword token so that a token
    # extension is not emitted prematurely. A finite audio clip may end on
    # that token, however, so no following frame will ever arrive. Reusing the
    # last score row once at EOF flushes that valid candidate and matches the
    # offline feature-mining behavior.
    for stream_index in range(frame_count + 1):
        finalized_at_eof = stream_index == frame_count
        index = min(stream_index, frame_count - 1)
        row = score_traces[index:index + 1]
        top, margin, winner = rank_keyword_scores(row)
        winner_index = int(winner[0])
        start_frame = int(start_traces[index, winner_index])
        end_frame = int(end_traces[index, winner_index])
        comparison: Any | None = None
        if stage1_gate_score == "normalized_ctc_score":
            gate_value = float(top[0])
            above_by_keyword = row[0] >= thresholds
            candidate_key: tuple[int, int, int] | None = None
        else:
            above_by_keyword = np.zeros(len(keywords), dtype=bool)
            candidate_key = None
            gate_value = 0.0
            # The CTC score is a proposal floor only. It is not an operating
            # threshold: the configured per-keyword gate below is the same
            # normalized keyword-versus-filler confidence used during mining.
            if (
                float(top[0]) >= ctc_proposal_score_floor
                and start_frame >= 0
                and end_frame >= start_frame
                and end_frame < stream_index
            ):
                candidate_key = (winner_index, start_frame, end_frame)
                if cached_keys[winner_index] != candidate_key:
                    cached_comparisons[winner_index] = ctc_keyword_vs_filler_score(
                        ctc[start_frame:end_frame + 1],
                        keywords[winner_index].token_ids,
                        blank_id=stage1.contract.blank_id,
                        beam_size=competitor_beam_size,
                        token_prune=competitor_token_prune,
                    )
                    cached_keys[winner_index] = candidate_key
                comparison = cached_comparisons[winner_index]
                assert comparison is not None
                gate_value = float(comparison.normalized_confidence)
                above_by_keyword[winner_index] = gate_value >= float(thresholds[winner_index])
        # For a normalized-confidence gate, ``above_by_keyword`` stays false
        # until a completed candidate has an actual keyword-vs-filler score.
        # This matters for legitimate threshold 0.0 entries: an unavailable
        # confidence must never be treated as a passing zero.
        is_candidate = bool(above_by_keyword[winner_index] and not previous_above[winner_index])
        if candidate_key is not None and candidate_key == emitted_keys[winner_index]:
            is_candidate = False
        previous_above = above_by_keyword
        if not is_candidate:
            continue
        if start_frame < 0 or end_frame < start_frame:
            continue
        if candidate_key is not None:
            emitted_keys[winner_index] = candidate_key
        features, frame_mask, crop_start, crop_end = _ctc_wac_candidate_features(
            encoder,
            start_frame=start_frame,
            end_frame=end_frame,
            pre_margin_frames=candidate_pre_margin_frames,
            post_margin_frames=candidate_post_margin_frames,
        )
        onehot = winner_onehot(winner, len(keywords))
        probability = stage2.run(
            None,
            {
                "encoder_features": features,
                "frame_mask": frame_mask,
                "top_score": top.reshape(1, 1).astype(np.float32),
                "margin": margin.reshape(1, 1).astype(np.float32),
                "winner_onehot": onehot,
            },
        )[0]
        frame_shift = float(getattr(stage1.contract, "encoder_frame_shift_ms", 40.0)) / 1000.0
        candidates.append(
            {
                "keyword_id": keywords[winner_index].id,
                "trigger_frame": index,
                "finalized_at_eof": finalized_at_eof,
                "candidate_start_frame": start_frame,
                "candidate_end_frame": end_frame,
                "crop_start_frame": crop_start,
                "crop_end_frame": crop_end,
                "start_time": float(start_frame * frame_shift),
                "end_time": float((end_frame + 1) * frame_shift),
                "candidate_duration_frames": end_frame - start_frame + 1,
                "stage1_gate_score_name": stage1_gate_score,
                "stage1_gate_score": gate_value,
                "stage1_ctc_score": float(top[0]),
                "stage1_normalized_confidence": (
                    float(comparison.normalized_confidence) if comparison is not None else None
                ),
                "stage1_confidence": float(comparison.confidence) if comparison is not None else None,
                "stage1_keyword_score": float(comparison.keyword_score) if comparison is not None else None,
                "stage1_filler_score": float(comparison.filler_score) if comparison is not None else None,
                "stage1_filler_token_ids": list(comparison.filler_token_ids) if comparison is not None else None,
                "margin": float(margin[0]),
                "stage2_score": float(np.asarray(probability).reshape(-1)[0]),
                # Keep the former key for existing report readers. It is the
                # Stage-2 classifier probability, never a Stage-1 score.
                "score": float(np.asarray(probability).reshape(-1)[0]),
            }
        )
    elapsed = time.perf_counter() - started
    expected_keyword = record.get("keyword_id", record.get("wakeword_id"))
    return {
        "id": record.get("id"),
        "path": str(path),
        "expected_label": expected_label,
        "expected_keyword_id": expected_keyword,
        "text": record.get("text", "") if expected_label == 1 else "",
        "duration_seconds": float(duration),
        "stage1_candidate_count": len(candidates),
        "stage1_candidates": candidates,
        "best_window": max(candidates, key=lambda value: value["score"], default=None),
        "processing_seconds": elapsed,
        "real_time_factor": elapsed / duration if duration else None,
    }


def _ctc_wac_markdown_report(
    *,
    ctx: Any,
    expected_label: int,
    requested: int,
    evaluated: int,
    evaluated_seconds: float,
    errors: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    stage1_candidate_clips: int,
    stage1_candidate_events: int,
    inference_seconds: float,
    per_keyword: dict[str, int],
    max_search_frames: int | None,
    encoder_frame_shift_ms: float,
    stage1_gate_score: str,
    ctc_proposal_score_floor: float,
) -> str:
    base = _markdown_report(
        step=ctx.step,
        model=_model(ctx),
        input_manifests=[item.path for item in _inputs(ctx)],
        expected_label=expected_label,
        debounce_seconds=number(ctx.section, "debounce_seconds", ctx.step, 1.0),
        requested=requested,
        evaluated=evaluated,
        evaluated_seconds=evaluated_seconds,
        errors=errors,
        rows=rows,
    ).rstrip()
    hours = evaluated_seconds / 3600.0
    lines = [
        base,
        "",
        "## Stage-1 CTC gate",
        "",
        f"- Stage-1 ONNX: `{_stage1_model(ctx)}`",
        f"- Contract: `{_stage1_contract(ctx)}`",
        f"- User-facing Stage-1 gate: `{stage1_gate_score}`",
        (
            f"- Internal CTC proposal floor: `{ctc_proposal_score_floor:.6g}` "
            "(not an acceptance threshold)"
            if stage1_gate_score == "normalized_confidence"
            else "- Internal CTC proposal floor: `not used`"
        ),
        (
            f"- Rolling CTC search horizon: `{max_search_frames}` encoder frames "
            f"({max_search_frames * encoder_frame_shift_ms / 1000.0:.6g} seconds)"
            if max_search_frames is not None
            else "- Rolling CTC search horizon: `unbounded` (legacy configuration)"
        ),
        f"- Candidate clips: `{stage1_candidate_clips}` / `{evaluated}`",
        f"- Candidate events: `{stage1_candidate_events}`",
        f"- Candidate events/hour: `{stage1_candidate_events / hours if hours else 0.0:.6f}`",
        f"- Stage-1 candidate recall: `{stage1_candidate_clips / evaluated if expected_label == 1 and evaluated else 'n/a'}`",
        f"- End-to-end inference RTF: `{inference_seconds / evaluated_seconds if evaluated_seconds else 0.0:.6f}`",
        "",
        "## Candidates by stage-1 winner",
        "",
        "| Keyword | Candidate events |",
        "| --- | ---: |",
    ]
    for key in sorted(per_keyword):
        lines.append(f"| {key} | {per_keyword[key]} |")
    return "\n".join(lines).rstrip() + "\n"


def _run_ctc_wac(ctx: Any) -> dict[str, Any]:
    normalized = stage_work_path(ctx, "input.jsonl")
    expected_label = integer(ctx.section, "expected_label", ctx.step)
    normalise_manifest_inputs(_inputs(ctx), normalized, label=expected_label)
    records = read_jsonl(normalized)
    positive_limit = optional_integer(ctx.section, "limit_positive", ctx.step)
    if expected_label == 1 and positive_limit is not None:
        records = records[:positive_limit]
    contract = Stage1Contract.from_json(_stage1_contract(ctx))
    max_search_frames = _ctc_max_search_frames(ctx, contract)
    keywords = load_keywords(_keywords(ctx))
    stage1_gate_score = _ctc_wac_stage1_gate_score(ctx)
    ctc_proposal_score_floor = _ctc_proposal_score_floor(ctx)
    competitor_beam_size, competitor_token_prune = _ctc_competitor_options(ctx)
    stage1 = StreamingCtcStage1(
        _stage1_model(ctx),
        contract,
        device=ctx.section.get("stage1_device", "cpu").lower(),
    )
    stage2, shapes = load_stage2_onnx(_model(ctx))
    feature_dim, keyword_count = _ctc_wac_stage2_shape(shapes)
    if keyword_count != len(keywords):
        raise ConfigurationError(
            f"Stage-2 ONNX expects {keyword_count} keywords but {len(keywords)} are configured"
        )
    thresholds = [float(value) for value in _thresholds(ctx)]
    accumulators = [
        {"threshold": threshold, "false_accept_events": 0, "false_accept_clips": 0, "false_rejects": 0}
        for threshold in thresholds
    ]
    negative_limit_seconds = optional_number(ctx.section, "limit_negative_seconds", ctx.step)
    debounce = number(ctx.section, "debounce_seconds", ctx.step, 1.0)
    evaluated = 0
    evaluated_seconds = 0.0
    inference_seconds = 0.0
    stage1_candidate_clips = 0
    stage1_candidate_events = 0
    per_keyword = {item.id: 0 for item in keywords}
    errors: list[dict[str, Any]] = []
    report = _output_report(ctx)
    details = _details_path(ctx)
    report.parent.mkdir(parents=True, exist_ok=True)
    details.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=details.parent, delete=False)
    temporary_path = Path(temporary_handle.name)
    try:
        for index, record in enumerate(tqdm(records, desc=f"Evaluate {ctx.step}")):
            if expected_label == 0 and negative_limit_seconds is not None and evaluated_seconds >= negative_limit_seconds:
                break
            try:
                detail = _ctc_wac_record(
                    record=record,
                    stage1=stage1,
                    keywords=keywords,
                    stage2=stage2,
                    feature_dim=feature_dim,
                    expected_label=expected_label,
                    candidate_pre_margin_frames=integer(ctx.section, "candidate_pre_margin_frames", ctx.step, 3),
                    candidate_post_margin_frames=integer(ctx.section, "candidate_post_margin_frames", ctx.step, 0),
                    max_search_frames=max_search_frames,
                    stage1_gate_score=stage1_gate_score,
                    ctc_proposal_score_floor=ctc_proposal_score_floor,
                    competitor_beam_size=competitor_beam_size,
                    competitor_token_prune=competitor_token_prune,
                )
                temporary_handle.write(json.dumps(detail) + "\n")
                candidates = list(detail["stage1_candidates"])
                duration = float(detail["duration_seconds"])
                evaluated += 1
                evaluated_seconds += duration
                inference_seconds += float(detail["processing_seconds"])
                stage1_candidate_clips += int(bool(candidates))
                stage1_candidate_events += len(candidates)
                for candidate in candidates:
                    per_keyword[str(candidate["keyword_id"])] = per_keyword.get(str(candidate["keyword_id"]), 0) + 1
                for accumulator in accumulators:
                    event_count = _event_count(candidates, float(accumulator["threshold"]), debounce)
                    if expected_label == 1:
                        accumulator["false_rejects"] += int(event_count == 0)
                    else:
                        accumulator["false_accept_clips"] += int(event_count > 0)
                        accumulator["false_accept_events"] += event_count
            except Exception as exc:
                error = {
                    "set": ctx.step,
                    "index": index,
                    "id": record.get("id"),
                    "path": record.get("path"),
                    "error": repr(exc),
                }
                errors.append(error)
                temporary_handle.write(json.dumps(error, sort_keys=True) + "\n")
        temporary_handle.flush()
        os.fsync(temporary_handle.fileno())
        temporary_handle.close()
        os.replace(temporary_path, details)
    except Exception:
        temporary_handle.close()
        temporary_path.unlink(missing_ok=True)
        raise
    rows = _metric_rows(
        accumulators,
        expected_label=expected_label,
        evaluated=evaluated,
        evaluated_seconds=evaluated_seconds,
    )
    report.write_text(
        _ctc_wac_markdown_report(
            ctx=ctx,
            expected_label=expected_label,
            requested=len(records),
            evaluated=evaluated,
            evaluated_seconds=evaluated_seconds,
            errors=errors,
            rows=rows,
            stage1_candidate_clips=stage1_candidate_clips,
            stage1_candidate_events=stage1_candidate_events,
            inference_seconds=inference_seconds,
            per_keyword=per_keyword,
            max_search_frames=max_search_frames,
            encoder_frame_shift_ms=contract.encoder_frame_shift_ms,
            stage1_gate_score=stage1_gate_score,
            ctc_proposal_score_floor=ctc_proposal_score_floor,
        ),
        encoding="utf-8",
    )
    if not validate_outputs(ctx):
        raise RuntimeError(f"CTC-WAC testing output validation failed for {ctx.step}")
    return {
        "structure": "ctc_wac",
        "report": str(report),
        "details": str(details),
        "expected_label": expected_label,
        "threshold_count": len(rows),
        "clips_evaluated": evaluated,
        "evaluated_hours": evaluated_seconds / 3600.0,
        "stage1_candidate_clips": stage1_candidate_clips,
        "stage1_candidate_events": stage1_candidate_events,
        "stage1_candidate_events_per_hour": stage1_candidate_events / (evaluated_seconds / 3600.0)
        if evaluated_seconds
        else 0.0,
        "real_time_factor": inference_seconds / evaluated_seconds if evaluated_seconds else 0.0,
        "error_count": len(errors),
    }


def run(ctx: Any) -> dict[str, Any]:
    if _structure(ctx) == "ctc_wac":
        return _run_ctc_wac(ctx)
    normalized = stage_work_path(ctx, "input.jsonl")
    expected_label = integer(ctx.section, "expected_label", ctx.step)
    normalise_manifest_inputs(_inputs(ctx), normalized, label=expected_label)
    records = read_jsonl(normalized)
    positive_limit = optional_integer(ctx.section, "limit_positive", ctx.step)
    if expected_label == 1 and positive_limit is not None:
        records = records[:positive_limit]

    legacy = get_legacy_module()
    model_path = _model(ctx)
    model_kwargs = legacy.feature_model_paths(str(_model_dir(ctx)))
    model = legacy.openwakeword.Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        **model_kwargs,
    )
    thresholds = [float(value) for value in _thresholds(ctx)]
    accumulators = [
        {
            "threshold": threshold,
            "false_accept_events": 0,
            "false_accept_clips": 0,
            "false_rejects": 0,
        }
        for threshold in thresholds
    ]
    main = ctx.config.section("main")
    debounce = number(ctx.section, "debounce_seconds", ctx.step, 1.0)
    score_config = {
        "sample_rate": integer(main, "sample_rate", "main", 16000),
        "chunk_size": integer(ctx.section, "chunk_size", ctx.step, 1280),
        "positive_padding": integer(ctx.section, "positive_padding", ctx.step, 1),
        "negative_padding": integer(ctx.section, "negative_padding", ctx.step, 0),
        "model_window_seconds": number(
            ctx.section,
            "model_window_seconds",
            ctx.step,
            number(main, "clip_seconds", "main", 2.0),
        ),
        "record_window_scores": True,
    }
    negative_limit_seconds = optional_number(ctx.section, "limit_negative_seconds", ctx.step)
    evaluated = 0
    evaluated_seconds = 0.0
    errors: list[dict[str, Any]] = []
    report = _output_report(ctx)
    details = _details_path(ctx)
    report.parent.mkdir(parents=True, exist_ok=True)
    details.parent.mkdir(parents=True, exist_ok=True)
    for obsolete_name in ("eval_summary.json", "eval_abnormal.jsonl", "evaluation_config.json"):
        (_output_dir(ctx) / obsolete_name).unlink(missing_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=details.parent, delete=False)
    temporary_path = Path(temporary_handle.name)
    try:
        for index, record in enumerate(tqdm(records, desc=f"Evaluate {ctx.step}")):
            if expected_label == 0 and negative_limit_seconds is not None and evaluated_seconds >= negative_limit_seconds:
                break
            try:
                detail = legacy.evaluate_one_record_scores(
                    model,
                    model_path.stem,
                    record,
                    ctx.step,
                    index,
                    expected_label,
                    score_config,
                )
                # Preserve the deliberate field order in the detail record, including
                # utterance text immediately before the best scoring window.
                temporary_handle.write(json.dumps(detail, default=legacy.json_default) + "\n")
                windows = detail.get("sliding_windows", [])
                duration = float(detail.get("duration_seconds") or 0.0)
                evaluated += 1
                evaluated_seconds += duration
                for accumulator in accumulators:
                    event_count = _event_count(windows, float(accumulator["threshold"]), debounce)
                    if expected_label == 1:
                        accumulator["false_rejects"] += int(event_count == 0)
                    else:
                        accumulator["false_accept_clips"] += int(event_count > 0)
                        accumulator["false_accept_events"] += event_count
            except Exception as exc:
                error = {
                    "set": ctx.step,
                    "index": index,
                    "id": record.get("id"),
                    "path": record.get("path"),
                    "error": repr(exc),
                }
                errors.append(error)
                temporary_handle.write(json.dumps(error, sort_keys=True) + "\n")
        temporary_handle.flush()
        os.fsync(temporary_handle.fileno())
        temporary_handle.close()
        os.replace(temporary_path, details)
    except Exception:
        temporary_handle.close()
        temporary_path.unlink(missing_ok=True)
        raise

    metric_rows = _metric_rows(
        accumulators,
        expected_label=expected_label,
        evaluated=evaluated,
        evaluated_seconds=evaluated_seconds,
    )
    report.write_text(
        _markdown_report(
            step=ctx.step,
            model=model_path,
            input_manifests=[item.path for item in _inputs(ctx)],
            expected_label=expected_label,
            debounce_seconds=debounce,
            requested=len(records),
            evaluated=evaluated,
            evaluated_seconds=evaluated_seconds,
            errors=errors,
            rows=metric_rows,
        ),
        encoding="utf-8",
    )
    if not validate_outputs(ctx):
        raise RuntimeError(f"Testing output validation failed for {ctx.step}")
    return {
        "report": str(report),
        "details": str(details),
        "expected_label": expected_label,
        "threshold_count": len(metric_rows),
        "clips_evaluated": evaluated,
        "evaluated_hours": evaluated_seconds / 3600.0,
        "error_count": len(errors),
    }


def prepare_slurm_shards(ctx: Any, work_dir: Path, task_count: int) -> list[dict[str, Any]]:
    normalized = work_dir / "input.jsonl"
    expected_label = integer(ctx.section, "expected_label", ctx.step)
    normalise_manifest_inputs(_inputs(ctx), normalized, label=expected_label)
    records = read_jsonl(normalized)
    positive_limit = optional_integer(ctx.section, "limit_positive", ctx.step)
    if expected_label == 1 and positive_limit is not None:
        records = records[:positive_limit]
        write_jsonl(normalized, records)
    actual_count = min(task_count, len(records))
    if actual_count < 1:
        raise RuntimeError(f"Testing stage {ctx.step} has no records to shard")
    base, extra = divmod(len(records), actual_count)
    result: list[dict[str, Any]] = []
    start = 0
    for task_id in range(actual_count):
        stop = start + base + (1 if task_id < extra else 0)
        shard_dir = work_dir / "shards" / f"{task_id:05d}"
        input_manifest = shard_dir / "input.jsonl"
        output_dir = shard_dir / "output"
        write_jsonl(input_manifest, records[start:stop])
        result.append(
            {
                "id": task_id,
                "start": start,
                "stop": stop,
                "count": stop - start,
                "input_manifest": str(input_manifest),
                "output_dir": str(output_dir),
                "output_report": str(output_dir / "threshold_summary.md"),
                "details": str(output_dir / "eval_details.jsonl"),
                "normalized_manifest": str(normalized),
            }
        )
        start = stop
    return result


def _slurm_task_context(ctx: Any, task: dict[str, Any]) -> Any:
    """Create a one-shard context without changing the user's INI on disk."""

    section = dict(ctx.section)
    section["input_jsonl"] = str(task["input_manifest"])
    section.pop("audio_base_dir", None)
    section.pop("limit_positive", None)
    section.pop("limit_negative_seconds", None)
    section["output_dir"] = str(task["output_dir"])
    section["output_report"] = str(task["output_report"])
    parser = ctx.config.parser
    parser.set(ctx.step, "input_jsonl", str(task["input_manifest"]))
    parser.remove_option(ctx.step, "audio_base_dir")
    parser.remove_option(ctx.step, "limit_positive")
    parser.remove_option(ctx.step, "limit_negative_seconds")
    return replace(
        ctx,
        section=section,
        work_dir=Path(str(task["output_dir"])).parent,
        force=True,
    )


def run_slurm_shard(ctx: Any, task: dict[str, Any]) -> dict[str, Any]:
    result = run(_slurm_task_context(ctx, task))
    return {**result, "task_id": int(task["id"]), "details": str(task["details"])}


def validate_slurm_shard(ctx: Any, task: dict[str, Any]) -> bool:
    report = Path(str(task["output_report"])).resolve()
    details = Path(str(task["details"])).resolve()
    if not report.is_file() or not details.is_file():
        return False
    try:
        return "| Threshold |" in report.read_text(encoding="utf-8")
    except OSError:
        return False


def _slurm_entries(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for task in tasks:
        path = Path(str(task["details"])).resolve()
        for local_index, entry in enumerate(read_jsonl(path, allow_empty=True)):
            updated = dict(entry)
            updated["index"] = int(task["start"]) + int(updated.get("index", local_index))
            entries.append(updated)
    return sorted(entries, key=lambda value: int(value.get("index", -1)))


def merge_slurm_shards(ctx: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    expected_label = integer(ctx.section, "expected_label", ctx.step)
    entries = _slurm_entries(tasks)
    requested = sum(int(task["count"]) for task in tasks)
    negative_limit_seconds = optional_number(ctx.section, "limit_negative_seconds", ctx.step)
    debounce = number(ctx.section, "debounce_seconds", ctx.step, 1.0)
    thresholds = [float(value) for value in _thresholds(ctx)]
    accumulators = [
        {"threshold": threshold, "false_accept_events": 0, "false_accept_clips": 0, "false_rejects": 0}
        for threshold in thresholds
    ]
    evaluated = 0
    evaluated_seconds = 0.0
    errors: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    is_ctc_wac = _structure(ctx) == "ctc_wac"
    inference_seconds = 0.0
    stage1_candidate_clips = 0
    stage1_candidate_events = 0
    per_keyword: dict[str, int] = {}
    if is_ctc_wac:
        per_keyword = {item.id: 0 for item in load_keywords(_keywords(ctx))}

    for entry in entries:
        if expected_label == 0 and negative_limit_seconds is not None and evaluated_seconds >= negative_limit_seconds:
            break
        selected.append(entry)
        if "error" in entry and "duration_seconds" not in entry:
            errors.append(entry)
            continue
        try:
            duration = float(entry["duration_seconds"])
            evaluated += 1
            evaluated_seconds += duration
            if is_ctc_wac:
                candidates = list(entry.get("stage1_candidates", []))
                inference_seconds += float(entry.get("processing_seconds", 0.0))
                stage1_candidate_clips += int(bool(candidates))
                stage1_candidate_events += len(candidates)
                for candidate in candidates:
                    key = str(candidate["keyword_id"])
                    per_keyword[key] = per_keyword.get(key, 0) + 1
                windows = candidates
            else:
                windows = list(entry.get("sliding_windows", []))
            for accumulator in accumulators:
                event_count = _event_count(windows, float(accumulator["threshold"]), debounce)
                if expected_label == 1:
                    accumulator["false_rejects"] += int(event_count == 0)
                else:
                    accumulator["false_accept_clips"] += int(event_count > 0)
                    accumulator["false_accept_events"] += event_count
        except Exception as exc:
            error = {
                "set": ctx.step,
                "index": entry.get("index"),
                "path": entry.get("path"),
                "error": repr(exc),
            }
            errors.append(error)

    details = _details_path(ctx)
    report = _output_report(ctx)
    write_jsonl(details, selected)
    rows = _metric_rows(
        accumulators,
        expected_label=expected_label,
        evaluated=evaluated,
        evaluated_seconds=evaluated_seconds,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    if is_ctc_wac:
        report.write_text(
            _ctc_wac_markdown_report(
                ctx=ctx,
                expected_label=expected_label,
                requested=requested,
                evaluated=evaluated,
                evaluated_seconds=evaluated_seconds,
                errors=errors,
                rows=rows,
                stage1_candidate_clips=stage1_candidate_clips,
                stage1_candidate_events=stage1_candidate_events,
                inference_seconds=inference_seconds,
                per_keyword=per_keyword,
                max_search_frames=_ctc_max_search_frames(ctx, Stage1Contract.from_json(_stage1_contract(ctx))),
                encoder_frame_shift_ms=Stage1Contract.from_json(_stage1_contract(ctx)).encoder_frame_shift_ms,
                stage1_gate_score=_ctc_wac_stage1_gate_score(ctx),
                ctc_proposal_score_floor=_ctc_proposal_score_floor(ctx),
            ),
            encoding="utf-8",
        )
    else:
        report.write_text(
            _markdown_report(
                step=ctx.step,
                model=_model(ctx),
                input_manifests=[item.path for item in _inputs(ctx)],
                expected_label=expected_label,
                debounce_seconds=debounce,
                requested=requested,
                evaluated=evaluated,
                evaluated_seconds=evaluated_seconds,
                errors=errors,
                rows=rows,
            ),
            encoding="utf-8",
        )
    if not validate_outputs(ctx):
        raise RuntimeError(f"Testing merge validation failed for {ctx.step}")
    result = {
        "report": str(report),
        "details": str(details),
        "expected_label": expected_label,
        "threshold_count": len(rows),
        "clips_evaluated": evaluated,
        "evaluated_hours": evaluated_seconds / 3600.0,
        "error_count": len(errors),
    }
    if is_ctc_wac:
        result.update(
            {
                "structure": "ctc_wac",
                "stage1_candidate_clips": stage1_candidate_clips,
                "stage1_candidate_events": stage1_candidate_events,
                "stage1_candidate_events_per_hour": stage1_candidate_events / (evaluated_seconds / 3600.0)
                if evaluated_seconds
                else 0.0,
                "real_time_factor": inference_seconds / evaluated_seconds if evaluated_seconds else 0.0,
            }
        )
    return result


def cleanup_slurm_shards(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        Path(str(task["details"])).unlink(missing_ok=True)
        Path(str(task["output_report"])).unlink(missing_ok=True)
