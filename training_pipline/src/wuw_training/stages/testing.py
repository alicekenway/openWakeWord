"""Independent ONNX test sets with per-set Markdown threshold reports."""

from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..artifacts import normalise_manifest_inputs, read_jsonl
from ..config import ConfigurationError, parse_json
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
    model_dir = _model_dir(ctx)
    if model_dir.exists() and not model_dir.is_dir():
        raise ConfigurationError(f"[{ctx.step}] model_dir is not a directory: {model_dir}")
    if number(ctx.section, "debounce_seconds", ctx.step, 1.0) < 0:
        raise ConfigurationError(f"[{ctx.step}] debounce_seconds must be >= 0")
    if integer(ctx.section, "chunk_size", ctx.step, 1280) < 1:
        raise ConfigurationError(f"[{ctx.step}] chunk_size must be >= 1")
    _output_report(ctx)


def input_paths(ctx: Any) -> list[Path]:
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
        return "| Threshold |" in text and "FA/hour" in text and "FR rate" in text
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
        "| Threshold | FA events | FA clips | FA/hour | False rejects | FR rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        fa_events = "n/a" if row["false_accept_events"] is None else str(row["false_accept_events"])
        fa_clips = "n/a" if row["false_accept_clips"] is None else str(row["false_accept_clips"])
        false_rejects = "n/a" if row["false_rejects"] is None else str(row["false_rejects"])
        lines.append(
            f"| {row['threshold']:.6g} | {fa_events} | {fa_clips} | "
            f"{_format_metric(row['false_accepts_per_hour'])} | {false_rejects} | "
            f"{_format_metric(row['false_reject_rate'])} |"
        )
    if errors:
        lines.extend(["", "## Evaluation errors", ""])
        for error in errors[:50]:
            lines.append(f"- index `{error.get('index')}` path `{error.get('path')}`: `{error.get('error')}`")
    return "\n".join(lines).rstrip() + "\n"


def run(ctx: Any) -> dict[str, Any]:
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
                temporary_handle.write(json.dumps(detail, sort_keys=True, default=legacy.json_default) + "\n")
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
