"""Threshold-sweep reporting built from per-set testing details."""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..artifacts import read_json, read_jsonl, write_json
from ..config import ConfigurationError, parse_json
from .common import csv_option, number, require


def _test_steps(ctx: Any) -> list[str]:
    values = csv_option(ctx.section, "tests", ctx.step)
    invalid = [value for value in values if not value.startswith("testing.")]
    if invalid:
        raise ConfigurationError(f"[{ctx.step}] tests must name testing.* blocks: {', '.join(invalid)}")
    return values


def _test_paths(ctx: Any, test_step: str) -> tuple[Path, Path]:
    section = ctx.config.section(test_step)
    output_dir_value = section.get("output_dir")
    if not output_dir_value:
        raise ConfigurationError(f"Missing required option [{test_step}] output_dir")
    root = ctx.config.resolve_path(output_dir_value)
    return root / "eval_summary.json", root / "eval_details.jsonl"


def _thresholds(ctx: Any) -> list[Decimal]:
    range_text = ctx.section.get("threshold_range")
    try:
        if range_text is not None:
            values = parse_json(range_text, f"[{ctx.step}] threshold_range", list)
            if len(values) != 2:
                raise ConfigurationError(f"[{ctx.step}] threshold_range must be [start, stop]")
            start = Decimal(str(values[0]))
            stop = Decimal(str(values[1]))
        else:
            start = Decimal(require(ctx.section, "threshold_start", ctx.step))
            stop = Decimal(require(ctx.section, "threshold_stop", ctx.step))
        increment = Decimal(require(ctx.section, "threshold_step", ctx.step))
    except InvalidOperation as exc:
        raise ConfigurationError(f"[{ctx.step}] threshold values must be decimal numbers") from exc
    if start < 0 or stop > 1 or start > stop or increment <= 0:
        raise ConfigurationError(f"[{ctx.step}] threshold range must satisfy 0 <= start <= stop <= 1 and step > 0")
    values: list[Decimal] = []
    current = start
    # Decimal avoids omitting the upper bound due to binary float rounding.
    while current <= stop:
        values.append(current)
        current += increment
    if values[-1] != stop:
        raise ConfigurationError(f"[{ctx.step}] threshold_step must reach threshold_stop exactly")
    return values


def _output_json(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_json", ctx.step))


def _output_report(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_report", ctx.step))


def _configured_keyword_thresholds(ctx: Any) -> tuple[Path, dict[str, float]] | None:
    raw_path = ctx.section.get("keyword_thresholds")
    if raw_path is None:
        return None
    path = ctx.config.resolve_path(raw_path)
    if not path.is_file():
        raise ConfigurationError(f"[{ctx.step}] keyword_thresholds does not exist: {path}")
    payload = read_json(path)
    values = payload.get("keywords") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ConfigurationError(
            f"[{ctx.step}] keyword_thresholds must contain a non-empty keywords list"
        )
    thresholds: dict[str, float] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ConfigurationError(
                f"[{ctx.step}] keyword_thresholds keywords[{index}] needs an id"
            )
        try:
            threshold = float(value["stage2_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"[{ctx.step}] keyword_thresholds keywords[{index}] needs stage2_threshold"
            ) from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ConfigurationError(
                f"[{ctx.step}] keyword_thresholds keywords[{index}].stage2_threshold must be in [0, 1]"
            )
        keyword_id = str(value["id"])
        if keyword_id in thresholds:
            raise ConfigurationError(
                f"[{ctx.step}] duplicate keyword threshold id {keyword_id!r}"
            )
        thresholds[keyword_id] = threshold
    return path, thresholds


def validate(ctx: Any) -> None:
    _thresholds(ctx)
    if number(ctx.section, "debounce_seconds", ctx.step, 1.0) < 0:
        raise ConfigurationError(f"[{ctx.step}] debounce_seconds must be >= 0")
    for test_step in _test_steps(ctx):
        test_section = ctx.config.section(test_step)
        try:
            expected = int(test_section.get("expected_label", ""))
        except ValueError as exc:
            raise ConfigurationError(f"[{test_step}] expected_label must be 0 or 1") from exc
        if expected not in {0, 1}:
            raise ConfigurationError(f"[{test_step}] expected_label must be 0 or 1")
    _output_json(ctx)
    _output_report(ctx)
    _configured_keyword_thresholds(ctx)


def input_paths(ctx: Any) -> list[Path]:
    paths: list[Path] = []
    for step in _test_steps(ctx):
        _summary, details = _test_paths(ctx, step)
        paths.append(details)
    configured = _configured_keyword_thresholds(ctx)
    if configured is not None:
        paths.append(configured[0])
    return paths


def output_paths(ctx: Any) -> list[Path]:
    return [_output_json(ctx), _output_report(ctx)]


def validate_outputs(ctx: Any) -> bool:
    output, report = output_paths(ctx)
    if not output.is_file() or not report.is_file():
        return False
    try:
        value = read_json(output)
        return isinstance(value.get("thresholds"), list) and bool(value["thresholds"])
    except Exception:
        return False


def _vad_passed(window: dict[str, Any]) -> bool:
    """Treat legacy and VAD-disabled candidates as eligible."""

    return bool(window.get("vad_passed", True))


def _events(windows: list[dict[str, Any]], threshold: float, debounce_seconds: float) -> int:
    return _events_matching(
        windows,
        lambda window: _vad_passed(window) and float(window["score"]) >= threshold,
        debounce_seconds,
    )


def _events_matching(
    windows: list[dict[str, Any]],
    accepted: Any,
    debounce_seconds: float,
) -> int:
    def event_time(value: dict[str, Any]) -> float:
        try:
            return float(value["end_time"])
        except (KeyError, TypeError, ValueError):
            return float("inf")

    previous = -float("inf")
    count = 0
    for window in sorted(windows, key=event_time):
        try:
            event_time = float(window["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            passes = bool(accepted(window))
        except (KeyError, TypeError, ValueError):
            continue
        if passes and event_time - previous >= debounce_seconds:
            count += 1
            previous = event_time
    return count


def _record_windows(record: dict[str, Any]) -> tuple[list[dict[str, Any]], str] | None:
    candidates = record.get("stage1_candidates")
    if isinstance(candidates, list):
        return candidates, "ctc_wac"
    windows = record.get("sliding_windows")
    if isinstance(windows, list):
        return windows, "openwakeword"
    return None


def _inference_performance_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    for record in records:
        values = record.get("inference_performance")
        if isinstance(values, list):
            measurements.extend(value for value in values if isinstance(value, dict))

    def summary(field: str) -> dict[str, float | None]:
        samples: list[float] = []
        for measurement in measurements:
            try:
                value = float(measurement[field])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                samples.append(value)
        return {
            "mean": sum(samples) / len(samples) if samples else None,
            "max": max(samples) if samples else None,
        }

    return {
        "inference_count": len(measurements),
        "real_time_factor": summary("real_time_factor"),
        "cpu_utilization_percent": summary("cpu_utilization_percent"),
        "peak_rss_mb": summary("peak_rss_mb"),
    }


def _crop_counts(
    record: dict[str, Any],
    windows: list[dict[str, Any]],
    kind: str,
    threshold: float,
) -> tuple[int, int]:
    if kind == "ctc_wac":
        evaluated = max(0, int(record.get("audio_window_count", 1)))
        accepted_indices: set[int] = set()
        for window in windows:
            try:
                if float(window["score"]) < threshold:
                    continue
                if not _vad_passed(window):
                    continue
                accepted_indices.add(int(window.get("audio_window_index") or 0))
            except (KeyError, TypeError, ValueError):
                continue
        return evaluated, min(evaluated, len(accepted_indices))
    evaluated = len(windows)
    accepted = 0
    for window in windows:
        try:
            accepted += int(_vad_passed(window) and float(window["score"]) >= threshold)
        except (KeyError, TypeError, ValueError):
            continue
    return evaluated, accepted


def _keyword_threshold_metrics(
    records: list[dict[str, Any]],
    expected_label: int,
    debounce_seconds: float,
    errors: int,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    def accepted(window: dict[str, Any]) -> bool:
        threshold = (
            thresholds.get(str(window.get("keyword_id", "")))
            if thresholds is not None
            else window.get("stage2_threshold")
        )
        return (
            threshold is not None
            and _vad_passed(window)
            and float(window["score"]) >= float(threshold)
        )

    evaluated = 0
    crops_evaluated = 0
    seconds = 0.0
    false_rejects = 0
    false_accept_clips = 0
    false_accept_crops = 0
    false_accept_events = 0
    for record in records:
        record_windows = _record_windows(record)
        if record_windows is None:
            continue
        windows, kind = record_windows
        events = _events_matching(windows, accepted, debounce_seconds)
        if kind == "ctc_wac":
            crop_count = max(0, int(record.get("audio_window_count", 1)))
            accepted_indices = {
                int(window.get("audio_window_index") or 0)
                for window in windows
                if accepted(window)
            }
            accepted_crop_count = min(crop_count, len(accepted_indices))
        else:
            crop_count = len(windows)
            accepted_crop_count = sum(int(accepted(window)) for window in windows)
        evaluated += 1
        crops_evaluated += crop_count
        seconds += float(record.get("duration_seconds") or 0.0)
        if expected_label == 1:
            false_rejects += int(events == 0)
        else:
            false_accept_clips += int(events > 0)
            false_accept_crops += accepted_crop_count
            false_accept_events += events
    result: dict[str, Any] = {
        "clips_evaluated": evaluated,
        "crops_evaluated": crops_evaluated,
        "evaluated_seconds": round(seconds, 6),
        "evaluated_hours": round(seconds / 3600.0, 6),
        "error_count": errors,
    }
    if expected_label == 1:
        result.update(
            {
                "false_rejects": false_rejects,
                "false_reject_rate": (false_rejects / evaluated) if evaluated else None,
                "recall": ((evaluated - false_rejects) / evaluated) if evaluated else None,
            }
        )
    else:
        result.update(
            {
                "false_accept_clips": false_accept_clips,
                "false_accept_crops": false_accept_crops,
                "false_accept_events": false_accept_events,
                "false_accepts_per_hour": (
                    false_accept_events / (seconds / 3600.0) if seconds else None
                ),
                "false_accept_rate": (
                    false_accept_crops / crops_evaluated if crops_evaluated else None
                ),
                "false_accept_rate_denominator": "evaluated_crops",
            }
        )
    return result


def _metrics(records: list[dict[str, Any]], expected_label: int, threshold: float, debounce_seconds: float, errors: int) -> dict[str, Any]:
    evaluated = 0
    crops_evaluated = 0
    seconds = 0.0
    false_rejects = 0
    false_accept_clips = 0
    false_accept_crops = 0
    false_accept_events = 0
    for record in records:
        record_windows = _record_windows(record)
        if record_windows is None:
            continue
        windows, kind = record_windows
        events = _events(windows, threshold, debounce_seconds)
        record_crop_count, accepted_crop_count = _crop_counts(record, windows, kind, threshold)
        evaluated += 1
        crops_evaluated += record_crop_count
        seconds += float(record.get("duration_seconds") or 0.0)
        if expected_label == 1:
            false_rejects += int(events == 0)
        else:
            false_accept_clips += int(events > 0)
            false_accept_crops += accepted_crop_count
            false_accept_events += events
    result: dict[str, Any] = {
        "clips_evaluated": evaluated,
        "crops_evaluated": crops_evaluated,
        "evaluated_seconds": round(seconds, 6),
        "evaluated_hours": round(seconds / 3600.0, 6),
        "error_count": errors,
    }
    if expected_label == 1:
        result.update(
            {
                "false_rejects": false_rejects,
                "false_reject_rate": (false_rejects / evaluated) if evaluated else None,
                "recall": ((evaluated - false_rejects) / evaluated) if evaluated else None,
            }
        )
    else:
        result.update(
            {
                "false_accept_clips": false_accept_clips,
                "false_accept_crops": false_accept_crops,
                "false_accept_events": false_accept_events,
                "false_accepts_per_hour": (false_accept_events / (seconds / 3600.0)) if seconds else None,
                "false_accept_rate": (
                    false_accept_crops / crops_evaluated if crops_evaluated else None
                ),
                "false_accept_rate_denominator": "evaluated_crops",
            }
        )
    return result


def _combined_negative(values: list[dict[str, Any]]) -> dict[str, Any]:
    clips = sum(int(value["clips_evaluated"]) for value in values)
    crops = sum(int(value["crops_evaluated"]) for value in values)
    seconds = sum(float(value["evaluated_seconds"]) for value in values)
    accepts = sum(int(value["false_accept_events"]) for value in values)
    accept_clips = sum(int(value["false_accept_clips"]) for value in values)
    accept_crops = sum(int(value["false_accept_crops"]) for value in values)
    return {
        "clips_evaluated": clips,
        "crops_evaluated": crops,
        "evaluated_seconds": round(seconds, 6),
        "evaluated_hours": round(seconds / 3600.0, 6),
        "false_accept_clips": accept_clips,
        "false_accept_crops": accept_crops,
        "false_accept_events": accepts,
        "false_accepts_per_hour": accepts / (seconds / 3600.0) if seconds else None,
        "false_accept_rate": accept_crops / crops if crops else None,
        "false_accept_rate_denominator": "evaluated_crops",
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Wake-Word Threshold Summary",
        "",
        f"- Debounce: `{payload['debounce_seconds']}` seconds",
        f"- Test blocks: `{', '.join(payload['tests'])}`",
        "- FA rate: false-accepted inference crops / evaluated inference crops",
        "- FA/hour: debounced false-accept events / evaluated audio hours",
    ]

    def metric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.6f}"

    selected = payload.get("keyword_threshold_operating_point")
    if isinstance(selected, dict):
        lines.extend(
            [
                "",
                "## Per-keyword Stage-2 operating point",
                "",
                f"- Thresholds: `{json.dumps(selected['thresholds'], sort_keys=True)}`",
                "",
            ]
        )
        for test_name, values in selected["sets"].items():
            if "recall" in values:
                lines.append(
                    f"- `{test_name}`: FR `{metric(values['false_reject_rate'])}`, "
                    f"recall `{metric(values['recall'])}`"
                )
            else:
                lines.append(
                    f"- `{test_name}`: FA/hour `{metric(values['false_accepts_per_hour'])}`, "
                    f"FA rate `{metric(values['false_accept_rate'])}`"
                )

    for test_name in payload["tests"]:
        values = [
            (float(item["threshold"]), item["sets"][test_name])
            for item in payload["thresholds"]
            if test_name in item["sets"]
        ]
        lines.extend(["", f"## {test_name}", ""])
        if not values:
            lines.append("No evaluated threshold rows.")
            continue
        first = values[0][1]
        lines.extend(
            [
                f"- Evaluated source files: `{first['clips_evaluated']}`",
                f"- Evaluated crops: `{first['crops_evaluated']}`",
                f"- Evaluated duration: `{first['evaluated_hours']:.6f}` hours",
                f"- Evaluation errors: `{first['error_count']}`",
                "",
            ]
        )
        performance = payload.get("performance_by_test", {}).get(test_name, {})
        vad = payload.get("vad_by_test", {}).get(test_name, {})
        if vad.get("enabled"):
            lines.extend(
                [
                    "### VAD gate",
                    "",
                    f"- Threshold: `{metric(vad.get('threshold'))}`",
                    f"- Padding: `{metric(vad.get('padding_ms'))}` ms",
                    f"- Passed candidates: `{vad.get('candidate_pass_count', 0)}`",
                    f"- Rejected candidates: `{vad.get('candidate_reject_count', 0)}`",
                    "",
                ]
            )
        if int(performance.get("inference_count", 0)) > 0:
            lines.extend(
                [
                    "### Inference performance",
                    "",
                    f"- Measured inferences: `{performance['inference_count']}`",
                    "",
                    "| Metric | Mean | Max |",
                    "| --- | ---: | ---: |",
                    f"| Real-time factor | {metric(performance['real_time_factor']['mean'])} | "
                    f"{metric(performance['real_time_factor']['max'])} |",
                    f"| Peak RSS (MiB) | {metric(performance['peak_rss_mb']['mean'])} | "
                    f"{metric(performance['peak_rss_mb']['max'])} |",
                    f"| CPU utilization (%) | "
                    f"{metric(performance['cpu_utilization_percent']['mean'])} | "
                    f"{metric(performance['cpu_utilization_percent']['max'])} |",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "### Inference performance",
                    "",
                    "No inference performance measurements are present in this test artifact.",
                    "",
                ]
            )
        if "recall" in first:
            lines.extend(
                [
                    "| Threshold | Accuracy / recall | False rejects | FR rate |",
                    "| ---: | ---: | ---: | ---: |",
                ]
            )
            for threshold, values_at_threshold in values:
                lines.append(
                    f"| {threshold:.6g} | {metric(values_at_threshold['recall'])} | "
                    f"{values_at_threshold['false_rejects']} | "
                    f"{metric(values_at_threshold['false_reject_rate'])} |"
                )
        else:
            lines.extend(
                [
                    "| Threshold | FA events | FA source files | FA crops | Evaluated crops | FA/hour | FA rate |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for threshold, values_at_threshold in values:
                lines.append(
                    f"| {threshold:.6g} | {values_at_threshold['false_accept_events']} | "
                    f"{values_at_threshold['false_accept_clips']} | "
                    f"{values_at_threshold['false_accept_crops']} | "
                    f"{values_at_threshold['crops_evaluated']} | "
                    f"{metric(values_at_threshold['false_accepts_per_hour'])} | "
                    f"{metric(values_at_threshold['false_accept_rate'])} |"
                )
    return "\n".join(lines).rstrip() + "\n"


def run(ctx: Any) -> dict[str, Any]:
    debounce = number(ctx.section, "debounce_seconds", ctx.step, 1.0)
    loaded: dict[str, tuple[int, list[dict[str, Any]], int]] = {}
    test_summaries: dict[str, dict[str, Any]] = {}
    for test_step in _test_steps(ctx):
        test_section = ctx.config.section(test_step)
        expected_label = int(test_section["expected_label"])
        summary_path, details_path = _test_paths(ctx, test_step)
        records = read_jsonl(details_path, allow_empty=True)
        if summary_path.is_file():
            test_summary = read_json(summary_path)
        else:
            test_summary = {}
        test_summaries[test_step] = test_summary
        if "error_count" in test_summary:
            error_count = int(test_summary.get("error_count", 0))
        else:
            # Read old inference artifacts once so users can generate a new
            # threshold report without repeating an expensive test run.
            legacy_error_count = max(
                (int(metrics.get("error_count", 0)) for metrics in test_summary.get("sets", {}).values()),
                default=0,
            )
            detail_error_count = sum(
                "error" in record and _record_windows(record) is None
                for record in records
            )
            error_count = max(legacy_error_count, detail_error_count)
        loaded[test_step] = (expected_label, records, error_count)

    threshold_rows: list[dict[str, Any]] = []
    for threshold_decimal in _thresholds(ctx):
        threshold = float(threshold_decimal)
        sets: dict[str, Any] = {}
        negatives: list[dict[str, Any]] = []
        for test_step, (expected_label, records, error_count) in loaded.items():
            metrics = _metrics(records, expected_label, threshold, debounce, error_count)
            sets[test_step] = metrics
            if expected_label == 0:
                negatives.append(metrics)
        threshold_rows.append(
            {
                "threshold": threshold,
                "sets": sets,
                "combined_negative": _combined_negative(negatives) if negatives else None,
            }
        )

    configured = _configured_keyword_thresholds(ctx)
    configured_thresholds: dict[str, float] = dict(configured[1]) if configured else {}
    keyword_thresholds_complete = configured is not None
    saw_ctc_candidate = configured is not None
    if configured is None:
        keyword_thresholds_complete = True
        for _expected_label, records, _error_count in loaded.values():
            for record in records:
                for candidate in record.get("stage1_candidates", []):
                    saw_ctc_candidate = True
                    keyword_id = str(candidate.get("keyword_id", ""))
                    raw_threshold = candidate.get("stage2_threshold")
                    if not keyword_id or raw_threshold is None:
                        keyword_thresholds_complete = False
                        continue
                    threshold = float(raw_threshold)
                    previous = configured_thresholds.setdefault(keyword_id, threshold)
                    if not math.isclose(previous, threshold):
                        raise ConfigurationError(
                            f"[{ctx.step}] inconsistent stage2_threshold values for {keyword_id!r}"
                        )
    selected_operating_point = None
    if saw_ctc_candidate and keyword_thresholds_complete:
        selected_sets: dict[str, Any] = {}
        selected_negatives: list[dict[str, Any]] = []
        for test_step, (expected_label, records, error_count) in loaded.items():
            metrics = _keyword_threshold_metrics(
                records,
                expected_label,
                debounce,
                error_count,
                configured_thresholds,
            )
            selected_sets[test_step] = metrics
            if expected_label == 0:
                selected_negatives.append(metrics)
        selected_operating_point = {
            "thresholds": configured_thresholds,
            "sets": selected_sets,
            "combined_negative": (
                _combined_negative(selected_negatives) if selected_negatives else None
            ),
        }

    payload = {
        "tests": _test_steps(ctx),
        "debounce_seconds": debounce,
        "performance_by_test": {
            test_step: _inference_performance_metrics(records)
            for test_step, (_expected_label, records, _error_count) in loaded.items()
        },
        "vad_by_test": {
            test_step: {
                "enabled": bool(test_summaries[test_step].get("vad_enabled", False)),
                "threshold": test_summaries[test_step].get("vad_threshold"),
                "padding_ms": test_summaries[test_step].get("vad_padding_ms"),
                "threads": test_summaries[test_step].get("vad_threads"),
                "candidate_pass_count": int(
                    test_summaries[test_step].get(
                        "vad_candidate_pass_count",
                        sum(int(record.get("vad_candidate_pass_count", 0)) for record in records),
                    )
                ),
                "candidate_reject_count": int(
                    test_summaries[test_step].get(
                        "vad_candidate_reject_count",
                        sum(int(record.get("vad_candidate_reject_count", 0)) for record in records),
                    )
                ),
            }
            for test_step, (_expected_label, records, _error_count) in loaded.items()
        },
        "metric_definitions": {
            "false_accept_rate": "false_accept_crops / crops_evaluated",
            "false_accepts_per_hour": "debounced false_accept_events / evaluated_hours",
            "false_reject_rate": "false_reject_clips / positive_clips_evaluated",
            "real_time_factor": "inference wall seconds / inference audio seconds",
            "cpu_utilization_percent": (
                "process CPU seconds / inference wall seconds * 100; may exceed 100 "
                "when inference uses multiple CPU cores"
            ),
            "peak_rss_mb": (
                "maximum process resident-set size sampled during one inference, in MiB"
            ),
        },
        "thresholds": threshold_rows,
        "keyword_threshold_operating_point": selected_operating_point,
    }
    write_json(_output_json(ctx), payload)
    _output_report(ctx).parent.mkdir(parents=True, exist_ok=True)
    _output_report(ctx).write_text(_markdown_report(payload), encoding="utf-8")
    return {"output_json": str(_output_json(ctx)), "output_report": str(_output_report(ctx)), "threshold_count": len(threshold_rows)}
