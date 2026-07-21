"""Threshold-sweep reporting built from per-set testing details."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..artifacts import read_json, read_jsonl, write_json
from ..config import ConfigurationError
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
    try:
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
        if test_section.get("record_window_scores", "yes").strip().lower() not in {"1", "yes", "true", "on"}:
            raise ConfigurationError(f"[{test_step}] record_window_scores must be yes for threshold summaries")
    _output_json(ctx)
    _output_report(ctx)


def input_paths(ctx: Any) -> list[Path]:
    paths: list[Path] = []
    for step in _test_steps(ctx):
        summary, details = _test_paths(ctx, step)
        paths.extend([summary, details])
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


def _events(windows: list[dict[str, Any]], threshold: float, debounce_seconds: float) -> int:
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


def _metrics(records: list[dict[str, Any]], expected_label: int, threshold: float, debounce_seconds: float, errors: int) -> dict[str, Any]:
    evaluated = 0
    seconds = 0.0
    false_rejects = 0
    false_accept_clips = 0
    false_accept_events = 0
    for record in records:
        windows = record.get("sliding_windows")
        if not isinstance(windows, list):
            continue
        events = _events(windows, threshold, debounce_seconds)
        evaluated += 1
        seconds += float(record.get("duration_seconds") or 0.0)
        if expected_label == 1:
            false_rejects += int(events == 0)
        else:
            false_accept_clips += int(events > 0)
            false_accept_events += events
    result: dict[str, Any] = {
        "clips_evaluated": evaluated,
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
                "false_accept_events": false_accept_events,
                "false_accepts_per_hour": (false_accept_events / (seconds / 3600.0)) if seconds else None,
                "false_accept_rate": (false_accept_clips / evaluated) if evaluated else None,
            }
        )
    return result


def _combined_negative(values: list[dict[str, Any]]) -> dict[str, Any]:
    clips = sum(int(value["clips_evaluated"]) for value in values)
    seconds = sum(float(value["evaluated_seconds"]) for value in values)
    accepts = sum(int(value["false_accept_events"]) for value in values)
    accept_clips = sum(int(value["false_accept_clips"]) for value in values)
    return {
        "clips_evaluated": clips,
        "evaluated_seconds": round(seconds, 6),
        "evaluated_hours": round(seconds / 3600.0, 6),
        "false_accept_clips": accept_clips,
        "false_accept_events": accepts,
        "false_accepts_per_hour": accepts / (seconds / 3600.0) if seconds else None,
        "false_accept_rate": accept_clips / clips if clips else None,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Wake-Word Threshold Summary",
        "",
        f"- Debounce: `{payload['debounce_seconds']}` seconds",
        f"- Test blocks: `{', '.join(payload['tests'])}`",
        "",
        "| Threshold | Recall / FRR by positive set | Combined negative FA/hour | Combined negative FA rate |",
        "| --- | --- | --- | --- |",
    ]
    for item in payload["thresholds"]:
        positive_parts = []
        for name, metrics in item["sets"].items():
            if "recall" in metrics:
                recall = metrics["recall"]
                frr = metrics["false_reject_rate"]
                if recall is None or frr is None:
                    positive_parts.append(f"{name}: n/a")
                else:
                    positive_parts.append(f"{name}: {recall:.4f} / {frr:.4f}")
        combined = item.get("combined_negative", {}).get("false_accepts_per_hour")
        combined_text = "n/a" if combined is None else f"{combined:.4f}"
        combined_rate = item.get("combined_negative", {}).get("false_accept_rate")
        combined_rate_text = "n/a" if combined_rate is None else f"{combined_rate:.2%}"
        lines.append(
            f"| {item['threshold']:.6g} | {'; '.join(positive_parts) or 'n/a'} | "
            f"{combined_text} | {combined_rate_text} |"
        )

    for item in payload["thresholds"]:
        lines.extend(["", f"## Threshold {item['threshold']:.6g}", ""])
        for name, metrics in item["sets"].items():
            lines.append(f"### {name}")
            for key, value in metrics.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(ctx: Any) -> dict[str, Any]:
    debounce = number(ctx.section, "debounce_seconds", ctx.step, 1.0)
    loaded: dict[str, tuple[int, list[dict[str, Any]], int]] = {}
    for test_step in _test_steps(ctx):
        test_section = ctx.config.section(test_step)
        expected_label = int(test_section["expected_label"])
        summary_path, details_path = _test_paths(ctx, test_step)
        test_summary = read_json(summary_path)
        error_count = max((int(metrics.get("error_count", 0)) for metrics in test_summary.get("sets", {}).values()), default=0)
        loaded[test_step] = (expected_label, read_jsonl(details_path, allow_empty=True), error_count)

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

    payload = {
        "tests": _test_steps(ctx),
        "debounce_seconds": debounce,
        "thresholds": threshold_rows,
    }
    write_json(_output_json(ctx), payload)
    _output_report(ctx).parent.mkdir(parents=True, exist_ok=True)
    _output_report(ctx).write_text(_markdown_report(payload), encoding="utf-8")
    return {"output_json": str(_output_json(ctx)), "output_report": str(_output_report(ctx)), "threshold_count": len(threshold_rows)}
