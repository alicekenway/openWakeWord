"""Threshold tables for frozen CTC candidate feature bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import read_json, read_jsonl, write_json
from ..config import ConfigurationError
from ..ctc_wac import feature_bundle_paths, feature_bundle_valid
from .common import csv_option, number, require


@dataclass(frozen=True)
class ReportBlock:
    name: str
    path: Path
    label: int
    split: str


def _references(ctx: Any) -> list[str]:
    return csv_option(ctx.section, "features", ctx.step)


def _blocks(ctx: Any) -> list[ReportBlock]:
    blocks: list[ReportBlock] = []
    for name in _references(ctx):
        if not name.startswith("feature."):
            raise ConfigurationError(f"[{ctx.step}] features entries must name feature.* blocks, got {name!r}")
        section = ctx.config.section(name)
        output = section.get("output_file")
        if not output:
            raise ConfigurationError(f"[{name}] is missing output_file")
        try:
            label = int(section.get("label", ""))
        except ValueError as exc:
            raise ConfigurationError(f"[{name}] label must be 0 or 1") from exc
        split = section.get("split", "").lower()
        if label not in {0, 1} or split not in {"train", "dev", "test", "false_positive"}:
            raise ConfigurationError(f"[{name}] must define label = 0|1 and a valid split")
        blocks.append(ReportBlock(name, ctx.config.resolve_path(output), label, split))
    return blocks


def _output_json(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_json", ctx.step))


def _output_report(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_report", ctx.step))


def _thresholds(ctx: Any) -> list[float]:
    start = number(ctx.section, "threshold_start", ctx.step, -5.0)
    stop = number(ctx.section, "threshold_stop", ctx.step, 0.0)
    step = number(ctx.section, "threshold_step", ctx.step, 0.05)
    if not all(math.isfinite(value) for value in (start, stop, step)) or start > stop or step <= 0:
        raise ConfigurationError(f"[{ctx.step}] threshold_start/stop/step must be finite with start <= stop and step > 0")
    count = int(round((stop - start) / step))
    values = [start + index * step for index in range(count + 1)]
    if not values or abs(values[-1] - stop) > max(1.0e-8, abs(step) * 1.0e-6):
        raise ConfigurationError(f"[{ctx.step}] threshold_step must reach threshold_stop exactly")
    values[-1] = stop
    return values


def validate(ctx: Any) -> None:
    if not _blocks(ctx):
        raise ConfigurationError(f"[{ctx.step}] requires at least one feature.* block")
    _thresholds(ctx)
    _output_json(ctx)
    _output_report(ctx)


def input_paths(ctx: Any) -> list[Path]:
    paths: list[Path] = []
    for block in _blocks(ctx):
        paths.extend(feature_bundle_paths(block.path).all())
    return list(dict.fromkeys(paths))


def output_paths(ctx: Any) -> list[Path]:
    return [_output_json(ctx), _output_report(ctx)]


def validate_outputs(ctx: Any) -> bool:
    output_json, output_report = output_paths(ctx)
    if not output_json.is_file() or not output_report.is_file():
        return False
    try:
        payload = read_json(output_json)
        return payload.get("report_schema") == 2 and "Stage-1 Threshold Tables" in output_report.read_text(
            encoding="utf-8"
        )
    except Exception:
        return False


def _keyword_ids(summary: dict[str, Any], scores: np.ndarray, block: ReportBlock) -> list[str]:
    keyword_ids = [str(value) for value in summary.get("keyword_ids", [])]
    if len(keyword_ids) != scores.shape[1]:
        raise RuntimeError(f"{block.name} has inconsistent keyword IDs and score columns")
    return keyword_ids


def _integer_counts(raw: Any, keyword_ids: list[str], *, name: str, block: ReportBlock) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"{block.name} has no {name}; regenerate this positive feature bundle with the current pipeline"
        )
    result: dict[str, int] = {}
    for keyword_id in keyword_ids:
        try:
            count = int(raw.get(keyword_id, 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{block.name} has invalid {name}[{keyword_id!r}]") from exc
        if count < 0:
            raise RuntimeError(f"{block.name} has negative {name}[{keyword_id!r}]")
        result[keyword_id] = count
    return result


def _positive_rows(
    *,
    block: ReportBlock,
    paths: Any,
    scores: np.ndarray,
    keyword_ids: list[str],
    summary: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    expected_counts = _integer_counts(
        summary.get("expected_keyword_counts"),
        keyword_ids,
        name="expected_keyword_counts",
        block=block,
    )
    invalid_counts = _integer_counts(
        summary.get("expected_keyword_invalid_alignment_counts"),
        keyword_ids,
        name="expected_keyword_invalid_alignment_counts",
        block=block,
    )
    rows = read_jsonl(paths.rows, allow_empty=True)
    by_keyword: dict[str, list[int]] = {keyword_id: [] for keyword_id in keyword_ids}
    seen_rows: set[int] = set()
    for metadata in rows:
        try:
            row = int(metadata["row"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{block.name} has a row without a valid row index") from exc
        if row < 0 or row >= scores.shape[0] or row in seen_rows:
            raise RuntimeError(f"{block.name} has invalid or duplicate row index {row}")
        seen_rows.add(row)
        expected = metadata.get("expected_keyword_id")
        if not isinstance(expected, str) or expected not in by_keyword:
            raise RuntimeError(
                f"{block.name} has no valid expected_keyword_id in {paths.rows}; regenerate this positive feature bundle"
            )
        by_keyword[expected].append(row)
    if len(seen_rows) != scores.shape[0]:
        raise RuntimeError(f"{block.name} rows metadata does not cover every score row")
    result = {
        keyword_id: np.asarray(indices, dtype=np.int64) for keyword_id, indices in by_keyword.items()
    }
    for keyword_id in keyword_ids:
        accounted = int(result[keyword_id].size) + invalid_counts[keyword_id]
        if accounted != expected_counts[keyword_id]:
            raise RuntimeError(
                f"{block.name} expected-keyword accounting mismatch for {keyword_id}: "
                f"{accounted} rows recorded but summary says {expected_counts[keyword_id]}"
            )
    return result, expected_counts


def _positive_table(
    *,
    block: ReportBlock,
    paths: Any,
    scores: np.ndarray,
    keyword_ids: list[str],
    summary: dict[str, Any],
    thresholds: list[float],
) -> dict[str, Any]:
    rows_by_keyword, expected_counts = _positive_rows(
        block=block,
        paths=paths,
        scores=scores,
        keyword_ids=keyword_ids,
        summary=summary,
    )
    table: list[dict[str, Any]] = []
    for threshold in thresholds:
        values: dict[str, Any] = {}
        for column, keyword_id in enumerate(keyword_ids):
            total = expected_counts[keyword_id]
            rows = rows_by_keyword[keyword_id]
            accepted = int(np.count_nonzero(scores[rows, column] >= threshold)) if rows.size else 0
            false_rejections = total - accepted
            values[keyword_id] = {
                "expected_rows": total,
                "accepted_rows": accepted,
                "false_rejections": false_rejections,
                "accuracy": accepted / total if total else None,
                "false_rejection_rate": false_rejections / total if total else None,
            }
        table.append({"threshold": threshold, "keywords": values})
    return {
        "metric": "accuracy_and_false_rejection_rate",
        "expected_keyword_rows": expected_counts,
        "threshold_table": table,
    }


def _negative_table(
    *,
    block: ReportBlock,
    scores: np.ndarray,
    keyword_ids: list[str],
    summary: dict[str, Any],
    thresholds: list[float],
) -> dict[str, Any]:
    try:
        duration_seconds = float(summary.get("input_duration_seconds", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{block.name} has an invalid input_duration_seconds") from exc
    if duration_seconds <= 0:
        raise RuntimeError(f"{block.name} needs a positive input_duration_seconds to calculate FA/h")
    try:
        input_rows = int(summary.get("input_count", scores.shape[0]))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{block.name} has an invalid input_count") from exc
    if input_rows <= 0:
        raise RuntimeError(f"{block.name} needs a positive input_count to calculate FA rate")
    if scores.shape[0] > input_rows:
        raise RuntimeError(
            f"{block.name} has {scores.shape[0]} candidate rows but only {input_rows} input rows"
        )
    duration_hours = duration_seconds / 3600.0
    table: list[dict[str, Any]] = []
    for threshold in thresholds:
        values: dict[str, Any] = {}
        for column, keyword_id in enumerate(keyword_ids):
            false_accepts = int(np.count_nonzero(scores[:, column] >= threshold))
            values[keyword_id] = {
                "false_accepts": false_accepts,
                "false_accepts_per_hour": false_accepts / duration_hours,
                "false_accept_rate": false_accepts / input_rows,
            }
        table.append({"threshold": threshold, "keywords": values})
    return {
        "metric": "stage1_false_accept_candidates_per_hour",
        "input_rows": input_rows,
        "input_duration_seconds": duration_seconds,
        "threshold_table": table,
    }


def _block_payload(block: ReportBlock, thresholds: list[float]) -> dict[str, Any]:
    if not feature_bundle_valid(block.path):
        raise RuntimeError(f"{block.name} is not a complete schema-3 CTC-WAC feature bundle: {block.path}")
    paths = feature_bundle_paths(block.path)
    summary = read_json(paths.summary)
    scores = np.asarray(np.load(paths.all_scores, mmap_mode="r"), dtype=np.float32)
    keyword_ids = _keyword_ids(summary, scores, block)
    common = {
        "name": block.name,
        "label": block.label,
        "split": block.split,
        "input_rows": int(summary.get("input_count", scores.shape[0])),
        "candidate_rows": int(scores.shape[0]),
        "invalid_alignment_rows": int(summary.get("invalid_alignment_rows", 0)),
        "keyword_ids": keyword_ids,
    }
    if block.label == 1:
        return {
            **common,
            **_positive_table(
                block=block,
                paths=paths,
                scores=scores,
                keyword_ids=keyword_ids,
                summary=summary,
                thresholds=thresholds,
            ),
        }
    return {
        **common,
        **_negative_table(
            block=block,
            scores=scores,
            keyword_ids=keyword_ids,
            summary=summary,
            thresholds=thresholds,
        ),
    }


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-1 Threshold Tables",
        "",
        "Each row is one configured normalized CTC log-score threshold.",
        "For positives, `Acc / FR` means the expected keyword score passed the threshold / did not pass it; no CTC alignment counts as FR.",
        (
            "For negatives, `FA/h` is the stage-1 candidate count per source-audio hour and `FA rate` is "
            "the share of all input clips whose candidate passed the threshold; neither is a final system metric."
        ),
    ]
    for block in payload["blocks"]:
        keyword_ids = block["keyword_ids"]
        lines.extend(["", f"## {block['name']}", ""])
        if block["label"] == 1:
            counts = block["expected_keyword_rows"]
            lines.extend(
                [
                    "Positive set. Each column header gives its expected-example count.",
                    "",
                    "| Threshold | "
                    + " | ".join(
                        f"{keyword_id} (Acc / FR; n={counts[keyword_id]})" for keyword_id in keyword_ids
                    )
                    + " |",
                    "| ---: | " + " | ".join("---:" for _ in keyword_ids) + " |",
                ]
            )
            for threshold_row in block["threshold_table"]:
                cells = []
                for keyword_id in keyword_ids:
                    value = threshold_row["keywords"][keyword_id]
                    cells.append(
                        f"{_format_percent(value['accuracy'])} / {_format_percent(value['false_rejection_rate'])}"
                    )
                lines.append(f"| {threshold_row['threshold']:.6g} | " + " | ".join(cells) + " |")
        else:
            duration = block["input_duration_seconds"] / 3600.0
            lines.extend(
                [
                    f"Negative set. Source duration: {duration:.6f} h.",
                    "",
                    "| Threshold | "
                    + " | ".join(f"{keyword_id} (FA/h / FA rate)" for keyword_id in keyword_ids)
                    + " |",
                    "| ---: | " + " | ".join("---:" for _ in keyword_ids) + " |",
                ]
            )
            for threshold_row in block["threshold_table"]:
                cells = []
                for keyword_id in keyword_ids:
                    value = threshold_row["keywords"][keyword_id]
                    cells.append(
                        f"{value['false_accepts_per_hour']:.4f} / "
                        f"{_format_percent(value['false_accept_rate'])}"
                    )
                lines.append(f"| {threshold_row['threshold']:.6g} | " + " | ".join(cells) + " |")
    return "\n".join(lines).rstrip() + "\n"


def run(ctx: Any) -> dict[str, Any]:
    thresholds = _thresholds(ctx)
    blocks = [_block_payload(block, thresholds) for block in _blocks(ctx)]
    payload = {
        "report_schema": 2,
        "threshold_sweep": {
            "start": thresholds[0],
            "stop": thresholds[-1],
            "step": thresholds[1] - thresholds[0] if len(thresholds) > 1 else 0.0,
        },
        "blocks": blocks,
    }
    output_json = _output_json(ctx)
    output_report = _output_report(ctx)
    write_json(output_json, payload)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_markdown(payload), encoding="utf-8")
    if not validate_outputs(ctx):
        raise RuntimeError(f"Stage-1 report output validation failed for {ctx.step}")
    return {"output_json": str(output_json), "output_report": str(output_report), "block_count": len(blocks)}
