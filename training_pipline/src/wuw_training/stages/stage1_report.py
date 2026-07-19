"""Threshold-selection report for frozen CTC candidate feature bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..artifacts import read_json, write_json
from ..config import ConfigurationError
from ..ctc_wac import feature_bundle_paths, feature_bundle_valid
from .common import csv_option, number, require


QUANTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


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
    blocks = _blocks(ctx)
    if not blocks:
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
        return payload.get("report_schema") == 1 and "Stage-1 CTC Candidate Report" in output_report.read_text(encoding="utf-8")
    except Exception:
        return False


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {str(value): None for value in QUANTILES}
    return {str(value): float(np.percentile(values, value)) for value in QUANTILES}


def _winner_counts(winners: np.ndarray, keyword_ids: list[str]) -> dict[str, int]:
    result = {value: 0 for value in keyword_ids}
    for index, count in zip(*np.unique(winners, return_counts=True)):
        result[keyword_ids[int(index)]] = int(count)
    return result


def _block_payload(block: ReportBlock) -> dict[str, Any]:
    if not feature_bundle_valid(block.path):
        raise RuntimeError(f"{block.name} is not a complete schema-2 CTC-WAC feature bundle: {block.path}")
    paths = feature_bundle_paths(block.path)
    summary = read_json(paths.summary)
    scores = np.asarray(np.load(paths.all_scores, mmap_mode="r"), dtype=np.float32)
    top = np.asarray(np.load(paths.top_score, mmap_mode="r"), dtype=np.float32).reshape(-1)
    margin = np.asarray(np.load(paths.margin, mmap_mode="r"), dtype=np.float32).reshape(-1)
    lengths = np.asarray(np.load(paths.lengths, mmap_mode="r"), dtype=np.int64).reshape(-1)
    winner = np.argmax(scores, axis=1).astype(np.int64)
    keyword_ids = [str(value) for value in summary.get("keyword_ids", [])]
    if len(keyword_ids) != scores.shape[1]:
        raise RuntimeError(f"{block.name} has inconsistent keyword IDs and score columns")
    frame_shift_ms = float(summary.get("encoder_frame_shift_ms", 40.0))
    duration = float(summary.get("input_duration_seconds", 0.0))
    return {
        "name": block.name,
        "label": block.label,
        "split": block.split,
        "input_rows": int(summary.get("input_count", top.size)),
        "candidate_rows": int(top.size),
        "invalid_alignment_rows": int(summary.get("invalid_alignment_rows", 0)),
        "input_duration_seconds": duration,
        "keyword_ids": keyword_ids,
        "score_quantiles": _quantiles(top),
        "score_exp_quantiles": _quantiles(np.exp(np.clip(top, -100.0, 0.0))),
        "margin_quantiles": _quantiles(margin),
        "candidate_length_frames_quantiles": _quantiles(lengths),
        "candidate_length_ms_quantiles": _quantiles(lengths * frame_shift_ms),
        "winner_counts": _winner_counts(winner, keyword_ids),
        # Internal arrays are removed before JSON output but allow aggregate
        # and per-keyword calculations without reading files twice.
        "_top": top,
        "_margin": margin,
        "_lengths": lengths,
        "_winner": winner,
        "_frame_shift_ms": frame_shift_ms,
    }


def _aggregate(values: Iterable[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    blocks = list(values)
    if not blocks:
        return {"candidate_rows": 0, "input_rows": 0, "threshold_sweep": []}
    top = np.concatenate([block["_top"] for block in blocks])
    margin = np.concatenate([block["_margin"] for block in blocks])
    lengths = np.concatenate([block["_lengths"] for block in blocks])
    winners = np.concatenate([block["_winner"] for block in blocks])
    ids = list(blocks[0]["keyword_ids"])
    if any(list(block["keyword_ids"]) != ids for block in blocks):
        raise RuntimeError("Stage-1 report blocks were generated with different keyword ID orderings")
    duration = sum(float(block["input_duration_seconds"]) for block in blocks)
    result = {
        "input_rows": sum(int(block["input_rows"]) for block in blocks),
        "candidate_rows": int(top.size),
        "invalid_alignment_rows": sum(int(block["invalid_alignment_rows"]) for block in blocks),
        "input_duration_seconds": duration,
        "score_quantiles": _quantiles(top),
        "score_exp_quantiles": _quantiles(np.exp(np.clip(top, -100.0, 0.0))),
        "margin_quantiles": _quantiles(margin),
        "candidate_length_frames_quantiles": _quantiles(lengths),
        "candidate_length_ms_quantiles": _quantiles(lengths * float(blocks[0]["_frame_shift_ms"])),
        "winner_counts": _winner_counts(winners, ids),
        "threshold_sweep": [],
    }
    for threshold in thresholds:
        retained = int(np.count_nonzero(top >= threshold))
        result["threshold_sweep"].append(
            {
                "threshold": threshold,
                "retained_rows": retained,
                "retention_rate": retained / int(top.size) if top.size else None,
                "candidate_windows_per_hour": retained / (duration / 3600.0) if duration else None,
            }
        )
    return result


def _per_keyword(values: Iterable[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    blocks = list(values)
    if not blocks:
        return {}
    keyword_ids = list(blocks[0]["keyword_ids"])
    if any(list(block["keyword_ids"]) != keyword_ids for block in blocks):
        raise RuntimeError("Stage-1 report blocks were generated with different keyword ID orderings")
    duration = sum(float(block["input_duration_seconds"]) for block in blocks)
    result: dict[str, Any] = {}
    for index, keyword_id in enumerate(keyword_ids):
        scores = [block["_top"][block["_winner"] == index] for block in blocks]
        margins = [block["_margin"][block["_winner"] == index] for block in blocks]
        lengths = [block["_lengths"][block["_winner"] == index] for block in blocks]
        top = np.concatenate(scores) if scores else np.empty((0,), dtype=np.float32)
        margin = np.concatenate(margins) if margins else np.empty((0,), dtype=np.float32)
        frames = np.concatenate(lengths) if lengths else np.empty((0,), dtype=np.int64)
        sweep = []
        for threshold in thresholds:
            retained = int(np.count_nonzero(top >= threshold))
            sweep.append(
                {
                    "threshold": threshold,
                    "retained_rows": retained,
                    "retention_rate": retained / int(top.size) if top.size else None,
                    "candidate_windows_per_hour": retained / (duration / 3600.0) if duration else None,
                }
            )
        result[keyword_id] = {
            "candidate_rows": int(top.size),
            "score_quantiles": _quantiles(top),
            "margin_quantiles": _quantiles(margin),
            "candidate_length_frames_quantiles": _quantiles(frames),
            "threshold_sweep": sweep,
        }
    return result


def _public_block(block: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in block.items() if not key.startswith("_")}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-1 CTC Candidate Report",
        "",
        "Normalized log score is authoritative; `exp(score)` is shown only as a readable view.",
        "Candidate windows/hour is a stage-1 screening statistic, not final false accepts/hour.",
        "",
        "## Aggregate candidate counts",
        "",
        "| Split | Label | Input rows | Candidate rows | Invalid alignments | Duration hours |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, values in payload["aggregates"].items():
        split, label = key.split(":", 1)
        lines.append(
            f"| {split} | {label} | {values['input_rows']} | {values['candidate_rows']} | "
            f"{values['invalid_alignment_rows']} | {values['input_duration_seconds'] / 3600.0:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Shared threshold sweep",
            "",
            "| Threshold | Positive retained | Positive retention | Negative retained | Negative windows/hour |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    positive = payload["aggregates"].get("train:positive", {"threshold_sweep": []})
    negative = payload["aggregates"].get("train:negative", {"threshold_sweep": []})
    for positive_row, negative_row in zip(positive.get("threshold_sweep", []), negative.get("threshold_sweep", [])):
        negative_rate = negative_row["candidate_windows_per_hour"]
        lines.append(
            f"| {positive_row['threshold']:.6g} | {positive_row['retained_rows']} | "
            f"{positive_row['retention_rate'] if positive_row['retention_rate'] is not None else 'n/a'} | "
            f"{negative_row['retained_rows']} | {negative_rate if negative_rate is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Per-keyword winner distributions",
            "",
            "| Keyword | Candidates | Score p50 | Score p95 | Margin p50 | Length p50 frames |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for keyword, values in sorted(payload["per_keyword"].items()):
        score = values["score_quantiles"]
        margin = values["margin_quantiles"]
        length = values["candidate_length_frames_quantiles"]
        lines.append(
            f"| {keyword} | {values['candidate_rows']} | {score['50']} | {score['95']} | "
            f"{margin['50']} | {length['50']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def run(ctx: Any) -> dict[str, Any]:
    thresholds = _thresholds(ctx)
    loaded = [_block_payload(block) for block in _blocks(ctx)]
    aggregates: dict[str, Any] = {}
    for split in ("train", "dev", "test", "false_positive"):
        for label, label_name in ((1, "positive"), (0, "negative")):
            selected = [block for block in loaded if block["split"] == split and block["label"] == label]
            if selected:
                aggregates[f"{split}:{label_name}"] = _aggregate(selected, thresholds)
    all_blocks = _aggregate(loaded, thresholds)
    winner_counts: dict[str, int] = {}
    for block in loaded:
        for key, value in block["winner_counts"].items():
            winner_counts[key] = winner_counts.get(key, 0) + int(value)
    per_keyword = _per_keyword(loaded, thresholds)
    payload = {
        "report_schema": 1,
        "threshold_sweep": {"start": thresholds[0], "stop": thresholds[-1], "step": thresholds[1] - thresholds[0] if len(thresholds) > 1 else 0.0},
        "blocks": [_public_block(block) for block in loaded],
        "aggregates": aggregates,
        "all": all_blocks,
        "winner_counts": winner_counts,
        "per_keyword": per_keyword,
    }
    output_json = _output_json(ctx)
    output_report = _output_report(ctx)
    write_json(output_json, payload)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_markdown(payload), encoding="utf-8")
    if not validate_outputs(ctx):
        raise RuntimeError(f"Stage-1 report output validation failed for {ctx.step}")
    return {"output_json": str(output_json), "output_report": str(output_report), "block_count": len(loaded)}
