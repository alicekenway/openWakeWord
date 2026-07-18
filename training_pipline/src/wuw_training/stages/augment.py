"""Augmentation stage: JSONL audio in, deterministic mixed WAVs out."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..artifacts import input_signatures, normalise_manifest_inputs, read_json, read_jsonl, write_json, write_jsonl
from ..config import ConfigurationError, parse_json
from ..legacy import get_legacy_module
from .common import boolean, integer, number, placement, require, stage_work_path


def _noise_dirs(ctx: Any) -> list[Path]:
    raw = ctx.section.get("noise_dir")
    if not raw:
        return []
    values: list[str]
    if raw.strip().startswith("["):
        parsed = parse_json(raw, f"[{ctx.step}] noise_dir", list)
        if not all(isinstance(item, str) for item in parsed):
            raise ConfigurationError(f"[{ctx.step}] noise_dir JSON entries must be strings")
        values = parsed
    else:
        values = [raw]
    return [ctx.config.resolve_path(value) for value in values]


def _input_manifests(ctx: Any):
    from ..artifacts import parse_manifest_inputs

    return parse_manifest_inputs(ctx.config, ctx.step)


def _noise_manifests(ctx: Any):
    from ..artifacts import parse_manifest_inputs

    return parse_manifest_inputs(
        ctx.config,
        ctx.step,
        key="noise_jsonl",
        base_key="noise_audio_base_dir",
        required=False,
    )


def _output_manifest(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_manifest", ctx.step))


def _output_dir(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_dir", ctx.step))


def validate(ctx: Any) -> None:
    inputs = _input_manifests(ctx)
    if not inputs:
        raise ConfigurationError(f"[{ctx.step}] requires at least one input_jsonl")
    noise_inputs = _noise_manifests(ctx)
    noise_dirs = _noise_dirs(ctx)
    if not noise_inputs and not noise_dirs:
        raise ConfigurationError(f"[{ctx.step}] requires noise_jsonl or noise_dir")
    for item in [*inputs, *noise_inputs]:
        if item.audio_base_dir and not item.audio_base_dir.is_dir():
            raise ConfigurationError(f"[{ctx.step}] audio_base_dir does not exist: {item.audio_base_dir}")
    for directory in noise_dirs:
        if directory.exists() and not directory.is_dir():
            raise ConfigurationError(f"[{ctx.step}] noise_dir is not a directory: {directory}")
    if integer(ctx.section, "rounds", ctx.step, 1) < 1:
        raise ConfigurationError(f"[{ctx.step}] rounds must be >= 1")
    if number(ctx.section, "snr_low", ctx.step, -5.0) > number(ctx.section, "snr_high", ctx.step, 15.0):
        raise ConfigurationError(f"[{ctx.step}] snr_low must be <= snr_high")
    probability = number(ctx.section, "artificial_probability", ctx.step, 0.0)
    if not 0.0 <= probability <= 1.0:
        raise ConfigurationError(f"[{ctx.step}] artificial_probability must be between 0 and 1")
    placement(ctx.section, ctx.step)
    _output_dir(ctx)
    _output_manifest(ctx)


def input_paths(ctx: Any) -> list[Path]:
    paths = [item.path for item in _input_manifests(ctx)]
    paths.extend(item.path for item in _noise_manifests(ctx))
    paths.extend(_noise_dirs(ctx))
    return paths


def output_paths(ctx: Any) -> list[Path]:
    manifest = _output_manifest(ctx)
    return [manifest, manifest.with_suffix(".summary.json")]


def validate_outputs(ctx: Any) -> bool:
    manifest, summary_path = output_paths(ctx)
    if not manifest.is_file() or not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        return int(summary.get("error_count", -1)) == 0 and int(summary.get("output_count", -1)) == len(read_jsonl(manifest))
    except Exception:
        return False


def run(ctx: Any) -> dict[str, Any]:
    inputs = _input_manifests(ctx)
    noise_inputs = _noise_manifests(ctx)
    normalized_input = stage_work_path(ctx, "input.jsonl")
    normalized_noise = stage_work_path(ctx, "noise.jsonl")
    _, input_count = normalise_manifest_inputs(
        inputs,
        normalized_input,
        default_placement=placement(ctx.section, ctx.step),
    )
    noise_manifest_args: list[str] = []
    if noise_inputs:
        normalise_manifest_inputs(noise_inputs, normalized_noise)
        noise_manifest_args.append(str(normalized_noise))

    legacy = get_legacy_module()
    legacy.command_augment_audio(
        argparse.Namespace(
            input_manifest=str(normalized_input),
            input_dir=None,
            noise_manifest=noise_manifest_args,
            noise_dir=[str(path) for path in _noise_dirs(ctx)],
            output_dir=str(_output_dir(ctx)),
            output_manifest=str(_output_manifest(ctx)),
            rounds=integer(ctx.section, "rounds", ctx.step, 1),
            snr_low=number(ctx.section, "snr_low", ctx.step, -5.0),
            snr_high=number(ctx.section, "snr_high", ctx.step, 15.0),
            artificial_prob=number(ctx.section, "artificial_probability", ctx.step, 0.0),
            random_gain_db=number(ctx.section, "random_gain_db", ctx.step, 0.0),
            clip_seconds=number(ctx.config.section("main"), "clip_seconds", "main", 2.0),
            sample_rate=integer(ctx.config.section("main"), "sample_rate", "main", 16000),
            placement=placement(ctx.section, ctx.step),
            seed=integer(ctx.config.section("main"), "seed", "main", 1337),
            overwrite=ctx.force or boolean(ctx.section, "overwrite", ctx.step, False),
            workers=integer(ctx.section, "workers", ctx.step, 1),
        )
    )
    if not validate_outputs(ctx):
        raise RuntimeError(f"Augmentation output validation failed for {ctx.step}")
    summary = read_json(_output_manifest(ctx).with_suffix(".summary.json"))
    return {
        "input_count": input_count,
        "output_count": summary.get("output_count"),
        "output_manifest": str(_output_manifest(ctx)),
        "input_signatures": input_signatures(inputs),
    }


def prepare_slurm_shards(ctx: Any, work_dir: Path, task_count: int) -> list[dict[str, Any]]:
    normalized_input = work_dir / "input.jsonl"
    normalized_noise = work_dir / "noise.jsonl"
    inputs = _input_manifests(ctx)
    noise_inputs = _noise_manifests(ctx)
    normalise_manifest_inputs(
        inputs,
        normalized_input,
        default_placement=placement(ctx.section, ctx.step),
    )
    if noise_inputs:
        normalise_manifest_inputs(noise_inputs, normalized_noise)
    records = read_jsonl(normalized_input)
    actual_count = min(task_count, len(records))
    if actual_count < 1:
        raise RuntimeError(f"Augmentation stage {ctx.step} has no records to shard")
    base, extra = divmod(len(records), actual_count)
    result: list[dict[str, Any]] = []
    start = 0
    for task_id in range(actual_count):
        stop = start + base + (1 if task_id < extra else 0)
        shard_dir = work_dir / "shards" / f"{task_id:05d}"
        shard_records = []
        for index, record in enumerate(records[start:stop], start=start):
            updated = dict(record)
            updated["_slurm_index"] = index
            shard_records.append(updated)
        input_manifest = shard_dir / "input.jsonl"
        output_manifest = shard_dir / "output.jsonl"
        write_jsonl(input_manifest, shard_records)
        result.append(
            {
                "id": task_id,
                "start": start,
                "stop": stop,
                "count": stop - start,
                "input_manifest": str(input_manifest),
                "output_manifest": str(output_manifest),
                "normalized_manifest": str(normalized_input),
                "noise_manifest": str(normalized_noise) if noise_inputs else None,
            }
        )
        start = stop
    return result


def run_slurm_shard(ctx: Any, task: dict[str, Any]) -> dict[str, Any]:
    noise_manifest = task.get("noise_manifest")
    legacy = get_legacy_module()
    legacy.command_augment_audio(
        argparse.Namespace(
            input_manifest=str(task["input_manifest"]),
            input_dir=None,
            noise_manifest=[str(noise_manifest)] if noise_manifest else [],
            noise_dir=[str(path) for path in _noise_dirs(ctx)],
            output_dir=str(_output_dir(ctx)),
            output_manifest=str(task["output_manifest"]),
            rounds=integer(ctx.section, "rounds", ctx.step, 1),
            snr_low=number(ctx.section, "snr_low", ctx.step, -5.0),
            snr_high=number(ctx.section, "snr_high", ctx.step, 15.0),
            artificial_prob=number(ctx.section, "artificial_probability", ctx.step, 0.0),
            random_gain_db=number(ctx.section, "random_gain_db", ctx.step, 0.0),
            clip_seconds=number(ctx.config.section("main"), "clip_seconds", "main", 2.0),
            sample_rate=integer(ctx.config.section("main"), "sample_rate", "main", 16000),
            placement=placement(ctx.section, ctx.step),
            seed=integer(ctx.config.section("main"), "seed", "main", 1337),
            overwrite=True,
            workers=integer(ctx.section, "workers", ctx.step, 1),
            index_offset=int(task["start"]),
        )
    )
    summary = read_json(Path(str(task["output_manifest"])).with_suffix(".summary.json"))
    return {
        "output_manifest": str(task["output_manifest"]),
        "output_count": summary.get("output_count"),
    }


def validate_slurm_shard(ctx: Any, task: dict[str, Any]) -> bool:
    output_manifest = Path(str(task["output_manifest"])).resolve()
    summary_path = output_manifest.with_suffix(".summary.json")
    if not output_manifest.is_file() or not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        return (
            int(summary.get("error_count", -1)) == 0
            and int(summary.get("input_count", -1)) == int(task["count"])
            and int(summary.get("output_count", -1)) == int(task["count"]) * integer(ctx.section, "rounds", ctx.step, 1)
            and len(read_jsonl(output_manifest)) == int(summary["output_count"])
        )
    except Exception:
        return False


def merge_slurm_shards(ctx: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for task in tasks:
        output_manifest = Path(str(task["output_manifest"])).resolve()
        merged.extend(read_jsonl(output_manifest))
        shard_summaries.append(read_json(output_manifest.with_suffix(".summary.json")))
    merged.sort(key=lambda record: (int(record.get("augmentation_round", 0)), int(record.get("_slurm_index", -1))))
    for record in merged:
        record.pop("_slurm_index", None)
    output_manifest = _output_manifest(ctx)
    write_jsonl(output_manifest, merged)

    legacy = get_legacy_module()
    normalized = Path(str(tasks[0]["normalized_manifest"])).resolve()
    items = legacy.feature_items_from_feature_inputs(
        [str(normalized)], [], placement(ctx.section, ctx.step)
    )
    first = shard_summaries[0]
    summary = {
        "input_count": len(items),
        "output_count": len(merged),
        "rounds": integer(ctx.section, "rounds", ctx.step, 1),
        "noise_count": first.get("noise_count"),
        "snr_low": number(ctx.section, "snr_low", ctx.step, -5.0),
        "snr_high": number(ctx.section, "snr_high", ctx.step, 15.0),
        "artificial_prob": number(ctx.section, "artificial_probability", ctx.step, 0.0),
        "placement": placement(ctx.section, ctx.step),
        "placement_counts": legacy.placement_counts(value for _path, value in items),
        "input_signature": legacy.feature_input_signature(items),
        "workers": integer(ctx.section, "workers", ctx.step, 1),
        "errors": [],
        "error_count": 0,
    }
    write_json(output_manifest.with_suffix(".summary.json"), summary)
    if not validate_outputs(ctx):
        raise RuntimeError(f"Augmentation merge validation failed for {ctx.step}")
    return {
        "input_count": len(items),
        "output_count": len(merged),
        "output_manifest": str(output_manifest),
        "input_signatures": input_signatures(_input_manifests(ctx)),
    }


def cleanup_slurm_shards(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        output_manifest = Path(str(task["output_manifest"])).resolve()
        output_manifest.unlink(missing_ok=True)
        output_manifest.with_suffix(".summary.json").unlink(missing_ok=True)
