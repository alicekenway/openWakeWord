"""Augmentation stage: JSONL audio in, deterministic mixed WAVs out."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..artifacts import input_signatures, normalise_manifest_inputs, read_json, read_jsonl
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
