"""Feature extraction stage: normalized JSONL audio to openWakeWord NPY arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import normalise_manifest_inputs, read_json
from ..config import ConfigurationError
from ..legacy import get_legacy_module
from .common import boolean, integer, number, placement, require, stage_work_path


def _inputs(ctx: Any):
    from ..artifacts import parse_manifest_inputs

    return parse_manifest_inputs(ctx.config, ctx.step)


def _output_file(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_file", ctx.step))


def _model_dir(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "model_dir", ctx.step))


def validate(ctx: Any) -> None:
    inputs = _inputs(ctx)
    for item in inputs:
        if item.audio_base_dir and not item.audio_base_dir.is_dir():
            raise ConfigurationError(f"[{ctx.step}] audio_base_dir does not exist: {item.audio_base_dir}")
    label = integer(ctx.section, "label", ctx.step)
    if label not in {0, 1}:
        raise ConfigurationError(f"[{ctx.step}] label must be 0 or 1")
    split = require(ctx.section, "split", ctx.step).lower()
    if split not in {"train", "dev", "test", "false_positive"}:
        raise ConfigurationError(f"[{ctx.step}] split must be train, dev, test, or false_positive")
    model_dir = _model_dir(ctx)
    if model_dir.exists() and not model_dir.is_dir():
        raise ConfigurationError(f"[{ctx.step}] model_dir is not a directory: {model_dir}")
    if integer(ctx.section, "batch_size", ctx.step, 64) < 1:
        raise ConfigurationError(f"[{ctx.step}] batch_size must be >= 1")
    if integer(ctx.section, "ncpu", ctx.step, 1) < 1:
        raise ConfigurationError(f"[{ctx.step}] ncpu must be >= 1")
    device = ctx.section.get("device", "auto").lower()
    if device not in {"auto", "cpu", "gpu"}:
        raise ConfigurationError(f"[{ctx.step}] device must be auto, cpu, or gpu")
    placement(ctx.section, ctx.step)
    _output_file(ctx)


def input_paths(ctx: Any) -> list[Path]:
    paths = [item.path for item in _inputs(ctx)]
    model_dir = _model_dir(ctx)
    paths.extend([model_dir / "melspectrogram.onnx", model_dir / "embedding_model.onnx"])
    return paths


def output_paths(ctx: Any) -> list[Path]:
    output = _output_file(ctx)
    return [output, output.with_suffix(".summary.json")]


def validate_outputs(ctx: Any) -> bool:
    output, summary_path = output_paths(ctx)
    if not output.is_file() or not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        values = np.load(output, mmap_mode="r")
        return int(summary.get("error_count", -1)) == 0 and int(summary.get("feature_count", -1)) == int(values.shape[0])
    except Exception:
        return False


def run(ctx: Any) -> dict[str, Any]:
    normalized = stage_work_path(ctx, "input.jsonl")
    inputs = _inputs(ctx)
    normalise_manifest_inputs(
        inputs,
        normalized,
        default_placement=placement(ctx.section, ctx.step),
        label=integer(ctx.section, "label", ctx.step),
    )
    main = ctx.config.section("main")
    legacy = get_legacy_module()
    legacy.command_generate_features(
        argparse.Namespace(
            audio_manifest=[str(normalized)],
            audio_dir=[],
            output_file=str(_output_file(ctx)),
            model_dir=str(_model_dir(ctx)),
            batch_size=integer(ctx.section, "batch_size", ctx.step, 64),
            audio_loader_workers=integer(ctx.section, "audio_loader_workers", ctx.step, 1),
            prefetch_batches=integer(ctx.section, "prefetch_batches", ctx.step, 1),
            ncpu=integer(ctx.section, "ncpu", ctx.step, 1),
            device=ctx.section.get("device", "auto").lower(),
            limit=None,
            clip_seconds=number(main, "clip_seconds", "main", 2.0),
            sample_rate=integer(main, "sample_rate", "main", 16000),
            placement=placement(ctx.section, ctx.step),
            seed=integer(main, "seed", "main", 1337),
            overwrite=ctx.force or boolean(ctx.section, "overwrite", ctx.step, False),
        )
    )
    if not validate_outputs(ctx):
        raise RuntimeError(f"Feature output validation failed for {ctx.step}")
    summary = read_json(_output_file(ctx).with_suffix(".summary.json"))
    return {
        "output_file": str(_output_file(ctx)),
        "feature_count": summary.get("feature_count"),
        "feature_shape": summary.get("feature_shape"),
        "label": integer(ctx.section, "label", ctx.step),
        "split": ctx.section["split"],
    }
