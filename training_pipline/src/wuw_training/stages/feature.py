"""Feature extraction stage: normalized JSONL audio to openWakeWord NPY arrays."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from ..artifacts import normalise_manifest_inputs, read_json, read_jsonl, write_json, write_jsonl
from ..config import ConfigurationError
from ..ctc_wac import (
    Stage1Contract,
    feature_bundle_paths,
    feature_bundle_valid,
    generate_ctc_wac_feature_bundle,
    load_keywords,
)
from ..legacy import get_legacy_module
from .common import boolean, integer, number, placement, require, stage_work_path


def _inputs(ctx: Any):
    from ..artifacts import parse_manifest_inputs

    return parse_manifest_inputs(ctx.config, ctx.step)


def _output_file(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_file", ctx.step))


def _model_dir(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "model_dir", ctx.step))


def _extractor(ctx: Any) -> str:
    return ctx.section.get("extractor", "openwakeword").strip().lower()


def _stage1_model(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "stage1_model", ctx.step))


def _stage1_contract(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "stage1_contract", ctx.step))


def _keywords(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "keywords", ctx.step))


def _keyword_tokens(ctx: Any) -> Path:
    """Use a token-only file when supplied, otherwise keep the simple form."""

    value = ctx.section.get("keyword_tokens")
    return ctx.config.resolve_path(value) if value else _keywords(ctx)


def _ctc_max_search_frames(ctx: Any, contract: Stage1Contract) -> int | None:
    """Convert an explicitly configured CTC window count into encoder frames.

    Existing feature configurations had no bounded search, so absence of
    ``window_count`` preserves that behavior.  New configurations set both
    ``window_seconds`` (the base streaming window) and ``window_count`` (the
    maximum number of consecutive base windows to inspect).
    """

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


def _validate_common_inputs(ctx: Any) -> None:
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
    # CTC-WAC consumes the already augmented waveform verbatim.  Placement is
    # therefore an augmentation concern, not a feature extraction concern.
    if _extractor(ctx) != "wenet_ctc_wac":
        placement(ctx.section, ctx.step)
    _output_file(ctx)


def validate(ctx: Any) -> None:
    _validate_common_inputs(ctx)
    extractor = _extractor(ctx)
    if extractor == "wenet_ctc_wac":
        model = _stage1_model(ctx)
        contract_path = _stage1_contract(ctx)
        keywords_path = _keyword_tokens(ctx)
        for name, path in (("stage1_model", model), ("stage1_contract", contract_path), ("keyword_tokens", keywords_path)):
            if not path.is_file():
                raise ConfigurationError(f"[{ctx.step}] {name} does not exist: {path}")
        contract = Stage1Contract.from_json(contract_path)
        load_keywords(keywords_path, require_threshold=False)
        configured_rate = integer(ctx.config.section("main"), "sample_rate", "main", contract.sample_rate)
        if configured_rate != contract.sample_rate:
            raise ConfigurationError(
                f"[{ctx.step}] [main] sample_rate={configured_rate} does not match stage-1 contract sample_rate={contract.sample_rate}"
            )
        if integer(ctx.section, "candidate_pre_margin_frames", ctx.step, 3) < 0:
            raise ConfigurationError(f"[{ctx.step}] candidate_pre_margin_frames must be >= 0")
        if integer(ctx.section, "candidate_post_margin_frames", ctx.step, 0) < 0:
            raise ConfigurationError(f"[{ctx.step}] candidate_post_margin_frames must be >= 0")
        _ctc_max_search_frames(ctx, contract)
        device = ctx.section.get("device", "cpu").lower()
        if device not in {"auto", "cpu", "gpu"}:
            raise ConfigurationError(f"[{ctx.step}] device must be auto, cpu, or gpu")
        return
    if extractor != "openwakeword":
        raise ConfigurationError(
            f"[{ctx.step}] extractor must be openwakeword or wenet_ctc_wac, got {extractor!r}"
        )
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
def input_paths(ctx: Any) -> list[Path]:
    paths = [item.path for item in _inputs(ctx)]
    if _extractor(ctx) == "wenet_ctc_wac":
        paths.extend([_stage1_model(ctx), _stage1_contract(ctx), _keyword_tokens(ctx)])
        return paths
    model_dir = _model_dir(ctx)
    paths.extend([model_dir / "melspectrogram.onnx", model_dir / "embedding_model.onnx"])
    return paths


def output_paths(ctx: Any) -> list[Path]:
    output = _output_file(ctx)
    if _extractor(ctx) == "wenet_ctc_wac":
        return feature_bundle_paths(output).all()
    return [output, output.with_suffix(".summary.json")]


def validate_outputs(ctx: Any) -> bool:
    if _extractor(ctx) == "wenet_ctc_wac":
        contract = Stage1Contract.from_json(_stage1_contract(ctx))
        return feature_bundle_valid(
            _output_file(ctx),
            expected_stage1_contract_fingerprint=contract.fingerprint(),
        )
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
        default_placement=ctx.section.get("placement", "end") if _extractor(ctx) == "wenet_ctc_wac" else placement(ctx.section, ctx.step),
        label=integer(ctx.section, "label", ctx.step),
    )
    if _extractor(ctx) == "wenet_ctc_wac":
        summary = generate_ctc_wac_feature_bundle(
            records=read_jsonl(normalized),
            output_file=_output_file(ctx),
            model_path=_stage1_model(ctx),
            contract_path=_stage1_contract(ctx),
            keywords_path=_keyword_tokens(ctx),
            candidate_pre_margin_frames=integer(ctx.section, "candidate_pre_margin_frames", ctx.step, 3),
            candidate_post_margin_frames=integer(ctx.section, "candidate_post_margin_frames", ctx.step, 0),
            max_search_frames=_ctc_max_search_frames(ctx, Stage1Contract.from_json(_stage1_contract(ctx))),
            device=ctx.section.get("device", "cpu").lower(),
            overwrite=ctx.force or boolean(ctx.section, "overwrite", ctx.step, False),
        )
        if not validate_outputs(ctx):
            raise RuntimeError(f"CTC-WAC feature output validation failed for {ctx.step}")
        return {
            "output_file": str(_output_file(ctx)),
            "feature_count": summary.get("feature_count"),
            "feature_shape": summary.get("feature_storage_shape"),
            "keyword_count": summary.get("keyword_count"),
            "stage1_gate_applied": False,
            "label": integer(ctx.section, "label", ctx.step),
            "split": ctx.section["split"],
        }
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


# Slurm helpers intentionally live beside the local implementation.  They use
# the same low-level feature functions, but write isolated shard outputs that
# the controller can validate and merge safely.
def prepare_slurm_shards(ctx: Any, work_dir: Path, task_count: int) -> list[dict[str, Any]]:
    normalized = work_dir / "input.jsonl"
    normalise_manifest_inputs(
        _inputs(ctx),
        normalized,
        default_placement=ctx.section.get("placement", "end") if _extractor(ctx) == "wenet_ctc_wac" else placement(ctx.section, ctx.step),
        label=integer(ctx.section, "label", ctx.step),
    )
    records = read_jsonl(normalized)
    actual_count = min(task_count, len(records))
    if actual_count < 1:
        raise RuntimeError(f"Feature stage {ctx.step} has no records to shard")
    base, extra = divmod(len(records), actual_count)
    result: list[dict[str, Any]] = []
    start = 0
    for task_id in range(actual_count):
        stop = start + base + (1 if task_id < extra else 0)
        shard_dir = work_dir / "shards" / f"{task_id:05d}"
        manifest = shard_dir / "input.jsonl"
        write_jsonl(manifest, records[start:stop])
        result.append(
            {
                "id": task_id,
                "start": start,
                "stop": stop,
                "count": stop - start,
                "input_manifest": str(manifest),
                "normalized_manifest": str(normalized),
                "output_file": str(shard_dir / "features.npy"),
            }
        )
        start = stop
    return result


def run_slurm_shard(ctx: Any, task: dict[str, Any]) -> dict[str, Any]:
    output = Path(str(task["output_file"])).resolve()
    manifest = Path(str(task["input_manifest"])).resolve()
    index_offset = int(task["start"])
    if _extractor(ctx) == "wenet_ctc_wac":
        summary = generate_ctc_wac_feature_bundle(
            records=read_jsonl(manifest),
            output_file=output,
            model_path=_stage1_model(ctx),
            contract_path=_stage1_contract(ctx),
            keywords_path=_keyword_tokens(ctx),
            candidate_pre_margin_frames=integer(ctx.section, "candidate_pre_margin_frames", ctx.step, 3),
            candidate_post_margin_frames=integer(ctx.section, "candidate_post_margin_frames", ctx.step, 0),
            max_search_frames=_ctc_max_search_frames(ctx, Stage1Contract.from_json(_stage1_contract(ctx))),
            device=ctx.section.get("device", "cpu").lower(),
            overwrite=True,
            index_offset=index_offset,
        )
        return {"output_file": str(output), "feature_count": summary.get("feature_count")}

    main = ctx.config.section("main")
    legacy = get_legacy_module()
    legacy.command_generate_features(
        argparse.Namespace(
            audio_manifest=[str(manifest)],
            audio_dir=[],
            output_file=str(output),
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
            overwrite=True,
            index_offset=index_offset,
        )
    )
    summary = read_json(output.with_suffix(".summary.json"))
    return {"output_file": str(output), "feature_count": summary.get("feature_count")}


def validate_slurm_shard(ctx: Any, task: dict[str, Any]) -> bool:
    output = Path(str(task["output_file"])).resolve()
    if _extractor(ctx) == "wenet_ctc_wac":
        contract = Stage1Contract.from_json(_stage1_contract(ctx))
        return feature_bundle_valid(
            output,
            expected_stage1_contract_fingerprint=contract.fingerprint(),
        )
    summary_path = output.with_suffix(".summary.json")
    if not output.is_file() or not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        values = np.load(output, mmap_mode="r")
        return (
            int(summary.get("error_count", -1)) == 0
            and int(summary.get("feature_count", -1)) == int(task["count"])
            and int(values.shape[0]) == int(task["count"])
        )
    except Exception:
        return False


def _atomic_merge_arrays(destination: Path, sources: list[Path]) -> tuple[int, tuple[int, ...], np.dtype[Any]]:
    arrays = [np.load(path, mmap_mode="r") for path in sources]
    if not arrays:
        raise RuntimeError("No feature shard outputs to merge")
    trailing = tuple(int(value) for value in arrays[0].shape[1:])
    dtype = arrays[0].dtype
    if any(tuple(int(value) for value in array.shape[1:]) != trailing or array.dtype != dtype for array in arrays):
        raise RuntimeError(f"Feature shard shapes or dtypes do not match for {destination}")
    count = sum(int(array.shape[0]) for array in arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.slurm.tmp")
    try:
        merged = open_memmap(temporary, mode="w+", dtype=dtype, shape=(count, *trailing))
        row = 0
        for array in arrays:
            next_row = row + int(array.shape[0])
            merged[row:next_row] = array
            row = next_row
        merged.flush()
        del merged
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count, trailing, dtype


def _merge_openwakeword_features(ctx: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    output = _output_file(ctx)
    sources = [Path(str(task["output_file"])).resolve() for task in tasks]
    count, shape, _dtype = _atomic_merge_arrays(output, sources)
    legacy = get_legacy_module()
    normalized = Path(str(tasks[0]["normalized_manifest"])).resolve()
    items = legacy.feature_items_from_feature_inputs(
        [str(normalized)], [], placement(ctx.section, ctx.step)
    )
    shard_summary = read_json(sources[0].with_suffix(".summary.json"))
    summary = {
        "output_file": str(output),
        "input_count": len(items),
        "input_signature": legacy.feature_input_signature(items),
        "feature_count": count,
        "feature_shape": list(shape),
        "clip_seconds": shard_summary.get("clip_seconds"),
        "requested_device": shard_summary.get("requested_device"),
        "device": shard_summary.get("device"),
        "ncpu": shard_summary.get("ncpu"),
        "batch_size": shard_summary.get("batch_size"),
        "audio_loader_workers": shard_summary.get("audio_loader_workers"),
        "prefetch_batches": shard_summary.get("prefetch_batches"),
        "placement": placement(ctx.section, ctx.step),
        "placement_counts": legacy.placement_counts(value for _path, value in items),
        "errors": [],
        "error_count": 0,
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return {
        "output_file": str(output),
        "feature_count": count,
        "feature_shape": list(shape),
        "label": integer(ctx.section, "label", ctx.step),
        "split": ctx.section["split"],
    }


def _merge_ctc_wac_features(ctx: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    output = _output_file(ctx)
    destination = feature_bundle_paths(output)
    source_paths = [feature_bundle_paths(Path(str(task["output_file"])).resolve()) for task in tasks]
    feature_frames, feature_shape, _dtype = _atomic_merge_arrays(
        destination.features, [paths.features for paths in source_paths]
    )
    row_count, _length_shape, _dtype = _atomic_merge_arrays(
        destination.lengths, [paths.lengths for paths in source_paths]
    )
    for name in ("all_scores", "top_score", "margin", "winner_onehot"):
        count_for_array, _trailing, _dtype = _atomic_merge_arrays(
            getattr(destination, name), [getattr(paths, name) for paths in source_paths]
        )
        if count_for_array != row_count:
            raise RuntimeError(f"CTC-WAC shard count mismatch while merging {name}")
    lengths = np.asarray(np.load(destination.lengths, mmap_mode="r"), dtype=np.int64)
    if int(lengths.sum()) != feature_frames:
        raise RuntimeError("CTC-WAC merged candidate lengths do not match flat feature frame count")
    offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])
    temporary_offsets = destination.offsets.with_name(f".{destination.offsets.name}.slurm.tmp")
    try:
        with temporary_offsets.open("wb") as handle:
            np.save(handle, offsets, allow_pickle=False)
        os.replace(temporary_offsets, destination.offsets)
    finally:
        temporary_offsets.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    for paths in source_paths:
        rows.extend(read_jsonl(paths.rows))
    if len(rows) != row_count:
        raise RuntimeError("CTC-WAC shard row metadata count does not match merged candidates")
    for index, row in enumerate(rows):
        row["row"] = index
    write_jsonl(destination.rows, rows)
    summary = read_json(source_paths[0].summary)
    summary.update(
        {
            "output_file": str(output),
            "feature_count": row_count,
            "feature_storage_shape": [feature_frames, *feature_shape],
            "feature_dim": int(feature_shape[0]),
            "candidate_length_frames": {
                "min": int(lengths.min()) if lengths.size else None,
                "max": int(lengths.max()) if lengths.size else None,
            },
            "input_count": sum(int(read_json(paths.summary).get("input_count", 0)) for paths in source_paths),
            "input_duration_seconds": sum(
                float(read_json(paths.summary).get("input_duration_seconds", 0.0)) for paths in source_paths
            ),
            "invalid_alignment_rows": sum(
                int(read_json(paths.summary).get("invalid_alignment_rows", 0)) for paths in source_paths
            ),
            "error_count": 0,
            "errors": [],
        }
    )
    summary.pop("index_offset", None)
    write_json(destination.summary, summary)
    return {
        "output_file": str(output),
        "feature_count": row_count,
        "feature_shape": [int(lengths.max()) if lengths.size else 0, int(feature_shape[0])],
        "keyword_count": summary.get("keyword_count"),
        "stage1_gate_applied": False,
        "label": integer(ctx.section, "label", ctx.step),
        "split": ctx.section["split"],
    }


def merge_slurm_shards(ctx: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    result = _merge_ctc_wac_features(ctx, tasks) if _extractor(ctx) == "wenet_ctc_wac" else _merge_openwakeword_features(ctx, tasks)
    if not validate_outputs(ctx):
        raise RuntimeError(f"Feature merge validation failed for {ctx.step}")
    return result


def cleanup_slurm_shards(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        output = Path(str(task["output_file"])).resolve()
        if output.exists():
            for path in feature_bundle_paths(output).all():
                path.unlink(missing_ok=True)
        output.with_suffix(".progress.json").unlink(missing_ok=True)
