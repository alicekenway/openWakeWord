"""Stage registry for the INI-driven pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import augment, export, feature, stage1_report, summary, testing, train


@dataclass(frozen=True)
class StageHandler:
    validate: Callable[[Any], None]
    input_paths: Callable[[Any], list[Path]]
    output_paths: Callable[[Any], list[Path]]
    validate_outputs: Callable[[Any], bool]
    run: Callable[[Any], dict[str, Any]]
    distributed: "DistributedStageHandler | None" = None


@dataclass(frozen=True)
class DistributedStageHandler:
    """Stage-specific pieces needed by the Slurm job-array backend."""

    prepare: Callable[[Any, Path, int], list[dict[str, Any]]]
    run_shard: Callable[[Any, dict[str, Any]], dict[str, Any]]
    validate_shard: Callable[[Any, dict[str, Any]], bool]
    merge: Callable[[Any, list[dict[str, Any]]], dict[str, Any]]
    cleanup: Callable[[list[dict[str, Any]]], None]


HANDLERS: dict[str, StageHandler] = {
    "augment": StageHandler(
        augment.validate,
        augment.input_paths,
        augment.output_paths,
        augment.validate_outputs,
        augment.run,
        DistributedStageHandler(
            augment.prepare_slurm_shards,
            augment.run_slurm_shard,
            augment.validate_slurm_shard,
            augment.merge_slurm_shards,
            augment.cleanup_slurm_shards,
        ),
    ),
    "feature": StageHandler(
        feature.validate,
        feature.input_paths,
        feature.output_paths,
        feature.validate_outputs,
        feature.run,
        DistributedStageHandler(
            feature.prepare_slurm_shards,
            feature.run_slurm_shard,
            feature.validate_slurm_shard,
            feature.merge_slurm_shards,
            feature.cleanup_slurm_shards,
        ),
    ),
    "train": StageHandler(
        train.validate,
        train.input_paths,
        train.output_paths,
        train.validate_outputs,
        train.run,
    ),
    "export": StageHandler(
        export.validate,
        export.input_paths,
        export.output_paths,
        export.validate_outputs,
        export.run,
    ),
    "testing": StageHandler(
        testing.validate,
        testing.input_paths,
        testing.output_paths,
        testing.validate_outputs,
        testing.run,
        DistributedStageHandler(
            testing.prepare_slurm_shards,
            testing.run_slurm_shard,
            testing.validate_slurm_shard,
            testing.merge_slurm_shards,
            testing.cleanup_slurm_shards,
        ),
    ),
    "summary": StageHandler(
        summary.validate,
        summary.input_paths,
        summary.output_paths,
        summary.validate_outputs,
        summary.run,
    ),
    "stage1_report": StageHandler(
        stage1_report.validate,
        stage1_report.input_paths,
        stage1_report.output_paths,
        stage1_report.validate_outputs,
        stage1_report.run,
    ),
}


def handler_for_step(step: str) -> StageHandler:
    prefix = step.split(".", 1)[0]
    if prefix not in HANDLERS:
        raise KeyError(prefix)
    if prefix in {"augment", "feature", "testing"} and "." not in step:
        raise KeyError(f"{prefix} stages require a named sub-block, for example {prefix}.positive_train")
    if prefix in {"train", "export", "summary", "stage1_report"} and step != prefix:
        raise KeyError(f"{prefix} must be listed as exactly '{prefix}'")
    return HANDLERS[prefix]
