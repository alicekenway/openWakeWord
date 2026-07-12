"""Stage registry for the INI-driven pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import augment, export, feature, summary, testing, train


@dataclass(frozen=True)
class StageHandler:
    validate: Callable[[Any], None]
    input_paths: Callable[[Any], list[Path]]
    output_paths: Callable[[Any], list[Path]]
    validate_outputs: Callable[[Any], bool]
    run: Callable[[Any], dict[str, Any]]


HANDLERS: dict[str, StageHandler] = {
    "augment": StageHandler(
        augment.validate,
        augment.input_paths,
        augment.output_paths,
        augment.validate_outputs,
        augment.run,
    ),
    "feature": StageHandler(
        feature.validate,
        feature.input_paths,
        feature.output_paths,
        feature.validate_outputs,
        feature.run,
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
    ),
    "summary": StageHandler(
        summary.validate,
        summary.input_paths,
        summary.output_paths,
        summary.validate_outputs,
        summary.run,
    ),
}


def handler_for_step(step: str) -> StageHandler:
    prefix = step.split(".", 1)[0]
    if prefix not in HANDLERS:
        raise KeyError(prefix)
    if prefix in {"augment", "feature", "testing"} and "." not in step:
        raise KeyError(f"{prefix} stages require a named sub-block, for example {prefix}.positive_train")
    if prefix in {"train", "export", "summary"} and step != prefix:
        raise KeyError(f"{prefix} must be listed as exactly '{prefix}'")
    return HANDLERS[prefix]
