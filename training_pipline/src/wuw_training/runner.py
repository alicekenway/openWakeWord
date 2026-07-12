"""Validated ordered execution for INI-defined pipeline stages."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import PIPELINE_VERSION
from .artifacts import file_signature, hash_payload, read_json, write_json
from .checkpoints import CheckpointManager, step_slug
from .config import ConfigurationError, IniConfig, parse_csv
from .context import StageContext
from .stages import StageHandler, handler_for_step


@dataclass(frozen=True)
class PlannedStep:
    name: str
    handler: StageHandler
    outputs: tuple[Path, ...]


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


class PipelineRunner:
    def __init__(self, config: IniConfig):
        self.config = config
        self.config.require_section("main")
        self.config.require_section("steps")
        main = self.config.section("main")
        if not main.get("experiment_dir"):
            raise ConfigurationError("Missing required option [main] experiment_dir")
        self.experiment_dir = self.config.resolve_path(main["experiment_dir"])
        checkpoint_value = main.get("pipeline_checkpoint_dir")
        self.checkpoint_dir = (
            self.config.resolve_path(checkpoint_value)
            if checkpoint_value
            else self.experiment_dir / "pipeline_state"
        )
        self.manager = CheckpointManager(self.checkpoint_dir)
        self._planned: list[PlannedStep] | None = None

    def steps(self) -> list[str]:
        raw = self.config.get("steps", "steps")
        assert raw is not None
        values = parse_csv(raw)
        if not values:
            raise ConfigurationError("[steps] steps cannot be empty")
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ConfigurationError(f"[steps] contains duplicate stage(s): {', '.join(duplicates)}")
        return values

    def _context(self, step: str, *, force: bool = False) -> StageContext:
        if not self.config.has_section(step):
            raise ConfigurationError(f"Stage {step!r} is listed in [steps] but has no [{step}] section")
        return StageContext(
            config=self.config,
            step=step,
            section=self.config.section(step),
            experiment_dir=self.experiment_dir,
            work_dir=self.experiment_dir / ".pipeline_work" / step_slug(step),
            force=force,
        )

    def plan(self) -> list[PlannedStep]:
        if self._planned is not None:
            return self._planned
        planned: list[PlannedStep] = []
        output_to_step: dict[Path, str] = {}
        for step in self.steps():
            try:
                handler = handler_for_step(step)
            except KeyError as exc:
                raise ConfigurationError(f"Unknown pipeline stage {step!r}: {exc}") from exc
            ctx = self._context(step)
            handler.validate(ctx)
            outputs = tuple(path.resolve() for path in handler.output_paths(ctx))
            if not outputs:
                raise ConfigurationError(f"Stage {step} does not declare an output")
            for output in outputs:
                if output in output_to_step:
                    raise ConfigurationError(
                        f"Stages {output_to_step[output]!r} and {step!r} both write {output}; outputs must be unique"
                    )
                parent = _nearest_existing_parent(output.parent)
                if parent.exists() and not os.access(parent, os.W_OK):
                    raise ConfigurationError(f"Output parent is not writable for {step}: {parent}")
                output_to_step[output] = step
            planned.append(PlannedStep(step, handler, outputs))

        seen: set[str] = set()
        for item in planned:
            ctx = self._context(item.name)
            for input_path in item.handler.input_paths(ctx):
                input_path = input_path.resolve()
                if input_path.exists():
                    continue
                producer = output_to_step.get(input_path)
                if producer is None:
                    raise ConfigurationError(f"Input for {item.name} does not exist and is not produced by the pipeline: {input_path}")
                if producer not in seen:
                    raise ConfigurationError(
                        f"Input {input_path} for {item.name} is produced by {producer}, which must appear earlier in [steps]"
                    )
            seen.add(item.name)
        self._planned = planned
        return planned

    def validate(self) -> list[PlannedStep]:
        planned = self.plan()
        return planned

    def _input_signature(self, item: PlannedStep) -> dict[str, Any]:
        ctx = self._context(item.name)
        entries: list[dict[str, Any]] = []
        producer_by_path = {path: step.name for step in self.plan() for path in step.outputs}
        for path in item.handler.input_paths(ctx):
            resolved = path.resolve()
            entry = file_signature(resolved)
            producer = producer_by_path.get(resolved)
            if producer:
                entry["producer"] = producer
                entry["producer_checkpoint"] = self.manager.completion_digest(producer)
            entries.append(entry)
        return {"inputs": entries}

    def _fingerprint(self, item: PlannedStep) -> str:
        ctx = self._context(item.name)
        return hash_payload({"pipeline_version": PIPELINE_VERSION, "step": item.name, "section": ctx.section})

    def _status_path(self) -> Path:
        return self.experiment_dir / "experiment_status.json"

    def _load_status(self) -> dict[str, Any]:
        path = self._status_path()
        if path.exists():
            try:
                value = read_json(path)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        return {"pipeline": "ini", "steps": {}}

    def _write_status(self, status: dict[str, Any]) -> None:
        status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(self._status_path(), status)

    def _select(self, *, from_step: str | None, to_step: str | None, only_step: str | None) -> list[PlannedStep]:
        planned = self.plan()
        names = [item.name for item in planned]
        if only_step:
            if only_step not in names:
                raise ConfigurationError(f"Step {only_step!r} is not listed in [steps]")
            return [planned[names.index(only_step)]]
        start = 0
        stop = len(planned)
        if from_step:
            if from_step not in names:
                raise ConfigurationError(f"--from {from_step!r} is not in [steps]")
            start = names.index(from_step)
        if to_step:
            if to_step not in names:
                raise ConfigurationError(f"--to {to_step!r} is not in [steps]")
            stop = names.index(to_step) + 1
        if start >= stop:
            raise ConfigurationError("--from must not come after --to")
        return planned[start:stop]

    def run(
        self,
        *,
        from_step: str | None = None,
        to_step: str | None = None,
        only_step: str | None = None,
        force_steps: set[str] | None = None,
    ) -> dict[str, Any]:
        self.validate()
        force_steps = force_steps or set()
        selected = self._select(from_step=from_step, to_step=to_step, only_step=only_step)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config.write_resolved(self.experiment_dir / "config.resolved.ini")
        status = self._load_status()
        status.update(
            {
                "pipeline": "ini",
                "config": str(self.config.path),
                "experiment_dir": str(self.experiment_dir),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "steps": status.get("steps", {}),
            }
        )

        for item in selected:
            input_signature = self._input_signature(item)
            fingerprint = self._fingerprint(item)
            requested_force = item.name in force_steps
            base_ctx = self._context(item.name)
            complete, reason = self.manager.is_complete(
                item.name,
                fingerprint=fingerprint,
                input_signature=input_signature,
                outputs=list(item.outputs),
                output_validator=lambda: item.handler.validate_outputs(base_ctx),
            )
            if complete and not requested_force:
                status["steps"][item.name] = {"status": "skipped", "reason": reason}
                self._write_status(status)
                print(f"Skipping completed step: {item.name}", flush=True)
                continue

            # A stale completion record means old artifacts must be regenerated rather
            # than adopted by a legacy helper's own shallow output check.
            force = requested_force or (self.manager.read_complete(item.name) is not None and not complete)
            ctx = self._context(item.name, force=force)
            for path in item.handler.input_paths(ctx):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Input for {item.name} is unavailable: {path}. Run its prerequisite stage first."
                    )
            status["steps"][item.name] = {"status": "running", "reason": reason}
            self._write_status(status)
            started = time.time()
            print(f"\n### {item.name}", flush=True)
            try:
                result = item.handler.run(ctx)
                if not item.handler.validate_outputs(ctx):
                    raise RuntimeError(f"Stage output validation failed: {item.name}")
                self.manager.mark_complete(
                    item.name,
                    fingerprint=fingerprint,
                    input_signature=input_signature,
                    outputs=list(item.outputs),
                    result=result,
                )
                status["steps"][item.name] = {
                    "status": "done",
                    "elapsed_seconds": round(time.time() - started, 6),
                    "result": result,
                }
                self._write_status(status)
            except Exception as exc:
                self.manager.mark_failed(item.name, exc)
                status["steps"][item.name] = {
                    "status": "failed",
                    "elapsed_seconds": round(time.time() - started, 6),
                    "error": repr(exc),
                }
                self._write_status(status)
                raise
        status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_status(status)
        return status

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {"config": str(self.config.path), "experiment_dir": str(self.experiment_dir), "steps": {}}
        for item in self.plan():
            ctx = self._context(item.name)
            complete, reason = self.manager.is_complete(
                item.name,
                fingerprint=self._fingerprint(item),
                input_signature=self._input_signature(item),
                outputs=list(item.outputs),
                output_validator=lambda: item.handler.validate_outputs(ctx),
            )
            result["steps"][item.name] = {"status": "complete" if complete else "pending_or_stale", "reason": reason}
        return result
