"""Validated ordered execution for INI-defined pipeline stages."""

from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import PIPELINE_VERSION
from .artifacts import file_signature, hash_payload, read_json, write_json
from .checkpoints import CheckpointManager, step_slug
from .config import ConfigurationError, IniConfig, parse_step_groups
from .context import StageContext
from .slurm import SlurmExecutor, execution_mode
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
        self.execution_mode = execution_mode(self.config)
        self.slurm = SlurmExecutor(self.config) if self.execution_mode == "slurm" else None
        self._planned: list[PlannedStep] | None = None
        self._planned_groups: list[list[PlannedStep]] | None = None

    def step_groups(self) -> list[list[str]]:
        raw = self.config.get("steps", "steps")
        assert raw is not None
        groups = parse_step_groups(raw)
        values = [value for group in groups for value in group]
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ConfigurationError(f"[steps] contains duplicate stage(s): {', '.join(duplicates)}")
        return groups

    def steps(self) -> list[str]:
        return [value for group in self.step_groups() for value in group]

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
            execution_role="slurm_controller" if self.execution_mode == "slurm" else "controller",
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

        by_name = {item.name: item for item in planned}
        planned_groups = [[by_name[name] for name in group] for group in self.step_groups()]
        seen: set[str] = set()
        for group in planned_groups:
            group_names = {item.name for item in group}
            for item in group:
                ctx = self._context(item.name)
                for input_path in item.handler.input_paths(ctx):
                    input_path = input_path.resolve()
                    producer = output_to_step.get(input_path)
                    if producer is not None:
                        if producer in group_names:
                            raise ConfigurationError(
                                f"Input {input_path} for {item.name} is produced by {producer}, but both are in the "
                                "same parallel group. Move the producer to an earlier group."
                            )
                        if producer not in seen:
                            raise ConfigurationError(
                                f"Input {input_path} for {item.name} is produced by {producer}, which must appear "
                                "in an earlier [steps] group"
                            )
                    elif not input_path.exists():
                        raise ConfigurationError(
                            f"Input for {item.name} does not exist and is not produced by the pipeline: {input_path}"
                        )
            seen.update(group_names)
        self._planned = planned
        self._planned_groups = planned_groups
        return planned

    def plan_groups(self) -> list[list[PlannedStep]]:
        self.plan()
        assert self._planned_groups is not None
        return self._planned_groups

    def validate(self) -> list[PlannedStep]:
        planned = self.plan()
        if self.slurm is not None:
            self.slurm.validate(planned)
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
        section = dict(ctx.section)
        if item.name.startswith("testing."):
            # Threshold reporting belongs to [summary]. Keep obsolete testing
            # options fingerprint-neutral so editing a sweep cannot invalidate
            # expensive model inference in an older configuration.
            for option in (
                "threshold_range",
                "threshold_start",
                "threshold_stop",
                "threshold_step",
                "debounce_seconds",
                "output_report",
                "record_window_scores",
            ):
                section.pop(option, None)
        return hash_payload(
            {
                "pipeline_version": PIPELINE_VERSION,
                "step": item.name,
                "section": section,
            }
        )

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

    def _select(
        self,
        *,
        from_step: str | None,
        to_step: str | None,
        only_step: str | None,
    ) -> list[list[PlannedStep]]:
        planned = self.plan()
        names = [item.name for item in planned]
        if only_step:
            if only_step not in names:
                raise ConfigurationError(f"Step {only_step!r} is not listed in [steps]")
            return [[planned[names.index(only_step)]]]
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
        selected_names = set(names[start:stop])
        return [
            [item for item in group if item.name in selected_names]
            for group in self.plan_groups()
            if any(item.name in selected_names for item in group)
        ]

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

        for group in selected:
            prepared: list[dict[str, Any]] = []
            for item in group:
                input_signature = self._input_signature(item)
                fingerprint = self._fingerprint(item)
                requested_force = item.name in force_steps
                base_ctx = self._context(item.name)
                complete, reason = self.manager.is_complete(
                    item.name,
                    fingerprint=fingerprint,
                    input_signature=input_signature,
                    outputs=list(item.outputs),
                    output_validator=lambda ctx=base_ctx, handler=item.handler: handler.validate_outputs(ctx),
                )
                if complete and not requested_force:
                    status["steps"][item.name] = {"status": "skipped", "reason": reason}
                    print(f"Skipping completed step: {item.name}", flush=True)
                    continue

                # A stale completion record means old artifacts must be regenerated
                # rather than adopted by a legacy helper's shallow output check.
                force = requested_force or (
                    self.manager.read_complete(item.name) is not None and not complete
                )
                ctx = self._context(item.name, force=force)
                for path in item.handler.input_paths(ctx):
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Input for {item.name} is unavailable: {path}. "
                            "Run its prerequisite stage first."
                        )
                status["steps"][item.name] = {"status": "running", "reason": reason}
                prepared.append(
                    {
                        "item": item,
                        "ctx": ctx,
                        "input_signature": input_signature,
                        "fingerprint": fingerprint,
                        "force": force,
                        "started": time.time(),
                    }
                )
            self._write_status(status)
            if not prepared:
                continue

            if len(prepared) > 1:
                names = ", ".join(value["item"].name for value in prepared)
                print(f"\n### Parallel group: {names}", flush=True)

            def execute(value: dict[str, Any]) -> dict[str, Any]:
                item = value["item"]
                ctx = value["ctx"]
                if len(prepared) == 1:
                    print(f"\n### {item.name}", flush=True)
                else:
                    print(f"Starting parallel step: {item.name}", flush=True)
                if self.slurm is not None:
                    return self.slurm.run_stage(
                        name=item.name,
                        handler=item.handler,
                        ctx=ctx,
                        fingerprint=value["fingerprint"],
                        input_signature=value["input_signature"],
                        force=value["force"],
                    )
                return item.handler.run(ctx)

            def finish(
                value: dict[str, Any],
                *,
                result: dict[str, Any] | None = None,
                error: Exception | None = None,
            ) -> Exception | None:
                item = value["item"]
                ctx = value["ctx"]
                elapsed = round(time.time() - float(value["started"]), 6)
                if error is None:
                    try:
                        if result is None:
                            raise RuntimeError(f"Stage {item.name} returned no result")
                        if not item.handler.validate_outputs(ctx):
                            raise RuntimeError(f"Stage output validation failed: {item.name}")
                    except Exception as exc:
                        error = exc
                if error is not None:
                    self.manager.mark_failed(item.name, error)
                    status["steps"][item.name] = {
                        "status": "failed",
                        "elapsed_seconds": elapsed,
                        "error": repr(error),
                    }
                    self._write_status(status)
                    return error
                assert result is not None
                self.manager.mark_complete(
                    item.name,
                    fingerprint=value["fingerprint"],
                    input_signature=value["input_signature"],
                    outputs=list(item.outputs),
                    result=result,
                )
                status["steps"][item.name] = {
                    "status": "done",
                    "elapsed_seconds": elapsed,
                    "result": result,
                }
                self._write_status(status)
                return None

            if len(prepared) == 1:
                value = prepared[0]
                try:
                    error = finish(value, result=execute(value))
                except Exception as exc:
                    error = finish(value, error=exc)
                if error is not None:
                    raise error
                continue

            future_to_value: dict[Future[dict[str, Any]], dict[str, Any]] = {}
            first_error: Exception | None = None
            with ThreadPoolExecutor(max_workers=len(prepared), thread_name_prefix="pipeline-step") as executor:
                for value in prepared:
                    future_to_value[executor.submit(execute, value)] = value
                for future in as_completed(future_to_value):
                    value = future_to_value[future]
                    try:
                        error = finish(value, result=future.result())
                    except Exception as exc:
                        error = finish(value, error=exc)
                    if first_error is None and error is not None:
                        first_error = error

            if first_error is not None:
                # Every already-started member of the group is allowed to finish so
                # successful Slurm jobs remain reusable. The next group never starts.
                raise first_error
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
