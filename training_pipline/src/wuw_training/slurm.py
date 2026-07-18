"""Slurm execution backend for the INI training pipeline.

The controller stays on the submission host.  It prepares deterministic
manifests, submits one array for a shardable stage, waits for every array
element, and performs the final merge/checkpoint locally.  Workers only write
their own shard data and an atomic task state file, so a failed array can be
resumed without repeating successful elements.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import hash_payload, read_json, write_json
from .config import ConfigurationError, IniConfig, load_ini_config
from .context import StageContext
from .stages import StageHandler, handler_for_step


WORKER_PROTOCOL = 1
SHARDED_PREFIXES = {"augment", "feature", "testing"}
RESERVED_SBATCH_OPTIONS = {
    "--array",
    "-a",
    "--wait",
    "-W",
    "--parsable",
    "--output",
    "-o",
    "--error",
    "-e",
    "--job-name",
    "-J",
    "--chdir",
    "-D",
    "--wrap",
}


def execution_mode(config: IniConfig) -> str:
    value = config.get("main", "execution_mode", required=False, fallback="local") or "local"
    normalized = value.strip().lower()
    if normalized not in {"local", "slurm"}:
        raise ConfigurationError("[main] execution_mode must be local or slurm")
    return normalized


def _command_words(value: str, field: str) -> list[str]:
    try:
        values = shlex.split(value)
    except ValueError as exc:
        raise ConfigurationError(f"{field} is not valid shell-style text: {exc}") from exc
    if not values:
        raise ConfigurationError(f"{field} cannot be empty")
    return values


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


@dataclass(frozen=True)
class SlurmSettings:
    sbatch_command: tuple[str, ...]
    squeue_command: tuple[str, ...]
    python_executable: str
    setup_commands: str

    @classmethod
    def from_config(cls, config: IniConfig) -> "SlurmSettings":
        if not config.has_section("slurm"):
            raise ConfigurationError("Slurm mode requires a [slurm] section")
        sbatch = config.get("slurm", "sbatch_command", required=False, fallback="sbatch") or "sbatch"
        squeue = config.get("slurm", "squeue_command", required=False, fallback="squeue") or "squeue"
        python = config.get("slurm", "python_executable", required=False, fallback=sys.executable) or sys.executable
        setup = config.get("slurm", "setup_commands", required=False, fallback="") or ""
        return cls(
            sbatch_command=tuple(_command_words(sbatch, "[slurm] sbatch_command")),
            squeue_command=tuple(_command_words(squeue, "[slurm] squeue_command")),
            python_executable=python,
            setup_commands=setup,
        )


@dataclass(frozen=True)
class SlurmStepSettings:
    section_name: str
    tasks: int
    sbatch_args: tuple[str, ...]


def _step_settings(config: IniConfig, step: str) -> SlurmStepSettings:
    section_name = f"slurm.{step}"
    if not config.has_section(section_name):
        raise ConfigurationError(f"Slurm mode requires a [{section_name}] section")
    section = config.section(section_name)
    prefix = step.split(".", 1)[0]
    raw_tasks = section.get("tasks")
    if raw_tasks is None:
        tasks = 1
    else:
        try:
            tasks = int(raw_tasks)
        except ValueError as exc:
            raise ConfigurationError(f"[{section_name}] tasks must be an integer, got {raw_tasks!r}") from exc
    if tasks < 1:
        raise ConfigurationError(f"[{section_name}] tasks must be >= 1")
    if prefix not in SHARDED_PREFIXES and tasks != 1:
        raise ConfigurationError(f"[{section_name}] tasks must be omitted or 1 for the {step} stage")
    raw_args = section.get("sbatch_args", "")
    try:
        args = tuple(shlex.split(raw_args))
    except ValueError as exc:
        raise ConfigurationError(f"[{section_name}] sbatch_args is not valid shell-style text: {exc}") from exc
    for token in args:
        if _option_name(token) in RESERVED_SBATCH_OPTIONS:
            raise ConfigurationError(
                f"[{section_name}] sbatch_args must not set {_option_name(token)}; the pipeline controls it"
            )
    return SlurmStepSettings(section_name=section_name, tasks=tasks, sbatch_args=args)


def _task_path(work_dir: Path, task_id: int, suffix: str) -> Path:
    return work_dir / "tasks" / f"{task_id:05d}.{suffix}.json"


def _read_task_state(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


class SlurmExecutor:
    def __init__(self, config: IniConfig):
        self.config = config
        self.settings = SlurmSettings.from_config(config)

    def validate(self, steps: Iterable[Any]) -> None:
        for item in steps:
            _step_settings(self.config, item.name)

    def _ensure_commands(self) -> None:
        command = self.settings.sbatch_command[0]
        if shutil.which(command) is None and not Path(command).is_file():
            raise RuntimeError(
                f"Slurm mode needs {command!r} on the submission host; set [slurm] sbatch_command if needed"
            )

    def _work_dir(
        self,
        ctx: StageContext,
        *,
        fingerprint: str,
        input_signature: dict[str, Any],
        requested_tasks: int,
    ) -> Path:
        identity = hash_payload(
            {
                "worker_protocol": WORKER_PROTOCOL,
                "step": ctx.step,
                "fingerprint": fingerprint,
                "input_signature": input_signature,
                "main": self.config.section("main"),
                "requested_tasks": requested_tasks,
            }
        )
        return ctx.work_dir / "slurm" / identity

    def _write_snapshot(self, work_dir: Path) -> Path:
        snapshot = work_dir / "config.resolved.ini"
        self.config.write_resolved(snapshot)
        return snapshot

    def _write_spec(
        self,
        *,
        work_dir: Path,
        snapshot: Path,
        ctx: StageContext,
        fingerprint: str,
        input_signature: dict[str, Any],
        kind: str,
        tasks: list[dict[str, Any]],
        force: bool,
    ) -> Path:
        spec = {
            "worker_protocol": WORKER_PROTOCOL,
            "kind": kind,
            "step": ctx.step,
            "experiment_dir": str(ctx.experiment_dir),
            "stage_work_dir": str(work_dir),
            "config_path": str(snapshot),
            "config_base_dir": str(self.config.root),
            "fingerprint": fingerprint,
            "input_signature": input_signature,
            "force": bool(force),
            "tasks": tasks,
        }
        path = work_dir / "spec.json"
        write_json(path, spec)
        return path

    def _task_is_complete(
        self,
        work_dir: Path,
        ctx: StageContext,
        handler: StageHandler,
        task: dict[str, Any],
        *,
        kind: str,
    ) -> bool:
        state = _read_task_state(_task_path(work_dir, int(task["id"]), "done"))
        if state is None or state.get("worker_protocol") != WORKER_PROTOCOL:
            return False
        try:
            if kind == "shard":
                return bool(handler.distributed and handler.distributed.validate_shard(ctx, task))
            return bool(handler.validate_outputs(ctx))
        except Exception:
            return False

    def _batch_script(self, work_dir: Path, snapshot: Path, spec: Path) -> Path:
        entrypoint = Path(__file__).resolve().parents[1] / "wuw_pipeline.py"
        static_args = [
            self.settings.python_executable,
            str(entrypoint),
            "__slurm-worker",
            "--config",
            str(snapshot),
            "--config-root",
            str(self.config.root),
            "--spec",
            str(spec),
        ]
        command = " ".join(shlex.quote(value) for value in static_args)
        script = work_dir / "run_worker.sh"
        lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
        if self.settings.setup_commands.strip():
            lines.append(self.settings.setup_commands.rstrip())
        lines.append(f'exec {command} --task-id "${{SLURM_ARRAY_TASK_ID:-0}}"')
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o700)
        return script

    def _submit_and_wait(
        self,
        *,
        work_dir: Path,
        step: str,
        step_settings: SlurmStepSettings,
        script: Path,
        task_ids: list[int],
        is_array: bool,
    ) -> dict[str, Any]:
        logs = work_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        command = [
            *self.settings.sbatch_command,
            *step_settings.sbatch_args,
            "--parsable",
            "--wait",
            f"--job-name=wuw-{step.replace('.', '-')}",
            f"--output={logs}/%x_%A_%a.out",
            f"--error={logs}/%x_%A_%a.err",
        ]
        if is_array:
            command.append("--array=" + ",".join(str(value) for value in task_ids))
        command.append(str(script))
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        first_line = process.stdout.readline()
        job_id = first_line.strip().split(";", 1)[0] if first_line.strip() else None
        submission = {
            "command": command,
            "job_id": job_id,
            "task_ids": task_ids,
            "returncode": None,
            "stdout": first_line,
            "stderr": "",
        }
        write_json(work_dir / "submission.json", submission)
        remaining_stdout, stderr = process.communicate()
        submission.update(
            {
                "returncode": process.returncode,
                "stdout": first_line + remaining_stdout,
                "stderr": stderr,
            }
        )
        write_json(work_dir / "submission.json", submission)
        return submission

    def _has_active_submission(self, work_dir: Path) -> bool:
        submission = _read_task_state(work_dir / "submission.json")
        if not submission or submission.get("returncode") is not None:
            return False
        job_id = submission.get("job_id")
        if not job_id:
            return False
        command = [*self.settings.squeue_command, "--noheader", "--jobs", str(job_id)]
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
        except OSError:
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _failure_message(self, work_dir: Path, tasks: list[dict[str, Any]], submission: dict[str, Any]) -> str:
        details: list[str] = []
        for task in tasks:
            task_id = int(task["id"])
            failed = _read_task_state(_task_path(work_dir, task_id, "failed"))
            if failed is not None:
                details.append(f"task {task_id}: {failed.get('error', 'worker failed')}")
            elif not _task_path(work_dir, task_id, "done").exists():
                details.append(f"task {task_id}: no completion marker")
        suffix = "; ".join(details) if details else "scheduler returned a failure status"
        return (
            f"Slurm stage failed after all submitted tasks finished ({suffix}). "
            f"Job {submission.get('job_id') or 'unknown'} logs are in {work_dir / 'logs'}"
        )

    def run_stage(
        self,
        *,
        name: str,
        handler: StageHandler,
        ctx: StageContext,
        fingerprint: str,
        input_signature: dict[str, Any],
        force: bool,
    ) -> dict[str, Any]:
        self._ensure_commands()
        step_settings = _step_settings(self.config, name)
        sharded = name.split(".", 1)[0] in SHARDED_PREFIXES
        if sharded and handler.distributed is None:
            raise RuntimeError(f"Stage {name} does not implement Slurm sharding")
        work_dir = self._work_dir(
            ctx,
            fingerprint=fingerprint,
            input_signature=input_signature,
            requested_tasks=step_settings.tasks,
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._write_snapshot(work_dir)
        if sharded:
            tasks = handler.distributed.prepare(ctx, work_dir, step_settings.tasks)
            kind = "shard"
        else:
            tasks = [{"id": 0}]
            kind = "whole"
        if force:
            for task in tasks:
                _task_path(work_dir, int(task["id"]), "done").unlink(missing_ok=True)
                _task_path(work_dir, int(task["id"]), "failed").unlink(missing_ok=True)
            if sharded and handler.distributed is not None:
                handler.distributed.cleanup(tasks)
        spec = self._write_spec(
            work_dir=work_dir,
            snapshot=snapshot,
            ctx=ctx,
            fingerprint=fingerprint,
            input_signature=input_signature,
            kind=kind,
            tasks=tasks,
            force=force,
        )
        pending = [
            task for task in tasks if not self._task_is_complete(work_dir, ctx, handler, task, kind=kind)
        ]
        if pending:
            if self._has_active_submission(work_dir):
                existing = _read_task_state(work_dir / "submission.json") or {}
                raise RuntimeError(
                    f"Slurm stage {name} already has active job {existing.get('job_id')}; "
                    "wait for it to finish before starting another controller"
                )
            script = self._batch_script(work_dir, snapshot, spec)
            submission = self._submit_and_wait(
                work_dir=work_dir,
                step=name,
                step_settings=step_settings,
                script=script,
                task_ids=[int(task["id"]) for task in pending],
                is_array=sharded,
            )
            failed = [task for task in tasks if not self._task_is_complete(work_dir, ctx, handler, task, kind=kind)]
            if submission["returncode"] != 0 or failed:
                raise RuntimeError(self._failure_message(work_dir, failed or tasks, submission))
        if sharded:
            assert handler.distributed is not None
            result = handler.distributed.merge(ctx, tasks)
            handler.distributed.cleanup(tasks)
            result["slurm"] = {
                "requested_tasks": step_settings.tasks,
                "actual_tasks": len(tasks),
                "work_dir": str(work_dir),
            }
            return result
        state = _read_task_state(_task_path(work_dir, 0, "done"))
        if state is None or not handler.validate_outputs(ctx):
            raise RuntimeError(f"Slurm stage {name} completed without valid outputs")
        result = state.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Slurm stage {name} completion marker has no result")
        result = dict(result)
        result["slurm"] = {
            "requested_tasks": 1,
            "actual_tasks": 1,
            "work_dir": str(work_dir),
        }
        return result


def run_worker(
    *,
    config_path: str | Path,
    config_root: str | Path,
    spec_path: str | Path,
    task_id: int,
) -> dict[str, Any]:
    spec = read_json(Path(spec_path))
    if not isinstance(spec, dict) or spec.get("worker_protocol") != WORKER_PROTOCOL:
        raise RuntimeError(f"Unsupported Slurm worker specification: {spec_path}")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("Slurm worker specification has no task list")
    task = next((value for value in tasks if isinstance(value, dict) and int(value.get("id", -1)) == task_id), None)
    if task is None:
        raise RuntimeError(f"Slurm worker task {task_id} is not listed in {spec_path}")
    config = load_ini_config(config_path, base_dir=config_root)
    step = str(spec["step"])
    handler = handler_for_step(step)
    work_dir = Path(str(spec["stage_work_dir"])).resolve()
    context = StageContext(
        config=config,
        step=step,
        section=config.section(step),
        experiment_dir=Path(str(spec["experiment_dir"])).resolve(),
        work_dir=work_dir,
        force=bool(spec.get("force", False)),
        execution_role="slurm_worker",
    )
    done_path = _task_path(work_dir, task_id, "done")
    failed_path = _task_path(work_dir, task_id, "failed")
    try:
        handler.validate(context)
        if spec.get("kind") == "shard":
            if handler.distributed is None:
                raise RuntimeError(f"Stage {step} does not implement Slurm sharding")
            result = handler.distributed.run_shard(context, task)
            valid = handler.distributed.validate_shard(context, task)
        elif spec.get("kind") == "whole":
            result = handler.run(context)
            valid = handler.validate_outputs(context)
        else:
            raise RuntimeError(f"Unknown Slurm worker kind {spec.get('kind')!r}")
        if not valid:
            raise RuntimeError(f"Slurm worker output validation failed for {step} task {task_id}")
        state = {
            "worker_protocol": WORKER_PROTOCOL,
            "step": step,
            "task_id": task_id,
            "result": result,
        }
        write_json(done_path, state)
        failed_path.unlink(missing_ok=True)
        return state
    except Exception as exc:
        write_json(
            failed_path,
            {
                "worker_protocol": WORKER_PROTOCOL,
                "step": step,
                "task_id": task_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
