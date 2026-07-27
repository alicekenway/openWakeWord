"""Slurm execution backend for the INI training pipeline.

The controller prepares deterministic manifests and submits one extraction
array for each shardable stage. Once the array finishes, a separate Slurm
merge job combines and validates its shards. Workers only write their own
shard data and atomic state files, so a failed array or merge can be resumed
without repeating successful extraction work.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import threading
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
    merge_max_parallel: int

    @classmethod
    def from_config(cls, config: IniConfig) -> "SlurmSettings":
        if not config.has_section("slurm"):
            raise ConfigurationError("Slurm mode requires a [slurm] section")
        sbatch = config.get("slurm", "sbatch_command", required=False, fallback="sbatch") or "sbatch"
        squeue = config.get("slurm", "squeue_command", required=False, fallback="squeue") or "squeue"
        python = config.get("slurm", "python_executable", required=False, fallback=sys.executable) or sys.executable
        setup = config.get("slurm", "setup_commands", required=False, fallback="") or ""
        raw_merge_parallel = config.get("slurm", "merge_max_parallel", required=False, fallback="1") or "1"
        try:
            merge_max_parallel = int(raw_merge_parallel)
        except ValueError as exc:
            raise ConfigurationError("[slurm] merge_max_parallel must be an integer") from exc
        if merge_max_parallel < 1:
            raise ConfigurationError("[slurm] merge_max_parallel must be >= 1")
        return cls(
            sbatch_command=tuple(_command_words(sbatch, "[slurm] sbatch_command")),
            squeue_command=tuple(_command_words(squeue, "[slurm] squeue_command")),
            python_executable=python,
            setup_commands=setup,
            merge_max_parallel=merge_max_parallel,
        )


@dataclass(frozen=True)
class SlurmStepSettings:
    section_name: str
    tasks: int
    sbatch_args: tuple[str, ...]
    merge_sbatch_args: tuple[str, ...]


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
    def parse_sbatch_args(option: str, fallback: str | None = None) -> tuple[str, ...]:
        raw_args = section.get(option, fallback if fallback is not None else "")
        try:
            args = tuple(shlex.split(raw_args))
        except ValueError as exc:
            raise ConfigurationError(
                f"[{section_name}] {option} is not valid shell-style text: {exc}"
            ) from exc
        for token in args:
            if _option_name(token) in RESERVED_SBATCH_OPTIONS:
                raise ConfigurationError(
                    f"[{section_name}] {option} must not set {_option_name(token)}; the pipeline controls it"
                )
        return args

    raw_sbatch_args = section.get("sbatch_args", "")
    return SlurmStepSettings(
        section_name=section_name,
        tasks=tasks,
        sbatch_args=parse_sbatch_args("sbatch_args"),
        merge_sbatch_args=parse_sbatch_args("merge_sbatch_args", raw_sbatch_args),
    )


def _task_path(work_dir: Path, task_id: int, suffix: str) -> Path:
    return work_dir / "tasks" / f"{task_id:05d}.{suffix}.json"


def _merge_state_path(work_dir: Path, suffix: str) -> Path:
    return work_dir / f"merge.{suffix}.json"


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
        self._merge_slots = threading.BoundedSemaphore(self.settings.merge_max_parallel)

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
        # Do not start the legacy wuw_pipeline.py wrapper here.  It imports the
        # original openWakeWord stack before dispatching the INI subcommand,
        # so a CTC-WAC-only Slurm job would incorrectly require legacy-only
        # dependencies such as scipy and torchinfo.
        module_root = Path(__file__).resolve().parents[1]
        static_args = [
            self.settings.python_executable,
            "-m",
            "wuw_training.cli",
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
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"export PYTHONPATH={shlex.quote(str(module_root))}${{PYTHONPATH:+:${{PYTHONPATH}}}}",
        ]
        if self.settings.setup_commands.strip():
            lines.append(self.settings.setup_commands.rstrip())
        lines.append(f'exec {command} --task-id "${{SLURM_ARRAY_TASK_ID:-0}}"')
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o700)
        return script

    def _merge_batch_script(self, work_dir: Path, snapshot: Path, spec: Path) -> Path:
        """Create the single-job entrypoint used to merge completed shards."""

        module_root = Path(__file__).resolve().parents[1]
        static_args = [
            self.settings.python_executable,
            "-m",
            "wuw_training.cli",
            "__slurm-merge",
            "--config",
            str(snapshot),
            "--config-root",
            str(self.config.root),
            "--spec",
            str(spec),
        ]
        command = " ".join(shlex.quote(value) for value in static_args)
        script = work_dir / "run_merge.sh"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"export PYTHONPATH={shlex.quote(str(module_root))}${{PYTHONPATH:+:${{PYTHONPATH}}}}",
        ]
        if self.settings.setup_commands.strip():
            lines.append(self.settings.setup_commands.rstrip())
        lines.append(f"exec {command}")
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o700)
        return script

    def _submit_and_wait(
        self,
        *,
        work_dir: Path,
        job_name: str,
        step_settings: SlurmStepSettings,
        script: Path,
        task_ids: list[int],
        is_array: bool,
        submission_path: Path,
        use_merge_args: bool = False,
    ) -> dict[str, Any]:
        logs = work_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        command = [
            *self.settings.sbatch_command,
            *(step_settings.merge_sbatch_args if use_merge_args else step_settings.sbatch_args),
            "--parsable",
            "--wait",
            f"--job-name={job_name}",
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
        write_json(submission_path, submission)
        remaining_stdout, stderr = process.communicate()
        submission.update(
            {
                "returncode": process.returncode,
                "stdout": first_line + remaining_stdout,
                "stderr": stderr,
            }
        )
        write_json(submission_path, submission)
        return submission

    def _has_active_submission(self, submission_path: Path) -> bool:
        submission = _read_task_state(submission_path)
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

    def _completed_merge_state(
        self,
        work_dir: Path,
        ctx: StageContext,
        handler: StageHandler,
    ) -> dict[str, Any] | None:
        state = _read_task_state(_merge_state_path(work_dir, "done"))
        if state is None or state.get("worker_protocol") != WORKER_PROTOCOL:
            return None
        result = state.get("result")
        if not isinstance(result, dict):
            return None
        try:
            return dict(result) if handler.validate_outputs(ctx) else None
        except Exception:
            return None

    def _merge_failure_message(self, work_dir: Path, submission: dict[str, Any]) -> str:
        failed = _read_task_state(_merge_state_path(work_dir, "failed"))
        detail = failed.get("error", "merge worker did not write a completion marker") if failed else (
            "merge worker did not write a completion marker"
        )
        return (
            f"Slurm merge failed ({detail}). Job {submission.get('job_id') or 'unknown'} "
            f"logs are in {work_dir / 'logs'}"
        )

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
                _merge_state_path(work_dir, "done").unlink(missing_ok=True)
                _merge_state_path(work_dir, "failed").unlink(missing_ok=True)
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
        if sharded:
            assert handler.distributed is not None
            merged = self._completed_merge_state(work_dir, ctx, handler)
            if merged is not None:
                merged["slurm"] = {
                    "requested_tasks": step_settings.tasks,
                    "actual_tasks": len(tasks),
                    "work_dir": str(work_dir),
                    "array_job_id": (_read_task_state(work_dir / "submission.json") or {}).get("job_id"),
                    "merge_job_id": (_read_task_state(work_dir / "merge_submission.json") or {}).get("job_id"),
                }
                return merged

        pending = [
            task for task in tasks if not self._task_is_complete(work_dir, ctx, handler, task, kind=kind)
        ]
        if pending:
            array_submission_path = work_dir / "submission.json"
            if self._has_active_submission(array_submission_path):
                existing = _read_task_state(array_submission_path) or {}
                raise RuntimeError(
                    f"Slurm stage {name} already has active job {existing.get('job_id')}; "
                    "wait for it to finish before starting another controller"
                )
            script = self._batch_script(work_dir, snapshot, spec)
            submission = self._submit_and_wait(
                work_dir=work_dir,
                job_name=f"wuw-{name.replace('.', '-')}",
                step_settings=step_settings,
                script=script,
                task_ids=[int(task["id"]) for task in pending],
                is_array=sharded,
                submission_path=array_submission_path,
            )
            failed = [task for task in tasks if not self._task_is_complete(work_dir, ctx, handler, task, kind=kind)]
            if submission["returncode"] != 0 or failed:
                raise RuntimeError(self._failure_message(work_dir, failed or tasks, submission))
        if sharded:
            assert handler.distributed is not None
            merge_submission_path = work_dir / "merge_submission.json"
            if self._has_active_submission(merge_submission_path):
                existing = _read_task_state(merge_submission_path) or {}
                raise RuntimeError(
                    f"Slurm merge for {name} already has active job {existing.get('job_id')}; "
                    "wait for it to finish before starting another controller"
                )
            merge_script = self._merge_batch_script(work_dir, snapshot, spec)
            with self._merge_slots:
                merge_submission = self._submit_and_wait(
                    work_dir=work_dir,
                    job_name=f"wuw-merge-{name.replace('.', '-')}",
                    step_settings=step_settings,
                    script=merge_script,
                    task_ids=[0],
                    is_array=False,
                    submission_path=merge_submission_path,
                    use_merge_args=True,
                )
            result = self._completed_merge_state(work_dir, ctx, handler)
            if merge_submission["returncode"] != 0 or result is None:
                raise RuntimeError(self._merge_failure_message(work_dir, merge_submission))
            result["slurm"] = {
                "requested_tasks": step_settings.tasks,
                "actual_tasks": len(tasks),
                "work_dir": str(work_dir),
                "array_job_id": (_read_task_state(work_dir / "submission.json") or {}).get("job_id"),
                "merge_job_id": merge_submission.get("job_id"),
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


def run_merge_worker(
    *,
    config_path: str | Path,
    config_root: str | Path,
    spec_path: str | Path,
) -> dict[str, Any]:
    """Merge one completed shard stage inside its own Slurm job."""

    spec = read_json(Path(spec_path))
    if not isinstance(spec, dict) or spec.get("worker_protocol") != WORKER_PROTOCOL:
        raise RuntimeError(f"Unsupported Slurm merge specification: {spec_path}")
    if spec.get("kind") != "shard":
        raise RuntimeError("Slurm merge worker requires a shard-stage specification")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("Slurm merge specification has no task list")
    config = load_ini_config(config_path, base_dir=config_root)
    step = str(spec["step"])
    handler = handler_for_step(step)
    if handler.distributed is None:
        raise RuntimeError(f"Stage {step} does not implement Slurm sharding")
    work_dir = Path(str(spec["stage_work_dir"])).resolve()
    context = StageContext(
        config=config,
        step=step,
        section=config.section(step),
        experiment_dir=Path(str(spec["experiment_dir"])).resolve(),
        work_dir=work_dir,
        force=bool(spec.get("force", False)),
        execution_role="slurm_merge_worker",
    )
    done_path = _merge_state_path(work_dir, "done")
    failed_path = _merge_state_path(work_dir, "failed")
    try:
        handler.validate(context)
        incomplete = [
            int(task.get("id", -1))
            for task in tasks
            if not isinstance(task, dict) or not handler.distributed.validate_shard(context, task)
        ]
        if incomplete:
            raise RuntimeError(f"Cannot merge {step}: incomplete shard task(s) {incomplete[:10]}")
        result = handler.distributed.merge(context, tasks)
        if not handler.validate_outputs(context):
            raise RuntimeError(f"Slurm merge output validation failed for {step}")
        handler.distributed.cleanup(tasks)
        state = {
            "worker_protocol": WORKER_PROTOCOL,
            "step": step,
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
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
