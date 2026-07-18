#!/usr/bin/env python3
"""Submit Slurm-array Silero VAD trimming jobs for a JSONL manifest.

Each array task runs the existing ``trim_jsonl_silero_vad.py`` behavior over
one small manifest.  A dependent merge job combines successful shard manifests
into the normal output JSONL after the array completes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
TRIMMER_PATH = SCRIPT_PATH.with_name("trim_jsonl_silero_vad.py")
DEFAULT_ARRAY_TASKS = 1
ARRAY_TASKS_ENV = "VAD_SLURM_ARRAY_TASKS"
RESERVED_SBATCH_OPTIONS = {
    "--array",
    "-a",
    "--dependency",
    "-d",
    "--parsable",
    "--job-name",
    "-J",
    "--output",
    "-o",
    "--error",
    "-e",
    "--wrap",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} of {path} must be an object")
            records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def array_tasks(value: int | None) -> int:
    if value is None:
        raw_value = os.environ.get(ARRAY_TASKS_ENV, str(DEFAULT_ARRAY_TASKS))
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{ARRAY_TASKS_ENV} must be a positive integer, got {raw_value!r}") from exc
    if value < 1:
        raise ValueError("--array-tasks must be >= 1")
    return value


def parse_sbatch_args(value: str, option_name: str) -> list[str]:
    try:
        parsed = shlex.split(value)
    except ValueError as exc:
        raise ValueError(f"{option_name} is not valid shell-style text: {exc}") from exc
    for item in parsed:
        option = item.split("=", 1)[0]
        if option in RESERVED_SBATCH_OPTIONS:
            raise ValueError(f"{option_name} must not set {option}; this tool controls it")
    return parsed


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"


def split_records(
    records: list[dict[str, Any]],
    *,
    work_dir: Path,
    output_dir: Path,
    wav_dir_name: str,
    requested_tasks: int,
) -> list[dict[str, Any]]:
    task_count = min(requested_tasks, len(records))
    rows_per_task, remainder = divmod(len(records), task_count)
    tasks: list[dict[str, Any]] = []
    start = 0
    for task_id in range(task_count):
        count = rows_per_task + int(task_id < remainder)
        stop = start + count
        shard_records = []
        for index, record in enumerate(records[start:stop], start=start):
            updated = dict(record)
            updated["_vad_slurm_index"] = index
            shard_records.append(updated)
        shard_dir = work_dir / "shards" / f"{task_id:05d}"
        input_jsonl = shard_dir / "input.jsonl"
        output_jsonl = shard_dir / "output.jsonl"
        trim_output_dir = output_dir / wav_dir_name / f"{task_id:05d}"
        write_jsonl(input_jsonl, shard_records)
        tasks.append(
            {
                "id": task_id,
                "start": start,
                "stop": stop,
                "count": len(shard_records),
                "input_jsonl": str(input_jsonl),
                "output_jsonl": str(output_jsonl),
                "trim_output_dir": str(trim_output_dir),
            }
        )
        start = stop
    return tasks


def trim_options(args: argparse.Namespace, *, workers: str) -> dict[str, Any]:
    return {
        "audio_base_dir": str(Path(args.audio_base_dir).expanduser().resolve()),
        "sample_rate": int(args.sample_rate),
        "threshold": float(args.threshold),
        "frame_ms": float(args.frame_ms),
        "pad_ms": float(args.pad_ms),
        "pre_pad_ms": args.pre_pad_ms,
        "post_pad_ms": args.post_pad_ms,
        "vad_model": str(Path(args.vad_model).expanduser().resolve()),
        "vad_threads": int(args.vad_threads),
        "workers": workers,
        "no_speech_policy": str(args.no_speech_policy),
        "overwrite": bool(args.overwrite),
        "absolute_paths": bool(args.absolute_paths),
    }


def prepare_spec(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else output_dir / "metadata.jsonl"
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_jsonl}")
    if not TRIMMER_PATH.is_file():
        raise FileNotFoundError(f"Base VAD trimmer does not exist: {TRIMMER_PATH}")
    records = read_jsonl(input_jsonl)
    if not records:
        raise ValueError(f"Input JSONL has no non-empty records: {input_jsonl}")
    requested_tasks = array_tasks(args.array_tasks)
    run_id = args.run_id or new_run_id()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain only letters, numbers, dot, underscore, and hyphen")
    work_dir = output_dir / ".slurm_vad" / run_id
    if work_dir.exists():
        raise FileExistsError(f"Slurm VAD work directory already exists: {work_dir}; choose a different --run-id")
    workers = str(args.workers or os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    tasks = split_records(
        records,
        work_dir=work_dir,
        output_dir=output_dir,
        wav_dir_name=args.wav_dir_name,
        requested_tasks=requested_tasks,
    )
    spec = {
        "schema_version": 1,
        "run_id": run_id,
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "output_jsonl": str(output_jsonl),
        "wav_dir_name": str(args.wav_dir_name),
        "requested_array_tasks": requested_tasks,
        "array_task_count": len(tasks),
        "trim_options": trim_options(args, workers=workers),
        "tasks": tasks,
    }
    spec_path = work_dir / "spec.json"
    write_json(spec_path, spec)
    return spec_path, spec


def write_batch_scripts(spec_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    work_dir = spec_path.parent
    python = str(Path(args.python).expanduser().resolve()) if args.python else sys.executable
    if not Path(python).is_file():
        raise FileNotFoundError(f"Python executable does not exist: {python}")
    array_script = work_dir / "run_array_task.sh"
    merge_script = work_dir / "merge.sh"
    command = " ".join(shlex.quote(value) for value in (python, str(SCRIPT_PATH), "--worker-spec", str(spec_path)))
    array_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f'exec {command} --task-id "${{SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}}"\n',
        encoding="utf-8",
    )
    merge_command = " ".join(shlex.quote(value) for value in (python, str(SCRIPT_PATH), "--merge-spec", str(spec_path)))
    merge_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + merge_command + "\n", encoding="utf-8")
    array_script.chmod(0o700)
    merge_script.chmod(0o700)
    return array_script, merge_script


def submit_sbatch(command: list[str]) -> tuple[str, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cannot submit Slurm job; command not found: {command[0]}") from exc
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"sbatch submission failed: {message}")
    match = re.search(r"(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(f"Cannot determine Slurm job id from sbatch output: {result.stdout!r}")
    return match.group(1), result.stdout.strip()


def submit_jobs(spec_path: Path, spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if shutil.which(args.sbatch) is None and not Path(args.sbatch).is_file():
        raise RuntimeError(f"Cannot submit Slurm job; sbatch command not found: {args.sbatch}")
    array_script, merge_script = write_batch_scripts(spec_path, args)
    work_dir = spec_path.parent
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    common_args = parse_sbatch_args(args.sbatch_args, "--sbatch-args")
    array_command = [
        args.sbatch,
        *common_args,
        "--parsable",
        f"--job-name={args.job_name}",
        f"--array=0-{len(spec['tasks']) - 1}",
        f"--output={logs_dir}/array_%A_%a.out",
        f"--error={logs_dir}/array_%A_%a.err",
        str(array_script),
    ]
    array_job_id, array_stdout = submit_sbatch(array_command)
    submission: dict[str, Any] = {
        "array_job_id": array_job_id,
        "array_command": array_command,
        "array_stdout": array_stdout,
        "merge_job_id": None,
        "merge_command": None,
        "merge_stdout": None,
    }
    write_json(work_dir / "submission.json", submission)
    if not args.no_merge_job:
        merge_args = parse_sbatch_args(args.merge_sbatch_args or args.sbatch_args, "--merge-sbatch-args")
        merge_command = [
            args.sbatch,
            *merge_args,
            "--parsable",
            f"--job-name={args.job_name}-merge",
            f"--dependency=afterok:{array_job_id}",
            f"--output={logs_dir}/merge_%j.out",
            f"--error={logs_dir}/merge_%j.err",
            str(merge_script),
        ]
        merge_job_id, merge_stdout = submit_sbatch(merge_command)
        submission.update(
            {
                "merge_job_id": merge_job_id,
                "merge_command": merge_command,
                "merge_stdout": merge_stdout,
            }
        )
    write_json(work_dir / "submission.json", submission)
    return submission


def load_trimmer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trim_jsonl_silero_vad", TRIMMER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base VAD trimmer: {TRIMMER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_worker(spec_path: Path, task_id: int) -> dict[str, Any]:
    spec = read_json(spec_path)
    tasks = spec.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Slurm VAD spec has no task list: {spec_path}")
    task = next((item for item in tasks if int(item["id"]) == task_id), None)
    if task is None:
        raise ValueError(f"Task id {task_id} is not present in {spec_path}")
    options = dict(spec["trim_options"])
    options.update(
        {
            "input_jsonl": str(task["input_jsonl"]),
            "output_dir": str(task["trim_output_dir"]),
            "wav_dir_name": "",
            "output_jsonl": str(task["output_jsonl"]),
            # Shards use absolute paths internally; merge_spec converts them
            # to paths relative to the final output JSONL when requested.
            "absolute_paths": True,
        }
    )
    worker_args = SimpleNamespace(**options)
    trimmer = load_trimmer()
    summary = trimmer.run(worker_args)
    if int(summary.get("error_count", 0)) != 0:
        raise RuntimeError(f"Shard {task_id} completed with {summary['error_count']} VAD error(s)")
    return summary


def final_audio_path(path_value: str, output_jsonl: Path, *, absolute_paths: bool) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"Shard output path must be absolute: {path_value}")
    if absolute_paths:
        return str(path)
    try:
        return str(path.relative_to(output_jsonl.parent))
    except ValueError:
        return str(path)


def cleanup_shard_jsonls(tasks: list[dict[str, Any]]) -> int:
    """Remove temporary task manifests only after a successful final merge."""

    removed = 0
    for task in tasks:
        output_jsonl = Path(str(task["output_jsonl"])).resolve()
        input_jsonl = Path(str(task["input_jsonl"])).resolve()
        for path in (input_jsonl, output_jsonl, output_jsonl.with_suffix(".summary.json")):
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def merge_spec(spec_path: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    output_jsonl = Path(str(spec["output_jsonl"])).resolve()
    options = dict(spec["trim_options"])
    tasks = list(spec["tasks"])
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: int(item["id"])):
        shard_jsonl = Path(str(task["output_jsonl"])).resolve()
        shard_summary_path = shard_jsonl.with_suffix(".summary.json")
        if not shard_jsonl.is_file() or not shard_summary_path.is_file():
            raise FileNotFoundError(f"Shard {task['id']} did not produce its JSONL and summary")
        shard_summary = read_json(shard_summary_path)
        if int(shard_summary.get("error_count", -1)) != 0:
            raise RuntimeError(f"Shard {task['id']} has VAD errors and cannot be merged")
        summaries.append(shard_summary)
        records.extend(read_jsonl(shard_jsonl))
    records.sort(key=lambda record: int(record.get("_vad_slurm_index", -1)))
    for record in records:
        record.pop("_vad_slurm_index", None)
        record["path"] = final_audio_path(
            str(record["path"]),
            output_jsonl,
            absolute_paths=bool(options["absolute_paths"]),
        )
    written_rows = write_jsonl(output_jsonl, records)
    submission_path = spec_path.parent / "submission.json"
    submission = read_json(submission_path) if submission_path.is_file() else {}
    summary = {
        "input_jsonl": spec["input_jsonl"],
        "audio_base_dir": options["audio_base_dir"],
        "output_dir": spec["output_dir"],
        "wav_dir": str(Path(str(spec["output_dir"])) / str(spec["wav_dir_name"])),
        "wav_dir_name": spec["wav_dir_name"],
        "output_jsonl": str(output_jsonl),
        "sample_rate": options["sample_rate"],
        "threshold": options["threshold"],
        "frame_ms": options["frame_ms"],
        "pad_ms": options["pad_ms"],
        "pre_pad_ms": options["pre_pad_ms"] if options["pre_pad_ms"] is not None else options["pad_ms"],
        "post_pad_ms": options["post_pad_ms"] if options["post_pad_ms"] is not None else options["pad_ms"],
        "vad_model": options["vad_model"],
        "vad_threads": options["vad_threads"],
        "workers": options["workers"],
        "input_rows": sum(int(item.get("input_rows", 0)) for item in summaries),
        "written_rows": written_rows,
        "skipped_rows": sum(int(item.get("skipped_rows", 0)) for item in summaries),
        "no_speech_rows": sum(int(item.get("no_speech_rows", 0)) for item in summaries),
        "error_count": 0,
        "source_duration_seconds": round(sum(float(item.get("source_duration_seconds", 0.0)) for item in summaries), 6),
        "trimmed_duration_seconds": round(sum(float(item.get("trimmed_duration_seconds", 0.0)) for item in summaries), 6),
        "removed_duration_seconds": round(
            max(
                0.0,
                sum(float(item.get("source_duration_seconds", 0.0)) for item in summaries)
                - sum(float(item.get("trimmed_duration_seconds", 0.0)) for item in summaries),
            ),
            6,
        ),
        "no_speech_policy": options["no_speech_policy"],
        "absolute_paths": options["absolute_paths"],
        "overwrite": options["overwrite"],
        "errors": [],
        "skipped": [],
        "slurm": {
            "run_id": spec["run_id"],
            "requested_array_tasks": spec["requested_array_tasks"],
            "array_task_count": len(tasks),
            "array_job_id": submission.get("array_job_id"),
            "merge_job_id": submission.get("merge_job_id"),
            "work_dir": str(spec_path.parent),
        },
    }
    write_json(output_jsonl.with_suffix(".summary.json"), summary)
    summary["slurm"]["temporary_shard_jsonls_removed"] = cleanup_shard_jsonls(tasks)
    write_json(output_jsonl.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Slurm-array Silero VAD trimming jobs for a JSONL file.")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--merge-spec", help=argparse.SUPPRESS)
    parser.add_argument("--task-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--input-jsonl", help="Input JSONL with audio path fields")
    parser.add_argument("--audio-base-dir", default=".", help="Base directory for relative audio paths")
    parser.add_argument("--output-dir", help="Output directory for WAVs, shard JSONLs, and final metadata")
    parser.add_argument("--wav-dir-name", default="wav", help="Trimmed WAV subdirectory name under output-dir")
    parser.add_argument("--output-jsonl", help="Final merged JSONL path. Default: output-dir/metadata.jsonl")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output WAV sample rate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Silero VAD speech threshold")
    parser.add_argument("--frame-ms", type=float, default=30.0, help="VAD frame size in milliseconds")
    parser.add_argument("--pad-ms", type=float, default=100.0, help="Default audio to keep before/after speech")
    parser.add_argument("--pre-pad-ms", type=float, help="Audio to keep before first speech frame")
    parser.add_argument("--post-pad-ms", type=float, help="Audio to keep after last speech frame")
    parser.add_argument(
        "--vad-model",
        default=str(SCRIPT_PATH.parents[1] / "openwakeword" / "resources" / "models" / "silero_vad.onnx"),
        help="Path to silero_vad.onnx",
    )
    parser.add_argument("--vad-threads", type=int, default=1, help="ONNX Runtime threads per trim worker")
    parser.add_argument(
        "--workers",
        "--max-workers",
        help="Trim worker processes per array task; default: $SLURM_CPUS_PER_TASK or 1",
    )
    parser.add_argument("--no-speech-policy", choices=["copy", "skip", "error"], default="copy")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute audio paths in final JSONL")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing trimmed WAV files")
    parser.add_argument(
        "--array-tasks",
        type=int,
        help=f"Requested Slurm array-task count. Default: ${ARRAY_TASKS_ENV} or {DEFAULT_ARRAY_TASKS}",
    )
    parser.add_argument("--run-id", help="Unique work-directory name under output-dir/.slurm_vad")
    parser.add_argument("--sbatch", default="sbatch", help="sbatch command path")
    parser.add_argument("--sbatch-args", default="", help="Extra sbatch arguments for the array and merge jobs")
    parser.add_argument("--merge-sbatch-args", help="Extra sbatch arguments for only the merge job")
    parser.add_argument("--job-name", default="silero-vad-trim", help="Base Slurm job name")
    parser.add_argument("--python", help="Python executable used in array and merge jobs")
    parser.add_argument("--prepare-only", action="store_true", help="Write shards and scripts but do not call sbatch")
    parser.add_argument("--no-merge-job", action="store_true", help="Submit only the array; run --merge-spec manually later")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_spec:
        if args.task_id is None:
            raise ValueError("--worker-spec requires --task-id")
        run_worker(Path(args.worker_spec).expanduser().resolve(), args.task_id)
        return
    if args.merge_spec:
        merge_spec(Path(args.merge_spec).expanduser().resolve())
        return
    if not args.input_jsonl or not args.output_dir:
        raise ValueError("--input-jsonl and --output-dir are required when submitting jobs")
    spec_path, spec = prepare_spec(args)
    array_script, merge_script = write_batch_scripts(spec_path, args)
    result: dict[str, Any] = {
        "spec": str(spec_path),
        "array_script": str(array_script),
        "merge_script": str(merge_script),
        "array_task_count": len(spec["tasks"]),
        "requested_array_tasks": spec["requested_array_tasks"],
    }
    if not args.prepare_only:
        result["submission"] = submit_jobs(spec_path, spec, args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
