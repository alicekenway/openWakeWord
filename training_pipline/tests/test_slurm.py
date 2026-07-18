"""Tests for the scheduler-independent parts of the Slurm backend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wuw_training.artifacts import write_json, write_jsonl  # noqa: E402
from wuw_training.config import ConfigurationError, load_ini_config  # noqa: E402
from wuw_training.context import StageContext  # noqa: E402
from wuw_training.slurm import (  # noqa: E402
    WORKER_PROTOCOL,
    SlurmExecutor,
    _step_settings,
    _task_path,
    run_worker,
)
from wuw_training.stages import DistributedStageHandler, StageHandler  # noqa: E402
from wuw_training.stages.feature import prepare_slurm_shards  # noqa: E402


def _slurm_config(tmp_path: Path, *, step: str = "feature.demo", tasks: int = 2, args: str = "") -> Path:
    path = tmp_path / "pipeline.ini"
    path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}
execution_mode = slurm

[slurm]
python_executable = {sys.executable}

[slurm.{step}]
tasks = {tasks}
sbatch_args = {args}
""",
        encoding="utf-8",
    )
    return path


def test_slurm_step_settings_validate_task_count_and_owned_options(tmp_path: Path) -> None:
    config = load_ini_config(_slurm_config(tmp_path, step="train", tasks=2))
    with pytest.raises(ConfigurationError, match="tasks must be omitted or 1"):
        _step_settings(config, "train")

    config = load_ini_config(_slurm_config(tmp_path, args="--mem=12G --partition=gpu"))
    settings = _step_settings(config, "feature.demo")
    assert settings.tasks == 2
    assert settings.sbatch_args == ("--mem=12G", "--partition=gpu")

    config = load_ini_config(_slurm_config(tmp_path, args="--array=0-1"))
    with pytest.raises(ConfigurationError, match="must not set --array"):
        _step_settings(config, "feature.demo")


def test_feature_shards_are_contiguous_and_cap_tasks_at_record_count(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    write_jsonl(manifest, [{"path": f"clip-{index}.wav"} for index in range(3)])
    config_path = tmp_path / "feature.ini"
    config_path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}

[feature.demo]
input_jsonl = {manifest}
label = 1
split = train
placement = random
output_file = {tmp_path / 'output.npy'}
model_dir = {tmp_path}
""",
        encoding="utf-8",
    )
    config = load_ini_config(config_path)
    context = StageContext(
        config=config,
        step="feature.demo",
        section=config.section("feature.demo"),
        experiment_dir=tmp_path / "experiment",
        work_dir=tmp_path / "work",
    )
    tasks = prepare_slurm_shards(context, context.work_dir, 10)

    assert [(task["id"], task["start"], task["stop"], task["count"]) for task in tasks] == [
        (0, 0, 1, 1),
        (1, 1, 2, 1),
        (2, 2, 3, 1),
    ]
    assert all(Path(str(task["input_manifest"])).is_file() for task in tasks)


def test_slurm_executor_reuses_completed_shards_after_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_ini_config(_slurm_config(tmp_path, tasks=2))
    executor = SlurmExecutor(config)
    context = StageContext(
        config=config,
        step="feature.demo",
        section={},
        experiment_dir=tmp_path / "experiment",
        work_dir=tmp_path / "experiment" / ".pipeline_work" / "feature_demo",
        execution_role="slurm_controller",
    )
    submitted: list[list[int]] = []

    def prepare(_ctx: Any, work_dir: Path, task_count: int) -> list[dict[str, Any]]:
        assert task_count == 2
        return [
            {"id": index, "output": str(work_dir / "shards" / f"{index}.txt")}
            for index in range(task_count)
        ]

    def validate_shard(_ctx: Any, task: dict[str, Any]) -> bool:
        return Path(str(task["output"])).is_file()

    def merge(ctx: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        values = [Path(str(task["output"])).read_text(encoding="utf-8") for task in tasks]
        final = ctx.experiment_dir / "final.txt"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_text("".join(values), encoding="utf-8")
        return {"final": str(final)}

    distributed = DistributedStageHandler(
        prepare=prepare,
        run_shard=lambda _ctx, _task: {},
        validate_shard=validate_shard,
        merge=merge,
        cleanup=lambda _tasks: None,
    )
    handler = StageHandler(
        validate=lambda _ctx: None,
        input_paths=lambda _ctx: [],
        output_paths=lambda _ctx: [],
        validate_outputs=lambda ctx: (ctx.experiment_dir / "final.txt").is_file(),
        run=lambda _ctx: {},
        distributed=distributed,
    )

    monkeypatch.setattr(executor, "_ensure_commands", lambda: None)

    def fake_submit(**kwargs: Any) -> dict[str, Any]:
        work_dir = Path(str(kwargs["work_dir"]))
        task_ids = list(kwargs["task_ids"])
        submitted.append(task_ids)
        successful = [0] if len(submitted) == 1 else task_ids
        for task_id in successful:
            output = work_dir / "shards" / f"{task_id}.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(str(task_id), encoding="utf-8")
            write_json(
                _task_path(work_dir, task_id, "done"),
                {"worker_protocol": WORKER_PROTOCOL, "task_id": task_id, "result": {}},
            )
            _task_path(work_dir, task_id, "failed").unlink(missing_ok=True)
        if len(submitted) == 1:
            write_json(
                _task_path(work_dir, 1, "failed"),
                {"worker_protocol": WORKER_PROTOCOL, "task_id": 1, "error": "synthetic failure"},
            )
            return {"returncode": 1, "job_id": "1"}
        return {"returncode": 0, "job_id": "2"}

    monkeypatch.setattr(executor, "_submit_and_wait", fake_submit)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        executor.run_stage(
            name="feature.demo",
            handler=handler,
            ctx=context,
            fingerprint="fingerprint",
            input_signature={"inputs": []},
            force=False,
        )

    result = executor.run_stage(
        name="feature.demo",
        handler=handler,
        ctx=context,
        fingerprint="fingerprint",
        input_signature={"inputs": []},
        force=False,
    )
    assert submitted == [[0, 1], [1]]
    assert Path(str(result["final"])).read_text(encoding="utf-8") == "01"


def test_slurm_worker_uses_original_config_directory_and_writes_done_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _slurm_config(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + "\n[feature.demo]\n", encoding="utf-8")
    config = load_ini_config(source)
    snapshot = tmp_path / "work" / "config.resolved.ini"
    config.write_resolved(snapshot)
    work_dir = tmp_path / "work"
    output = tmp_path / "experiment" / "worker-output.txt"
    spec = work_dir / "spec.json"
    write_json(
        spec,
        {
            "worker_protocol": WORKER_PROTOCOL,
            "kind": "whole",
            "step": "feature.demo",
            "experiment_dir": str(tmp_path / "experiment"),
            "stage_work_dir": str(work_dir),
            "force": False,
            "tasks": [{"id": 0}],
        },
    )

    def run(ctx: Any) -> dict[str, Any]:
        assert ctx.config.root == source.parent
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("done", encoding="utf-8")
        return {"output": str(output)}

    handler = StageHandler(
        validate=lambda _ctx: None,
        input_paths=lambda _ctx: [],
        output_paths=lambda _ctx: [output],
        validate_outputs=lambda _ctx: output.is_file(),
        run=run,
    )
    monkeypatch.setattr("wuw_training.slurm.handler_for_step", lambda _step: handler)

    result = run_worker(
        config_path=snapshot,
        config_root=source.parent,
        spec_path=spec,
        task_id=0,
    )
    assert result["result"] == {"output": str(output)}
    assert _task_path(work_dir, 0, "done").is_file()
