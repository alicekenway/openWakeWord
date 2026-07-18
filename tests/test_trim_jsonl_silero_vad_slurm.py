"""Tests for Slurm preparation and merge of Silero VAD JSONL shards."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "data_processing_tools" / "trim_jsonl_silero_vad_slurm.py"
SPEC = importlib.util.spec_from_file_location("trim_jsonl_silero_vad_slurm", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
slurm_trim = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = slurm_trim
SPEC.loader.exec_module(slurm_trim)


def controller_args(monkeypatch: pytest.MonkeyPatch, input_jsonl: Path, output_dir: Path) -> object:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trim_jsonl_silero_vad_slurm.py",
            "--input-jsonl",
            str(input_jsonl),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "test-run",
            "--array-tasks",
            "3",
            "--prepare-only",
        ],
    )
    return slurm_trim.parse_args()


def test_prepare_spec_writes_one_small_manifest_per_array_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text(
        "\n".join(json.dumps({"path": f"clips/{index}.wav"}) for index in range(5)) + "\n",
        encoding="utf-8",
    )
    args = controller_args(monkeypatch, input_jsonl, tmp_path / "output")

    spec_path, spec = slurm_trim.prepare_spec(args)

    assert spec["requested_array_tasks"] == 3
    assert spec["array_task_count"] == 3
    assert [task["count"] for task in spec["tasks"]] == [2, 2, 1]
    assert spec_path.is_file()
    first_shard = slurm_trim.read_jsonl(Path(spec["tasks"][0]["input_jsonl"]))
    assert [row["_vad_slurm_index"] for row in first_shard] == [0, 1]
    assert Path(spec["tasks"][1]["trim_output_dir"]) == tmp_path / "output" / "wav" / "00001"


def test_merge_rewrites_internal_absolute_paths_for_final_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    work_dir = output_dir / ".slurm_vad" / "test-run"
    first_jsonl = work_dir / "shards" / "00000" / "output.jsonl"
    second_jsonl = work_dir / "shards" / "00001" / "output.jsonl"
    first_input = work_dir / "shards" / "00000" / "input.jsonl"
    second_input = work_dir / "shards" / "00001" / "input.jsonl"
    output_jsonl = output_dir / "metadata.jsonl"
    first_audio = output_dir / "wav" / "00000" / "first.wav"
    second_audio = output_dir / "wav" / "00001" / "second.wav"
    slurm_trim.write_jsonl(
        first_jsonl,
        [{"path": str(first_audio), "_vad_slurm_index": 1, "text": "second in source"}],
    )
    slurm_trim.write_jsonl(
        second_jsonl,
        [{"path": str(second_audio), "_vad_slurm_index": 0, "text": "first in source"}],
    )
    slurm_trim.write_jsonl(first_input, [{"path": "first.wav"}])
    slurm_trim.write_jsonl(second_input, [{"path": "second.wav"}])
    for path in (first_jsonl, second_jsonl):
        slurm_trim.write_json(
            path.with_suffix(".summary.json"),
            {
                "input_rows": 1,
                "written_rows": 1,
                "skipped_rows": 0,
                "no_speech_rows": 0,
                "error_count": 0,
                "source_duration_seconds": 2.0,
                "trimmed_duration_seconds": 1.5,
            },
        )
    spec_path = work_dir / "spec.json"
    slurm_trim.write_json(
        spec_path,
        {
            "run_id": "test-run",
            "input_jsonl": str(tmp_path / "input.jsonl"),
            "output_dir": str(output_dir),
            "output_jsonl": str(output_jsonl),
            "wav_dir_name": "wav",
            "requested_array_tasks": 3,
            "array_task_count": 2,
            "trim_options": {
                "audio_base_dir": str(tmp_path),
                "sample_rate": 16000,
                "threshold": 0.5,
                "frame_ms": 30.0,
                "pad_ms": 100.0,
                "pre_pad_ms": None,
                "post_pad_ms": None,
                "vad_model": str(tmp_path / "silero_vad.onnx"),
                "vad_threads": 1,
                "workers": "1",
                "no_speech_policy": "copy",
                "overwrite": False,
                "absolute_paths": False,
            },
            "tasks": [
                {"id": 0, "input_jsonl": str(first_input), "output_jsonl": str(first_jsonl)},
                {"id": 1, "input_jsonl": str(second_input), "output_jsonl": str(second_jsonl)},
            ],
        },
    )

    summary = slurm_trim.merge_spec(spec_path)

    merged = slurm_trim.read_jsonl(output_jsonl)
    assert [record["text"] for record in merged] == ["first in source", "second in source"]
    assert [record["path"] for record in merged] == ["wav/00001/second.wav", "wav/00000/first.wav"]
    assert all("_vad_slurm_index" not in record for record in merged)
    assert summary["written_rows"] == 2
    assert summary["removed_duration_seconds"] == 1.0
    assert summary["slurm"]["temporary_shard_jsonls_removed"] == 6
    assert not first_jsonl.exists()
    assert not second_jsonl.exists()
    assert not first_jsonl.with_suffix(".summary.json").exists()
    assert not second_jsonl.with_suffix(".summary.json").exists()
    assert not first_input.exists()
    assert not second_input.exists()


def test_split_caps_array_tasks_at_the_number_of_records(tmp_path: Path) -> None:
    tasks = slurm_trim.split_records(
        [{"path": "one.wav"}, {"path": "two.wav"}],
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        wav_dir_name="wav",
        requested_tasks=10,
    )

    assert len(tasks) == 2
    assert [task["count"] for task in tasks] == [1, 1]


def test_array_tasks_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(slurm_trim.ARRAY_TASKS_ENV, "3")
    assert slurm_trim.array_tasks(None) == 3
