"""Focused regression tests for the INI pipeline package."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wuw_training.artifacts import ManifestInput, normalise_manifest_inputs, write_json, write_jsonl  # noqa: E402
from wuw_training.checkpoints import CheckpointManager  # noqa: E402
from wuw_training.config import load_ini_config  # noqa: E402
from wuw_training.legacy import get_legacy_module  # noqa: E402
from wuw_training.runner import PipelineRunner  # noqa: E402
from wuw_training.stages.testing import _markdown_report, _metric_rows  # noqa: E402
from wuw_training.stages.train import _weighted_probability_bce  # noqa: E402


def _write_test_result(root: Path, name: str, expected_label: int) -> Path:
    output = root / name
    output.mkdir(parents=True)
    windows = [
        {"score": 0.15, "end_time": 0.5},
        {"score": 0.35, "end_time": 1.5},
        {"score": 0.55, "end_time": 2.5},
    ]
    write_jsonl(
        output / "eval_details.jsonl",
        [
            {
                "duration_seconds": 2.0,
                "sliding_windows": windows,
                "expected_label": expected_label,
            }
        ],
    )
    write_json(output / "eval_summary.json", {"sets": {name: {"error_count": 0}}})
    return output


def test_summary_runner_uses_ordered_ini_stage_and_skips_completed_work(tmp_path: Path) -> None:
    positive = _write_test_result(tmp_path, "positive", 1)
    negative = _write_test_result(tmp_path, "negative", 0)
    config_path = tmp_path / "pipeline.ini"
    config_path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}

[steps]
steps = summary

[testing.positive]
expected_label = 1
record_window_scores = yes
output_dir = {positive}

[testing.negative]
expected_label = 0
record_window_scores = yes
output_dir = {negative}

[summary]
tests = testing.positive, testing.negative
threshold_start = 0.1
threshold_stop = 0.5
threshold_step = 0.2
debounce_seconds = 1.0
output_json = ${{main:experiment_dir}}/thresholds.json
output_report = ${{main:experiment_dir}}/REPORT.md
""",
        encoding="utf-8",
    )

    runner = PipelineRunner(load_ini_config(config_path))
    first = runner.run()
    assert first["steps"]["summary"]["status"] == "done"
    payload = json.loads((tmp_path / "experiment" / "thresholds.json").read_text(encoding="utf-8"))
    assert [row["threshold"] for row in payload["thresholds"]] == [0.1, 0.3, 0.5]
    assert payload["thresholds"][0]["sets"]["testing.positive"]["recall"] == 1.0
    assert payload["thresholds"][2]["sets"]["testing.positive"]["false_reject_rate"] == 0.0

    second = runner.run()
    assert second["steps"]["summary"]["status"] == "skipped"


def test_completion_checkpoint_detects_output_changes(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "state")
    output = tmp_path / "artifact.txt"
    output.write_text("first", encoding="utf-8")
    inputs: dict[str, list[Any]] = {"inputs": []}
    manager.mark_complete("summary", fingerprint="fingerprint", input_signature=inputs, outputs=[output], result={})
    complete, reason = manager.is_complete(
        "summary",
        fingerprint="fingerprint",
        input_signature=inputs,
        outputs=[output],
        output_validator=lambda: output.is_file(),
    )
    assert complete, reason
    output.write_text("changed output", encoding="utf-8")
    complete, reason = manager.is_complete(
        "summary",
        fingerprint="fingerprint",
        input_signature=inputs,
        outputs=[output],
        output_validator=lambda: output.is_file(),
    )
    assert not complete
    assert reason == "declared outputs changed"


def test_configparser_interpolates_json_list_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "audio.jsonl"
    write_jsonl(manifest, [{"path": "clip.wav"}])
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        f"""[main]
root = {tmp_path}

[feature.example]
input_jsonl = ["${{main:root}}/audio.jsonl"]
audio_base_dir = ${{main:root}}
label = 1
split = train
output_file = ${{main:root}}/features.npy
model_dir = ${{main:root}}
""",
        encoding="utf-8",
    )
    config = load_ini_config(config_path)
    assert json.loads(config.get("feature.example", "input_jsonl")) == [str(manifest)]


def test_manifest_normalization_replaces_duplicate_audio_paths(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.touch()
    manifest = tmp_path / "input.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "audiofile_path": "old.wav",
                "path": str(audio),
                "source_path": "/old/source.wav",
                "vad_trim": {"source_path": "/older/source.wav", "threshold": 0.5},
                "text": "Hey Siri",
            }
        ],
    )
    output = tmp_path / "normalized.jsonl"

    normalise_manifest_inputs([ManifestInput(manifest, None)], output)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["path"] == str(audio.resolve())
    assert "audiofile_path" not in record
    assert "source_path" not in record
    assert "source_path" not in record["vad_trim"]
    assert record["vad_trim"]["threshold"] == 0.5


def test_probability_bce_disables_autocast_and_uses_float32() -> None:
    predictions = torch.tensor([0.8, 0.2], dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        loss = _weighted_probability_bce(predictions, labels, negative_weight=3.0)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_threshold_metrics_and_markdown_are_per_set() -> None:
    accumulators = [
        {"threshold": 0.1, "false_accept_events": 4, "false_accept_clips": 3, "false_rejects": 0},
        {"threshold": 0.3, "false_accept_events": 2, "false_accept_clips": 2, "false_rejects": 0},
    ]
    rows = _metric_rows(accumulators, expected_label=0, evaluated=10, evaluated_seconds=3600.0)
    assert rows[0]["false_accepts_per_hour"] == 4.0
    assert rows[0]["false_reject_rate"] is None
    report = _markdown_report(
        step="testing.negative",
        model=Path("model.onnx"),
        input_manifests=[Path("negative.jsonl")],
        expected_label=0,
        debounce_seconds=1.0,
        requested=10,
        evaluated=10,
        evaluated_seconds=3600.0,
        errors=[],
        rows=rows,
    )
    assert "| Threshold | FA events | FA clips | FA/hour | False rejects | FR rate |" in report
    assert "| 0.1 | 4 | 3 | 4.000000 | n/a | n/a |" in report


def test_score_details_do_not_contain_threshold_or_abnormal_fields(monkeypatch: Any, tmp_path: Path) -> None:
    legacy = get_legacy_module()
    monkeypatch.setattr(legacy, "load_audio_float", lambda path, sr: torch.zeros(16000))
    monkeypatch.setattr(legacy, "prediction_scores", lambda *args, **kwargs: [0.1, 0.8])
    detail = legacy.evaluate_one_record_scores(
        object(),
        "model",
        {"path": str(tmp_path / "clip.wav")},
        "testing.positive",
        0,
        1,
        {
            "sample_rate": 16000,
            "chunk_size": 1280,
            "positive_padding": 1,
            "negative_padding": 0,
            "model_window_seconds": 2.0,
            "record_window_scores": True,
        },
    )
    for field in ("threshold", "detected", "false_reject", "false_accept", "abnormal", "abnormal_type"):
        assert field not in detail
    assert detail["sliding_windows"]


def test_train_stage_resumes_from_a_model_checkpoint(tmp_path: Path) -> None:
    shape = (16, 96)
    feature_paths: dict[str, Path] = {}
    for name in ("positive_train", "negative_train", "positive_dev", "negative_dev", "background_dev"):
        path = tmp_path / f"{name}.npy"
        np.save(path, np.random.default_rng(7).standard_normal((3, *shape), dtype=np.float32))
        feature_paths[name] = path
    config_path = tmp_path / "train.ini"
    config_path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}
model_name = test_model
sample_rate = 16000
seed = 7

[steps]
steps = train

[feature.positive_train]
output_file = {feature_paths['positive_train']}
label = 1
split = train

[feature.negative_train]
output_file = {feature_paths['negative_train']}
label = 0
split = train

[feature.positive_dev]
output_file = {feature_paths['positive_dev']}
label = 1
split = dev

[feature.negative_dev]
output_file = {feature_paths['negative_dev']}
label = 0
split = dev

[feature.background_dev]
output_file = {feature_paths['background_dev']}
label = 0
split = dev

[train]
train = feature.positive_train, feature.negative_train
dev = feature.positive_dev, feature.negative_dev
false_positive = feature.background_dev
batch.feature.positive_train = 2
batch.feature.negative_train = 2
steps = 2
phase_step_ratios = [1.0]
phase_learning_rates = [0.0001]
validation_points = 1
model_type = dnn
layer_size = 8
max_negative_weight = 2
target_false_positives_per_hour = 1
resume = yes
checkpoint_interval_steps = 1
keep_checkpoints = 2
model_checkpoint_dir = ${{main:experiment_dir}}/model_checkpoints
output_model = ${{main:experiment_dir}}/trained_model/model.pt
output_summary = ${{main:experiment_dir}}/trained_model/training_summary.json
""",
        encoding="utf-8",
    )
    runner = PipelineRunner(load_ini_config(config_path))
    first = runner.run()
    assert first["steps"]["train"]["result"]["resumed"] is False
    training_log = tmp_path / "experiment" / "trained_model" / "training_summary.jsonl"
    events = [json.loads(line) for line in training_log.read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "train_step" and "loss" in event for event in events)
    assert any(event["event"] == "validation" and "val_accuracy" in event for event in events)
    assert events[-1]["event"] == "run_complete"

    checkpoint_path = tmp_path / "experiment" / "model_checkpoints" / "latest.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["phase_index"] = 0
    checkpoint["phase_step"] = 1
    checkpoint["global_step"] = 1
    torch.save(checkpoint, checkpoint_path)
    runner.manager.complete_path("train").unlink()

    resumed = runner.run()
    assert resumed["steps"]["train"]["result"]["resumed"] is True
    assert resumed["steps"]["train"]["result"]["global_steps"] == 2
