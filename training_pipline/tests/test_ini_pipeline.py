"""Focused regression tests for the INI pipeline package."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wuw_training.artifacts import ManifestInput, normalise_manifest_inputs, write_json, write_jsonl  # noqa: E402
from wuw_training.checkpoints import CheckpointManager  # noqa: E402
from wuw_training.config import ConfigurationError, load_ini_config, parse_step_groups  # noqa: E402
from wuw_training.legacy import get_legacy_module  # noqa: E402
from wuw_training.runner import PipelineRunner  # noqa: E402
from wuw_training.stages import StageHandler  # noqa: E402
from wuw_training.stages.summary import _metrics as _summary_metrics  # noqa: E402
from wuw_training.stages.train import _weighted_probability_bce  # noqa: E402
import wuw_training.runner as runner_module  # noqa: E402


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
    write_json(
        output / "eval_summary.json",
        {
            "schema_version": 1,
            "test_step": f"testing.{name}",
            "expected_label": expected_label,
            "clips_evaluated": 1,
            "error_count": 0,
        },
    )
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
    assert payload["thresholds"][0]["sets"]["testing.negative"]["false_accept_rate"] == 1.0
    assert payload["thresholds"][0]["combined_negative"]["false_accept_rate"] == 1.0
    report = (tmp_path / "experiment" / "REPORT.md").read_text(encoding="utf-8")
    assert "Combined negative FA crop rate" in report
    assert payload["thresholds"][2]["sets"]["testing.negative"]["false_accept_crops"] == 1
    assert payload["thresholds"][2]["sets"]["testing.negative"]["crops_evaluated"] == 3
    assert payload["thresholds"][2]["sets"]["testing.negative"]["false_accept_rate"] == pytest.approx(1 / 3)

    second = runner.run()
    assert second["steps"]["summary"]["status"] == "skipped"


def test_summary_can_reuse_old_details_without_inference_summary(tmp_path: Path) -> None:
    positive = _write_test_result(tmp_path, "positive", 1)
    (positive / "eval_summary.json").unlink()
    config_path = tmp_path / "summary_only.ini"
    config_path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}

[steps]
steps = summary

[testing.positive]
expected_label = 1
output_dir = {positive}

[summary]
tests = testing.positive
threshold_range = [0.2, 0.4]
threshold_step = 0.2
output_json = ${{main:experiment_dir}}/thresholds.json
output_report = ${{main:experiment_dir}}/REPORT.md
""",
        encoding="utf-8",
    )

    result = PipelineRunner(load_ini_config(config_path)).run()

    assert result["steps"]["summary"]["status"] == "done"
    payload = json.loads((tmp_path / "experiment" / "thresholds.json").read_text(encoding="utf-8"))
    assert [item["threshold"] for item in payload["thresholds"]] == [0.2, 0.4]


def test_step_groups_parse_parallel_brackets() -> None:
    assert parse_step_groups("step1,\nstep2,\nstep3") == [["step1"], ["step2"], ["step3"]]
    assert parse_step_groups("[step1.1, step1.2,\nstep1.3], step2") == [
        ["step1.1", "step1.2", "step1.3"],
        ["step2"],
    ]
    with pytest.raises(ConfigurationError, match="unclosed"):
        parse_step_groups("[step1, step2")


def test_parallel_step_group_finishes_before_next_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first.done"
    second = tmp_path / "second.done"
    final = tmp_path / "final.done"
    barrier = threading.Barrier(2)

    def output_paths(ctx: Any) -> list[Path]:
        return [ctx.config.resolve_path(ctx.section["output"])]

    def input_paths(ctx: Any) -> list[Path]:
        return [first, second] if ctx.step == "summary" else []

    def run_stage(ctx: Any) -> dict[str, Any]:
        output = output_paths(ctx)[0]
        if ctx.step.startswith("testing."):
            barrier.wait(timeout=2.0)
        else:
            assert first.is_file() and second.is_file()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(ctx.step, encoding="utf-8")
        return {"output": str(output)}

    handler = StageHandler(
        validate=lambda ctx: None,
        input_paths=input_paths,
        output_paths=output_paths,
        validate_outputs=lambda ctx: output_paths(ctx)[0].is_file(),
        run=run_stage,
    )
    monkeypatch.setattr(runner_module, "handler_for_step", lambda step: handler)
    config_path = tmp_path / "parallel.ini"
    config_path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}

[steps]
steps = [testing.one, testing.two], summary

[testing.one]
output = {first}

[testing.two]
output = {second}

[summary]
output = {final}
""",
        encoding="utf-8",
    )

    result = PipelineRunner(load_ini_config(config_path)).run()

    assert result["steps"]["testing.one"]["status"] == "done"
    assert result["steps"]["testing.two"]["status"] == "done"
    assert result["steps"]["summary"]["status"] == "done"
    assert final.read_text(encoding="utf-8") == "summary"


def test_parallel_group_rejects_an_internal_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    produced = tmp_path / "produced"
    handler = StageHandler(
        validate=lambda ctx: None,
        input_paths=lambda ctx: [produced] if ctx.step == "testing.consumer" else [],
        output_paths=lambda ctx: [produced if ctx.step == "testing.producer" else tmp_path / "consumed"],
        validate_outputs=lambda ctx: False,
        run=lambda ctx: {},
    )
    monkeypatch.setattr(runner_module, "handler_for_step", lambda step: handler)
    config_path = tmp_path / "invalid_parallel.ini"
    config_path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}

[steps]
steps = [testing.producer, testing.consumer]

[testing.producer]

[testing.consumer]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="same parallel group"):
        PipelineRunner(load_ini_config(config_path)).validate()


def test_testing_threshold_options_do_not_change_inference_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "scores.jsonl"
    handler = StageHandler(
        validate=lambda ctx: None,
        input_paths=lambda ctx: [],
        output_paths=lambda ctx: [output],
        validate_outputs=lambda ctx: output.is_file(),
        run=lambda ctx: {},
    )
    monkeypatch.setattr(runner_module, "handler_for_step", lambda step: handler)
    config_path = tmp_path / "fingerprint.ini"
    config_path.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}

[steps]
steps = testing.example

[testing.example]
output = {output}
threshold_range = [0.1, 0.9]
threshold_step = 0.1
""",
        encoding="utf-8",
    )
    config = load_ini_config(config_path)
    runner = PipelineRunner(config)
    item = runner.plan()[0]
    original = runner._fingerprint(item)

    config.parser.set("testing.example", "threshold_range", "[0.4, 0.8]")
    assert runner._fingerprint(item) == original

    config.parser.set("testing.example", "audio_window_seconds", "5.12")
    assert runner._fingerprint(item) != original


def test_ctc_summary_fa_rate_uses_audio_crops() -> None:
    records = [
        {
            "duration_seconds": 10.0,
            "audio_window_count": 4,
            "stage1_candidates": [
                {"score": 0.8, "end_time": 2.0, "audio_window_index": 1},
                {"score": 0.9, "end_time": 2.2, "audio_window_index": 1},
            ],
        }
    ]

    metrics = _summary_metrics(records, 0, 0.5, 1.0, 0)

    assert metrics["clips_evaluated"] == 1
    assert metrics["crops_evaluated"] == 4
    assert metrics["false_accept_clips"] == 1
    assert metrics["false_accept_crops"] == 1
    assert metrics["false_accept_rate"] == 0.25


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


def test_manifest_normalization_accepts_nemo_audio_filepath(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.touch()
    manifest = tmp_path / "input.jsonl"
    write_jsonl(manifest, [{"audio_filepath": str(audio), "text": "not a wake word"}])
    output = tmp_path / "normalized.jsonl"

    normalise_manifest_inputs([ManifestInput(manifest, None)], output)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["path"] == str(audio.resolve())
    assert "audio_filepath" not in record


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


def test_score_details_do_not_contain_threshold_or_abnormal_fields(monkeypatch: Any, tmp_path: Path) -> None:
    legacy = get_legacy_module()
    monkeypatch.setattr(legacy, "load_audio_float", lambda path, sr: torch.zeros(16000))
    monkeypatch.setattr(legacy, "prediction_scores", lambda *args, **kwargs: [0.1, 0.8])
    detail = legacy.evaluate_one_record_scores(
        object(),
        "model",
        {"path": str(tmp_path / "clip.wav"), "text": "Hey Siri"},
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
    assert detail["text"] == "Hey Siri"
    keys = list(detail)
    assert keys.index("text") + 1 == keys.index("best_window")
    assert detail["sliding_windows"]

    background_detail = legacy.evaluate_one_record_scores(
        object(),
        "model",
        {"path": str(tmp_path / "background.wav"), "text": "must not be included"},
        "testing.background",
        0,
        0,
        {
            "sample_rate": 16000,
            "chunk_size": 1280,
            "positive_padding": 1,
            "negative_padding": 0,
            "model_window_seconds": 2.0,
            "record_window_scores": True,
        },
    )
    assert background_detail["text"] == ""


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
dev = feature.positive_dev, feature.negative_dev, feature.background_dev
batch.feature.positive_train = 2
batch.feature.negative_train = 2
steps = 2
phase_step_ratios = [1.0]
phase_learning_rates = [0.0001]
validation_points = 1
model_type = dnn
layer_size = 8
max_negative_weight = 2
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
    assert any(event["event"] == "validation" and "val_loss" in event for event in events)
    per_set = [event for event in events if event["event"] == "validation_set"]
    assert {event["validation_set"] for event in per_set} == {
        "feature.positive_dev",
        "feature.negative_dev",
        "feature.background_dev",
    }
    assert all("loss" in event and "label" in event and "threshold" not in event for event in per_set)
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
