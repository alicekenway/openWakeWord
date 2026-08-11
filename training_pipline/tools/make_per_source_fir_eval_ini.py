#!/usr/bin/env python3
"""Generate the shared clean/noise/FIR per-source evaluation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path


SPEECH_SOURCES = (
    ("positive_wuw_audio", 1),
    ("negative_non_wuw_audio", 0),
    ("negative_car_control", 0),
    ("negative_cv_en_train_phoneme", 0),
)

BACKGROUND_SOURCES = (
    (
        "background_vehicle",
        "${main:input_data_dir}/background/all_metadata.jsonl",
    ),
    (
        "background_wac",
        "/mnt/users/jinyang_wang/open_source_data_collection/ENX/"
        "background_merged/wac_train_noise/metadata.jsonl",
    ),
)


def _testing_section(
    *,
    name: str,
    input_jsonl: str,
    expected_label: int,
    condition: str,
    vad_enabled: bool,
) -> str:
    vad_suffix = "on" if vad_enabled else "off"
    lines = [
        f"[testing.{name}_{condition}_vad_{vad_suffix}]",
        "structure = ${testing.common:structure}",
        f"input_jsonl = {input_jsonl}",
        f"expected_label = {expected_label}",
        "model = ${testing.common:model}",
        "stage1_model = ${testing.common:stage1_model}",
        "stage1_contract = ${testing.common:stage1_contract}",
        "keywords = ${testing.common:keywords}",
        "stage1_device = ${testing.common:stage1_device}",
        "stage1_gate_score = ${testing.common:stage1_gate_score}",
        "ctc_proposal_score_floor = ${testing.common:ctc_proposal_score_floor}",
        "competitor_beam_size = ${testing.common:competitor_beam_size}",
        "competitor_token_prune = ${testing.common:competitor_token_prune}",
        "candidate_pre_margin_frames = ${testing.common:candidate_pre_margin_frames}",
        "candidate_post_margin_frames = ${testing.common:candidate_post_margin_frames}",
        "window_seconds = ${testing.common:window_seconds}",
        "window_count = ${testing.common:window_count}",
        "audio_window_seconds = ${testing.common:audio_window_seconds}",
        "audio_window_stride_seconds = ${testing.common:audio_window_stride_seconds}",
        f"vad_enabled = {'yes' if vad_enabled else 'no'}",
    ]
    if vad_enabled:
        lines.extend(
            [
                "vad_model = ${testing.common:vad_model}",
                "vad_threshold = ${testing.common:vad_threshold}",
                "vad_padding_ms = ${testing.common:vad_padding_ms}",
                "vad_threads = ${testing.common:vad_threads}",
            ]
        )
    lines.append(
        "output_dir = ${main:experiment_dir}/evaluation/per_source/"
        f"{name}/{condition}_vad_{vad_suffix}"
    )
    return "\n".join(lines)


def build_config(*, experiment: str, model_name: str) -> str:
    experiment_dir = f"/mnt/users/jinyang_wang/WUW/train/{experiment}"
    batch9 = "/mnt/users/jinyang_wang/WUW/train/features/batch_9"
    input_dir = "/mnt/users/jinyang_wang/WUW/train/features/batch_1/input_data"
    tests: list[tuple[str, str, int, str, bool]] = []

    source_dirs = {
        "negative_cv_en_train_phoneme": "negative_cv-en-train_phoneme",
    }
    for source, expected_label in SPEECH_SOURCES:
        source_dir = source_dirs.get(source, source)
        manifests = {
            "clean": f"{batch9}/augmented/{source_dir}/clean_test.jsonl",
            "noise": f"{batch9}/augmented/{source_dir}/noise_test.jsonl",
            "fir": f"{batch9}/augmented/{source_dir}/fir_test.jsonl",
        }
        for condition, manifest in manifests.items():
            for vad_enabled in (False, True):
                tests.append((source, manifest, expected_label, condition, vad_enabled))

    for source, manifest in BACKGROUND_SOURCES:
        for vad_enabled in (False, True):
            tests.append((source, manifest, 0, "clean", vad_enabled))

    test_steps = [
        f"testing.{name}_{condition}_vad_{'on' if vad else 'off'}"
        for name, _, _, condition, vad in tests
    ]
    blocks = [
        "[main]",
        f"experiment_dir = {experiment_dir}",
        f"input_data_dir = {input_dir}",
        "sample_rate = 16000",
        "seed = 1337",
        "pipeline_checkpoint_dir = ${main:experiment_dir}/per_source_test_pipeline_state",
        "execution_mode = slurm",
        "slurm_array_tasks = 50",
        "slurm_cpu_sbatch_args = --partition=cpu --mem=8G --cpus-per-task=1",
        "",
        "[steps]",
        f"steps = [{', '.join(test_steps)}], summary",
        "",
        "[testing.common]",
        "structure = ctc_wac",
        f"model = ${{main:experiment_dir}}/trained_model/{model_name}.onnx",
        "stage1_model = /mnt/users/jinyang_wang/WUW/model/base_expts4/output/stage1-wuw.int8.onnx",
        "stage1_contract = /mnt/users/jinyang_wang/WUW/model/base_expts4/output/stage1-wuw.contract.json",
        "keywords = /mnt/users/jinyang_wang/WUW/train/expts3/thresholds.json",
        "stage1_device = cpu",
        "stage1_gate_score = normalized_confidence",
        "ctc_proposal_score_floor = -3",
        "competitor_beam_size = 16",
        "competitor_token_prune = 8",
        "candidate_pre_margin_frames = 3",
        "candidate_post_margin_frames = 0",
        "window_seconds = 2.56",
        "window_count = 2",
        "audio_window_seconds = 5.12",
        "audio_window_stride_seconds = 2.56",
        "vad_model = /mnt/users/jinyang_wang/WUW/model/silero/silero_vad.onnx",
        "vad_threshold = 0.5",
        "vad_padding_ms = 100",
        "vad_threads = 1",
    ]

    for name, manifest, label, condition, vad_enabled in tests:
        blocks.extend(
            [
                "",
                _testing_section(
                    name=name,
                    input_jsonl=manifest,
                    expected_label=label,
                    condition=condition,
                    vad_enabled=vad_enabled,
                ),
            ]
        )

    blocks.extend(
        [
            "",
            "[summary]",
            f"tests = {', '.join(test_steps)}",
            "threshold_start = 0.1",
            "threshold_stop = 0.9",
            "threshold_step = 0.1",
            "debounce_seconds = 1.0",
            "output_json = ${main:experiment_dir}/evaluation/per_source_comparison.json",
            "output_report = ${main:experiment_dir}/evaluation/PER_SOURCE_COMPARISON.md",
            "",
            "[slurm]",
            "merge_max_parallel = 2",
            "sbatch_command = sbatch",
            "squeue_command = squeue",
            "python_executable = /mnt/users/jinyang_wang/WUW/.wuw/bin/python",
            "",
            "[slurm.summary]",
            "tasks = 1",
            "sbatch_args = ${main:slurm_cpu_sbatch_args}",
        ]
    )
    for step in test_steps:
        blocks.extend(
            [
                "",
                f"[slurm.{step}]",
                "tasks = ${main:slurm_array_tasks}",
                "sbatch_args = ${main:slurm_cpu_sbatch_args}",
            ]
        )
    return "\n".join(blocks) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.write_text(
        build_config(experiment=args.experiment, model_name=args.model_name),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
