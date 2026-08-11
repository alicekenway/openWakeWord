#!/usr/bin/env python3
"""Generate matched five-condition full-test configurations.

The experiment follows the older ``expts3_full`` source layout while keeping
augmentation generation separate from model evaluation.  All model evaluation
blocks enable VAD and each dataset/condition pair has its own block.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple


TEST_ROOT = "/mnt/users/jinyang_wang/WUW/test"
DATA_ROOT = f"{TEST_ROOT}/data"
CAR_NOISE = (
    "/mnt/users/jinyang_wang/open_source_data_collection/ENX/"
    "background_merged/car_related_metadata.jsonl"
)
ORDINARY_NOISE = (
    "/mnt/users/jinyang_wang/open_source_data_collection/ENX/"
    "background_merged/ordinary_noise_metadata.jsonl"
)
FIR_LIST = (
    "/mnt/users/jinyang_wang/open_source_data_collection/ENX/"
    "v_class_impulse_responses_16k_mouth_to_standard_mic/fir.list"
)


class Dataset(NamedTuple):
    name: str
    expected_label: int
    input_jsonl: str
    audio_base_dir: str | None
    leading_context: str


DATASETS = (
    Dataset(
        "positive_wuw_audio",
        1,
        "/mnt/users/jinyang_wang/TTS_cosyvoice/generation/ENX/batch_6/"
        "wuw_audio/metadata.jsonl",
        "/mnt/users/jinyang_wang/TTS_cosyvoice/generation/ENX/batch_6/wuw_audio",
        "[0, 2.0]",
    ),
    Dataset(
        "positive_hei_siri_sampled",
        1,
        "/mnt/users/jinyang_wang/open_source_data_collection/ENX/"
        "heiSiriyinpincaiyang_nemo_dataset/meta.jsonl",
        "/mnt/users/jinyang_wang/open_source_data_collection/ENX/"
        "heiSiriyinpincaiyang_nemo_dataset",
        "[0, 2.0]",
    ),
    Dataset(
        "negative_cosyvoice",
        0,
        "/mnt/users/jinyang_wang/TTS_cosyvoice/generation/ENX/batch_6/"
        "non_wuw_audio/wav_merged/metadata.jsonl",
        None,
        "[0, 0]",
    ),
    Dataset(
        "negative_gigaspeech_m",
        0,
        "/mnt/users/jinyang_wang/open_source_data_collection/ENX/"
        "gigaspeech_m_wav/wav_merged/metadata.jsonl",
        None,
        "[0, 0]",
    ),
)

# Keep the order requested by the experiment specification.
CONDITIONS = (
    ("car_related", CAR_NOISE, 1.0, 0.0),
    ("fir", CAR_NOISE, 0.0, 1.0),
    ("car_related_fir", CAR_NOISE, 1.0, 1.0),
    ("ordinary", ORDINARY_NOISE, 1.0, 0.0),
    ("clean", CAR_NOISE, 0.0, 0.0),
)


def augmentation_step(dataset: Dataset, condition: str) -> str:
    return f"augment.{dataset.name}_{condition}"


def testing_step(dataset: Dataset, condition: str) -> str:
    return f"testing.{dataset.name}_{condition}"


def manifest_path(dataset: Dataset, condition: str) -> str:
    return f"{DATA_ROOT}/manifests/{dataset.name}/{condition}.jsonl"


def _augmentation_section(
    dataset: Dataset,
    condition: str,
    noise_jsonl: str,
    background_probability: float,
    fir_probability: float,
) -> str:
    lines = [
        f"[{augmentation_step(dataset, condition)}]",
        f"input_jsonl = {dataset.input_jsonl}",
    ]
    if dataset.audio_base_dir:
        lines.append(f"audio_base_dir = {dataset.audio_base_dir}")
    lines.extend(
        [
            f"noise_jsonl = {noise_jsonl}",
            f"background_probability = {background_probability:.1f}",
            "fir_list = ${augmentation:fir_list}",
            f"fir_probability = {fir_probability:.1f}",
            f"output_dir = {DATA_ROOT}/augmented/{dataset.name}/{condition}",
            f"output_manifest = {manifest_path(dataset, condition)}",
            "rounds = ${augmentation:rounds}",
            "snr_low = ${augmentation:snr_low}",
            "snr_high = ${augmentation:snr_high}",
            "artificial_probability = ${augmentation:artificial_probability}",
            "random_gain_db = ${augmentation:random_gain_db}",
            "placement = ${augmentation:placement}",
            "ctc_context = yes",
            f"leading_context_seconds_range = {dataset.leading_context}",
            "long_audio_mode = full",
            "full_mode_window_seconds = ${augmentation:full_mode_window_seconds}",
            "workers = ${augmentation:workers}",
        ]
    )
    return "\n".join(lines)


def build_generation_config() -> str:
    steps = [
        augmentation_step(dataset, condition)
        for dataset in DATASETS
        for condition, _, _, _ in CONDITIONS
    ]
    blocks = [
        "[main]",
        f"experiment_dir = {TEST_ROOT}/expts7_full",
        f"data_dir = {DATA_ROOT}",
        "sample_rate = 16000",
        "seed = 1337",
        "pipeline_checkpoint_dir = ${main:experiment_dir}/generation_pipeline_state",
        "execution_mode = slurm",
        "slurm_array_tasks = 200",
        "slurm_cpu_sbatch_args = --partition=cpu --mem=30G --cpus-per-task=1",
        "",
        "[steps]",
        f"steps = [{', '.join(steps)}]",
        "",
        "[augmentation]",
        f"car_noise_jsonl = {CAR_NOISE}",
        f"ordinary_noise_jsonl = {ORDINARY_NOISE}",
        f"fir_list = {FIR_LIST}",
        "rounds = 1",
        "snr_low = 10",
        "snr_high = 30",
        "artificial_probability = 0.0",
        "random_gain_db = 0.0",
        "placement = end",
        "full_mode_window_seconds = 30",
        "workers = 1",
    ]
    for dataset in DATASETS:
        for condition, noise_jsonl, background_probability, fir_probability in CONDITIONS:
            blocks.extend(
                [
                    "",
                    _augmentation_section(
                        dataset,
                        condition,
                        noise_jsonl,
                        background_probability,
                        fir_probability,
                    ),
                ]
            )
    blocks.extend(
        [
            "",
            "[slurm]",
            "merge_max_parallel = 2",
            "sbatch_command = sbatch",
            "squeue_command = squeue",
            "python_executable = /mnt/users/jinyang_wang/WUW/.wuw/bin/python",
        ]
    )
    for step in steps:
        blocks.extend(
            [
                "",
                f"[slurm.{step}]",
                "tasks = ${main:slurm_array_tasks}",
                "sbatch_args = ${main:slurm_cpu_sbatch_args}",
            ]
        )
    return "\n".join(blocks) + "\n"


def _testing_section(dataset: Dataset, condition: str) -> str:
    return "\n".join(
        [
            f"[{testing_step(dataset, condition)}]",
            "structure = ${testing.common:structure}",
            f"input_jsonl = {manifest_path(dataset, condition)}",
            f"expected_label = {dataset.expected_label}",
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
            "vad_enabled = yes",
            "vad_model = ${testing.common:vad_model}",
            "vad_threshold = ${testing.common:vad_threshold}",
            "vad_padding_ms = ${testing.common:vad_padding_ms}",
            "vad_threads = ${testing.common:vad_threads}",
            "output_dir = ${main:experiment_dir}/evaluation/"
            f"{dataset.name}/{condition}",
        ]
    )


def build_evaluation_config(*, experiment_name: str, model_path: str) -> str:
    steps = [
        testing_step(dataset, condition)
        for dataset in DATASETS
        for condition, _, _, _ in CONDITIONS
    ]
    blocks = [
        "[main]",
        f"experiment_dir = {TEST_ROOT}/{experiment_name}",
        f"data_dir = {DATA_ROOT}",
        "sample_rate = 16000",
        "seed = 1337",
        "pipeline_checkpoint_dir = ${main:experiment_dir}/pipeline_state",
        "execution_mode = slurm",
        "slurm_array_tasks = 50",
        "slurm_cpu_sbatch_args = --partition=cpu --mem=8G --cpus-per-task=1",
        "",
        "[steps]",
        f"steps = [{', '.join(steps)}], summary",
        "",
        "[testing.common]",
        "structure = ctc_wac",
        f"model = {model_path}",
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
    for dataset in DATASETS:
        for condition, _, _, _ in CONDITIONS:
            blocks.extend(["", _testing_section(dataset, condition)])
    blocks.extend(
        [
            "",
            "[summary]",
            f"tests = {', '.join(steps)}",
            "threshold_start = 0.1",
            "threshold_stop = 0.9",
            "threshold_step = 0.1",
            "debounce_seconds = 1.0",
            "output_json = ${main:experiment_dir}/evaluation/full_condition_comparison.json",
            "output_report = ${main:experiment_dir}/evaluation/FULL_CONDITION_COMPARISON.md",
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
    for step in steps:
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
    parser.add_argument("--generation-output", type=Path, required=True)
    parser.add_argument("--expts4-output", type=Path, required=True)
    parser.add_argument("--expts7-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.generation_output, args.expts4_output, args.expts7_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.generation_output.write_text(build_generation_config(), encoding="utf-8")
    args.expts4_output.write_text(
        build_evaluation_config(
            experiment_name="expts4_full",
            model_path=(
                "/mnt/users/jinyang_wang/WUW/train/expts4/trained_model/"
                "batch7_fir_balanced_car_noise.onnx"
            ),
        ),
        encoding="utf-8",
    )
    args.expts7_output.write_text(
        build_evaluation_config(
            experiment_name="expts7_full",
            model_path=(
                "/mnt/users/jinyang_wang/WUW/train/expts7/hardneg/trained_model/"
                "batch8_clean25_fir20_hardneg.onnx"
            ),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
