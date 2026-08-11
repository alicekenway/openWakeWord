from __future__ import annotations

import importlib.util
from pathlib import Path

from wuw_training.config import load_ini_config, parse_step_groups


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "make_full_5condition_eval_ini.py"
DATASETS = (
    "positive_wuw_audio",
    "positive_hei_siri_sampled",
    "negative_cosyvoice",
    "negative_gigaspeech_m",
)
CONDITIONS = ("car_related", "fir", "car_related_fir", "ordinary", "clean")


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_full_5condition_eval_ini", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_matrix_contains_every_dataset_and_condition(tmp_path: Path) -> None:
    module = _load_generator()
    path = tmp_path / "generate.ini"
    path.write_text(module.build_generation_config(), encoding="utf-8")
    config = load_ini_config(path)
    groups = parse_step_groups(config.get("steps", "steps"))

    assert len(groups) == 1
    assert len(groups[0]) == 20
    for dataset in DATASETS:
        manifests = set()
        for condition in CONDITIONS:
            step = f"augment.{dataset}_{condition}"
            assert step in groups[0]
            manifests.add(config.get(step, "output_manifest"))
            expected_background = condition in {"car_related", "car_related_fir", "ordinary"}
            expected_fir = condition in {"fir", "car_related_fir"}
            assert config.getfloat(step, "background_probability") == float(expected_background)
            assert config.getfloat(step, "fir_probability") == float(expected_fir)
            if condition == "ordinary":
                assert config.get(step, "noise_jsonl").endswith(
                    "/ordinary_noise_metadata.jsonl"
                )
            elif expected_background:
                assert config.get(step, "noise_jsonl").endswith(
                    "/car_related_metadata.jsonl"
                )
        assert len(manifests) == 5


def test_evaluation_matrix_has_independent_vad_on_blocks(tmp_path: Path) -> None:
    module = _load_generator()
    path = tmp_path / "eval.ini"
    path.write_text(
        module.build_evaluation_config(
            experiment_name="expts7_full",
            model_path="/models/expts7.onnx",
        ),
        encoding="utf-8",
    )
    config = load_ini_config(path)
    groups = parse_step_groups(config.get("steps", "steps"))

    assert len(groups) == 2
    assert len(groups[0]) == 20
    assert groups[1] == ["summary"]
    output_dirs = set()
    for dataset in DATASETS:
        for condition in CONDITIONS:
            step = f"testing.{dataset}_{condition}"
            assert step in groups[0]
            assert config.getboolean(step, "vad_enabled") is True
            assert config.get(step, "input_jsonl").endswith(
                f"/manifests/{dataset}/{condition}.jsonl"
            )
            output_dirs.add(config.get(step, "output_dir"))
    assert len(output_dirs) == 20


def test_model_configs_are_isolated() -> None:
    module = _load_generator()
    expts4 = module.build_evaluation_config(
        experiment_name="expts4_full", model_path="/models/expts4.onnx"
    )
    expts7 = module.build_evaluation_config(
        experiment_name="expts7_full", model_path="/models/expts7.onnx"
    )
    assert "experiment_dir = /mnt/users/jinyang_wang/WUW/test/expts4_full" in expts4
    assert "model = /models/expts4.onnx" in expts4
    assert "experiment_dir = /mnt/users/jinyang_wang/WUW/test/expts7_full" in expts7
    assert "model = /models/expts7.onnx" in expts7
