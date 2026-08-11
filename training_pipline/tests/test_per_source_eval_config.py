from __future__ import annotations

import importlib.util
from pathlib import Path

from wuw_training.config import load_ini_config, parse_step_groups


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "make_per_source_fir_eval_ini.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_per_source_fir_eval_ini", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_contains_every_speech_source_condition_and_vad_mode(tmp_path: Path) -> None:
    module = _load_generator()
    output = tmp_path / "matrix.ini"
    output.write_text(
        module.build_config(experiment="expts6", model_name="model6"),
        encoding="utf-8",
    )
    config = load_ini_config(output)
    groups = parse_step_groups(config.get("steps", "steps"))

    assert len(groups) == 2
    assert len(groups[0]) == 28
    assert groups[1] == ["summary"]
    for source in (
        "positive_wuw_audio",
        "negative_non_wuw_audio",
        "negative_car_control",
        "negative_cv_en_train_phoneme",
    ):
        for condition in ("clean", "noise", "fir"):
            for vad_mode in ("off", "on"):
                assert f"testing.{source}_{condition}_vad_{vad_mode}" in groups[0]
                manifest = config.get(
                    f"testing.{source}_{condition}_vad_{vad_mode}",
                    "input_jsonl",
                )
                assert "/features/batch_9/augmented/" in manifest


def test_background_sources_are_separate_and_never_augmented(tmp_path: Path) -> None:
    module = _load_generator()
    output = tmp_path / "matrix.ini"
    output.write_text(
        module.build_config(experiment="expts4", model_name="model4"),
        encoding="utf-8",
    )
    config = load_ini_config(output)
    steps = parse_step_groups(config.get("steps", "steps"))[0]
    background_steps = [step for step in steps if "background_" in step]

    assert background_steps == [
        "testing.background_vehicle_clean_vad_off",
        "testing.background_vehicle_clean_vad_on",
        "testing.background_wac_clean_vad_off",
        "testing.background_wac_clean_vad_on",
    ]
    for step in background_steps:
        manifest = config.get(step, "input_jsonl")
        assert "/augmented/" not in manifest
