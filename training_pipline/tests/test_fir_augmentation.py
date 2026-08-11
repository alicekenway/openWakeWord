"""Regression tests for WAC-compatible FIR augmentation."""

from __future__ import annotations

import math
import random
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wuw_training.config import ConfigurationError  # noqa: E402
from wuw_training.fir import (  # noqa: E402
    FirFilter,
    apply_fir,
    fir_paths_from_list,
    load_fir_bank,
    load_wac_fir,
)
from wuw_training.legacy import get_legacy_module  # noqa: E402


def _write_fir(path: Path, coefficients: list[float], sample_rate: int = 16_000) -> None:
    values = np.asarray(coefficients, dtype="<f4")
    path.write_bytes(
        struct.pack("<i", values.size)
        + values.tobytes()
        + struct.pack("<i", sample_rate)
    )


def test_wac_fir_list_supports_comments_relative_paths_and_validates_rate(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.fir"
    second = tmp_path / "second.fir"
    _write_fir(first, [1.0, 0.5])
    _write_fir(second, [0.25, -0.1, 0.05])
    fir_list = tmp_path / "fir.list"
    fir_list.write_text("# cabin bank\nfirst.fir\n\nsecond.fir # rear microphone\n")

    assert fir_paths_from_list(fir_list) == [first, second]
    bank = load_fir_bank(fir_list, expected_sample_rate=16_000)
    assert [item.taps for item in bank] == [2, 3]

    with pytest.raises(ConfigurationError, match="sample rate is 16000"):
        load_wac_fir(first, expected_sample_rate=8_000)


def test_wac_fir_rejects_truncated_and_non_finite_payloads(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.fir"
    truncated.write_bytes(struct.pack("<i", 3) + np.ones(2, dtype="<f4").tobytes())
    with pytest.raises(ConfigurationError, match="size does not match"):
        load_wac_fir(truncated, expected_sample_rate=16_000)

    non_finite = tmp_path / "non_finite.fir"
    _write_fir(non_finite, [1.0, float("nan")])
    with pytest.raises(ConfigurationError, match="non-finite"):
        load_wac_fir(non_finite, expected_sample_rate=16_000)


def test_apply_fir_preserves_length_and_wac_peak_normalization(tmp_path: Path) -> None:
    fir_filter = FirFilter(
        path=tmp_path / "identity.fir",
        coefficients=np.asarray([1.0], dtype=np.float32),
        sample_rate=16_000,
    )
    source = torch.tensor([0.25, -0.5, 0.1], dtype=torch.float32)

    output = apply_fir(source, fir_filter)

    assert output.numel() == source.numel()
    assert output.abs().max().item() == pytest.approx(10 ** (-10 / 20), rel=1e-6)
    assert (output[0] / output[1]).item() == pytest.approx(-0.5)


def test_fir_selection_is_deterministic_and_does_not_change_main_rng(tmp_path: Path) -> None:
    pytest.importorskip("torchaudio")
    legacy = get_legacy_module()
    filters = [
        FirFilter(tmp_path / f"{index}.fir", np.ones(1, dtype=np.float32), 16_000)
        for index in range(3)
    ]
    config = {"seed": 1337, "fir_probability": 1.0, "fir_filters": filters}
    main_rng = random.Random(99)
    expected_next = random.Random(99).random()

    first = legacy._select_fir_filter(config, round_index=2, record_index=17)
    second = legacy._select_fir_filter(config, round_index=2, record_index=17)

    assert first is not None and second is not None
    assert first.path == second.path
    assert main_rng.random() == expected_next
    assert legacy._select_fir_filter(
        {"seed": 1337, "fir_probability": 0.0, "fir_filters": filters},
        round_index=2,
        record_index=17,
    ) is None


def test_weighted_noise_sampling_balances_manifests_not_files(tmp_path: Path) -> None:
    pytest.importorskip("torchaudio")
    legacy = get_legacy_module()
    groups = [
        {
            "source_manifest": "small.jsonl",
            "weight": 1.0,
            "paths": [str(tmp_path / "small.wav")],
        },
        {
            "source_manifest": "large.jsonl",
            "weight": 1.0,
            "paths": [str(tmp_path / f"large_{index}.wav") for index in range(100)],
        },
    ]

    def selections(seed: int) -> list[str | None]:
        rng = random.Random(seed)
        return [
            legacy._select_noise_source(rng, noise_paths=[], noise_groups=groups)[1]
            for _ in range(10_000)
        ]

    first = selections(17)
    assert first == selections(17)
    small_fraction = first.count("small.jsonl") / len(first)
    assert 0.48 <= small_fraction <= 0.52


def test_background_gate_is_deterministic_and_matches_probability(tmp_path: Path) -> None:
    pytest.importorskip("torchaudio")
    legacy = get_legacy_module()
    config = {"seed": 1337, "background_probability": 0.75}

    first = [
        legacy._apply_background_noise(config, round_index=0, record_index=index)
        for index in range(10_000)
    ]
    second = [
        legacy._apply_background_noise(config, round_index=0, record_index=index)
        for index in range(10_000)
    ]

    assert first == second
    assert 0.73 <= sum(first) / len(first) <= 0.77
    assert legacy._apply_background_noise(
        {"seed": 1, "background_probability": 0.0},
        round_index=0,
        record_index=0,
    ) is False
    assert legacy._apply_background_noise(
        {"seed": 1, "background_probability": 1.0},
        round_index=0,
        record_index=0,
    ) is True


def test_ctc_worker_can_emit_clean_foreground_without_background(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torchaudio")
    legacy = get_legacy_module()
    source = torch.tensor([0.1, 0.2, -0.1], dtype=torch.float32)
    foreground = torch.cat([torch.zeros(2), source])
    captured: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        legacy,
        "_ctc_context_signal",
        lambda *_args, **_kwargs: (foreground.clone(), source.clone(), 2),
    )
    monkeypatch.setattr(
        legacy,
        "mix_with_noise",
        lambda *_args, **_kwargs: pytest.fail("background mixing should be skipped"),
    )
    monkeypatch.setattr(
        legacy,
        "save_wav",
        lambda _path, audio, sr: captured.setdefault("saved", audio.clone()),
    )
    legacy._init_augment_worker(
        {
            "output_dir": str(tmp_path / "output"),
            "noise_paths": [str(tmp_path / "noise.wav")],
            "target_samples": foreground.numel(),
            "snr_low": 10.0,
            "snr_high": 10.0,
            "artificial_prob": 0.0,
            "background_probability": 0.0,
            "random_gain_db": 0.0,
            "sample_rate": 16_000,
            "placement": "end",
            "ctc_context": True,
            "long_audio_mode": "filter",
            "leading_context_range_samples": None,
            "full_mode_window_samples": None,
            "fir_filters": [],
            "fir_probability": 0.0,
            "seed": 7,
            "overwrite": True,
        }
    )

    result = legacy._augment_audio_worker((0, 5, {"path": str(tmp_path / "source.wav")}))

    assert result["error"] is None
    assert result["record"]["background_applied"] is False
    assert result["record"]["noise_path"] is None
    assert torch.equal(captured["saved"], foreground)


def test_ctc_worker_reverbs_only_active_source_before_background(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torchaudio")
    legacy = get_legacy_module()
    source = torch.tensor([0.1, 0.2, -0.1], dtype=torch.float32)
    leading = 4
    foreground = torch.cat([torch.zeros(leading), source])
    fir_filter = FirFilter(
        path=tmp_path / "cabin.fir",
        coefficients=np.asarray([1.0], dtype=np.float32),
        sample_rate=16_000,
    )
    captured: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        legacy,
        "_ctc_context_signal",
        lambda *_args, **_kwargs: (foreground.clone(), source.clone(), leading),
    )
    monkeypatch.setattr(
        legacy,
        "_ctc_background_window",
        lambda _path, *, target_samples, rng, sample_rate: torch.zeros(target_samples),
    )

    def capture_mix(
        signal: torch.Tensor,
        noise: torch.Tensor,
        snr_db: float,
        *,
        signal_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        captured["signal"] = signal.clone()
        assert signal_reference is not None
        captured["reference"] = signal_reference.clone()
        return signal

    monkeypatch.setattr(legacy, "mix_with_noise", capture_mix)
    monkeypatch.setattr(
        legacy,
        "save_wav",
        lambda _path, audio, sr: captured.setdefault("saved", audio.clone()),
    )
    legacy._init_augment_worker(
        {
            "output_dir": str(tmp_path / "output"),
            "noise_paths": [str(tmp_path / "noise.wav")],
            "target_samples": foreground.numel(),
            "snr_low": 10.0,
            "snr_high": 10.0,
            "artificial_prob": 0.0,
            "random_gain_db": 0.0,
            "sample_rate": 16_000,
            "placement": "end",
            "ctc_context": True,
            "long_audio_mode": "filter",
            "leading_context_range_samples": None,
            "full_mode_window_samples": None,
            "fir_filters": [fir_filter],
            "fir_probability": 1.0,
            "seed": 7,
            "overwrite": True,
        }
    )

    result = legacy._augment_audio_worker((0, 5, {"path": str(tmp_path / "source.wav")}))

    assert result["error"] is None
    record = result["record"]
    assert record["fir_applied"] is True
    assert record["fir_path"] == str(fir_filter.path)
    assert record["fir_sample_rate"] == 16_000
    assert record["fir_taps"] == 1
    assert torch.count_nonzero(captured["signal"][:leading]).item() == 0
    assert torch.equal(captured["signal"][leading:], captured["reference"])
    assert captured["reference"].abs().max().item() == pytest.approx(
        math.pow(10.0, -10.0 / 20.0),
        rel=1e-6,
    )
