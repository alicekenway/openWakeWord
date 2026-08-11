#!/usr/bin/env python3
"""Run a tiny end-to-end FIR augmentation smoke test."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchaudio


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import wuw_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--fir-list", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.expanduser().resolve()
    fir_list = args.fir_list.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000

    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    source_one = 0.2 * torch.sin(2 * math.pi * 440.0 * time)
    source_two = 0.2 * torch.sin(2 * math.pi * (220.0 + 440.0 * time) * time)
    generator = torch.Generator().manual_seed(11)
    noise = 0.03 * torch.randn(sample_rate * 4, generator=generator)

    source_paths = [work_dir / "source_one.wav", work_dir / "source_two.wav"]
    for path, waveform in zip(source_paths, (source_one, source_two), strict=True):
        torchaudio.save(str(path), waveform.unsqueeze(0), sample_rate)
    noise_path = work_dir / "noise.wav"
    torchaudio.save(str(noise_path), noise.unsqueeze(0), sample_rate)

    input_manifest = work_dir / "input.jsonl"
    input_manifest.write_text(
        "".join(json.dumps({"path": str(path)}) + "\n" for path in source_paths),
        encoding="utf-8",
    )
    noise_manifest = work_dir / "noise.jsonl"
    noise_manifest.write_text(json.dumps({"path": str(noise_path)}) + "\n", encoding="utf-8")
    output_manifest = work_dir / "augmented.jsonl"
    output_dir = work_dir / "audio"

    wuw_pipeline.command_augment_audio(
        argparse.Namespace(
            input_manifest=str(input_manifest),
            input_dir=None,
            noise_manifest=[str(noise_manifest)],
            noise_dir=[],
            output_dir=str(output_dir),
            output_manifest=str(output_manifest),
            rounds=1,
            snr_low=15.0,
            snr_high=15.0,
            artificial_prob=0.0,
            fir_list=str(fir_list),
            fir_probability=1.0,
            random_gain_db=0.0,
            clip_seconds=2.0,
            sample_rate=sample_rate,
            placement="end",
            seed=1337,
            overwrite=True,
            workers=1,
            ctc_context=False,
            long_audio_mode="random",
            window_count=1,
            leading_context_seconds_range=None,
            full_mode_window_seconds=None,
        )
    )

    records = wuw_pipeline.read_jsonl(output_manifest)
    summary = wuw_pipeline.read_json(output_manifest.with_suffix(".summary.json"))
    if len(records) != 2 or int(summary["output_count"]) != 2:
        raise RuntimeError(f"Unexpected smoke output count: records={len(records)}, summary={summary}")
    if int(summary["error_count"]) != 0:
        raise RuntimeError(f"Smoke augmentation errors: {summary['errors']}")
    if int(summary["fir_count"]) != 114 or int(summary["fir_applied_count"]) != 2:
        raise RuntimeError(f"Unexpected FIR summary: {summary}")

    output_details = []
    for record in records:
        if not record.get("fir_applied"):
            raise RuntimeError(f"FIR was not applied: {record}")
        if int(record["fir_sample_rate"]) != sample_rate or int(record["fir_taps"]) != 1920:
            raise RuntimeError(f"Unexpected FIR metadata: {record}")
        path = Path(record["path"])
        waveform, decoded_rate = torchaudio.load(str(path))
        if decoded_rate != sample_rate or waveform.shape[-1] != sample_rate * 2:
            raise RuntimeError(
                f"Unexpected augmented audio shape/rate: {path}, {tuple(waveform.shape)}, {decoded_rate}"
            )
        output_details.append(
            {
                "path": str(path),
                "samples": int(waveform.shape[-1]),
                "peak": float(waveform.abs().max().item()),
                "fir": record["fir_path"],
            }
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "output_count": len(records),
                "fir_count": int(summary["fir_count"]),
                "fir_applied_count": int(summary["fir_applied_count"]),
                "outputs": output_details,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
