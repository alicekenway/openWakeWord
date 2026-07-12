#!/usr/bin/env python3
"""Resample a directory of WAV files to mono PCM WAV."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio


def collect_wavs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.wav") if path.is_file())


def resolve_workers(value: str) -> int:
    if value.lower() == "auto":
        cpu_count = os.cpu_count() or 2
        return max(1, min(cpu_count, 16))
    workers = int(value)
    if workers <= 0:
        raise ValueError("--workers must be 'auto' or a positive integer")
    return workers


def to_mono_float(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(torch.float32).clamp(-1.0, 1.0)


def iter_tasks(paths: list[Path], input_dir: Path, output_dir: Path, sample_rate: int, overwrite: bool) -> Iterable[dict[str, Any]]:
    for index, input_path in enumerate(paths):
        relative_path = input_path.relative_to(input_dir)
        yield {
            "index": index,
            "input_path": str(input_path),
            "output_path": str(output_dir / relative_path),
            "sample_rate": sample_rate,
            "overwrite": overwrite,
        }


def resample_one(task: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    sample_rate = int(task["sample_rate"])
    overwrite = bool(task["overwrite"])
    try:
        if overwrite or not output_path.exists():
            wav, source_sample_rate = torchaudio.load(str(input_path))
            wav = to_mono_float(wav)
            if source_sample_rate != sample_rate:
                wav = torchaudio.functional.resample(wav, source_sample_rate, sample_rate)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(output_path),
                wav,
                sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        return {
            "ok": True,
            "index": int(task["index"]),
            "input_path": str(input_path),
            "output_path": str(output_path),
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": int(task["index"]),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "error": repr(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample a WAV directory to mono PCM WAV.")
    parser.add_argument("--input-dir", required=True, help="Input directory containing WAV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for resampled WAV files")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate")
    parser.add_argument(
        "--workers",
        "--max-workers",
        default="auto",
        help="Number of worker processes, or 'auto'. Default: auto",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N WAV files, useful for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output WAV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    workers = resolve_workers(str(args.workers))
    paths = collect_wavs(input_dir)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than 0 when provided")
        paths = paths[:args.limit]
    task_iter = iter_tasks(paths, input_dir, output_dir, args.sample_rate, args.overwrite)

    errors: list[dict[str, Any]] = []
    converted = 0
    if workers == 1:
        results = map(resample_one, task_iter)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(resample_one, task_iter, chunksize=32)

    try:
        for result in results:
            if result["ok"]:
                converted += 1
            else:
                errors.append(result)
    finally:
        if pool is not None:
            pool.shutdown()

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sample_rate": args.sample_rate,
        "workers": workers,
        "limit": args.limit,
        "input_wav_files": len(paths),
        "converted_files": converted,
        "error_count": len(errors),
        "errors": errors[:50],
        "overwrite": args.overwrite,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "resample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
