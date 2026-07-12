#!/usr/bin/env python3
"""Convert audio referenced by a metadata JSONL file to WAV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio


AUDIO_PATH_KEYS = ["path", "audiofile_path", "audio_file", "audio_path", "file", "filename"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)


def iter_tasks(
    records: list[dict[str, Any]],
    audio_base_dir: Path,
    wav_dir: Path,
    sample_rate: int,
    overwrite: bool,
) -> Iterable[dict[str, Any]]:
    for index, record in enumerate(records):
        yield {
            "index": index,
            "record": record,
            "audio_base_dir": str(audio_base_dir),
            "wav_dir": str(wav_dir),
            "sample_rate": sample_rate,
            "overwrite": overwrite,
        }


def detect_audio_path(record: dict[str, Any]) -> tuple[str, str]:
    for key in AUDIO_PATH_KEYS:
        value = record.get(key)
        if value:
            return key, str(value)
    raise ValueError(f"Record has no supported audio path key: {record}")


def resolve_audio_path(audio_path: str, audio_base_dir: Path) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return audio_base_dir / path


def safe_output_name(index: int, source_path: Path) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{index:08d}_{source_path.stem}_{digest}.wav"


def waveform_to_mono_float(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(torch.float32).clamp(-1.0, 1.0)


def convert_one(task: dict[str, Any]) -> dict[str, Any]:
    index = int(task["index"])
    record = dict(task["record"])
    audio_base_dir = Path(task["audio_base_dir"])
    wav_dir = Path(task["wav_dir"])
    sample_rate = int(task["sample_rate"])
    overwrite = bool(task["overwrite"])

    try:
        _, raw_audio_path = detect_audio_path(record)
        source_path = resolve_audio_path(raw_audio_path, audio_base_dir)
        output_name = safe_output_name(index, source_path)
        output_path = wav_dir / output_name

        if overwrite or not output_path.exists():
            wav, source_sample_rate = torchaudio.load(str(source_path))
            wav = waveform_to_mono_float(wav)
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

        for key in AUDIO_PATH_KEYS:
            record.pop(key, None)
        record.pop("source_path", None)
        vad_trim = record.get("vad_trim")
        if isinstance(vad_trim, dict) and "source_path" in vad_trim:
            record["vad_trim"] = {key: value for key, value in vad_trim.items() if key != "source_path"}
        record["path"] = f"wav/{output_name}"
        return {
            "ok": True,
            "index": index,
            "record": record,
            "source_path": str(source_path),
            "output_path": str(output_path),
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": index,
            "record": record,
            "error": repr(exc),
        }


def resolve_workers(value: str) -> int:
    if value.lower() == "auto":
        cpu_count = os.cpu_count() or 2
        return max(1, min(cpu_count, 16))
    workers = int(value)
    if workers <= 0:
        raise ValueError("--workers must be 'auto' or a positive integer")
    return workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert audio referenced by JSONL metadata to WAV.")
    parser.add_argument("--input-jsonl", required=True, help="Input metadata JSONL")
    parser.add_argument("--audio-base-dir", required=True, help="Base directory for relative audio paths in JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory. WAV files are written to output-dir/wav")
    parser.add_argument(
        "--output-jsonl",
        help="Output JSONL path. Default: output-dir/metadata.jsonl",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output WAV sample rate")
    parser.add_argument(
        "--workers",
        "--max-workers",
        default="auto",
        help="Number of worker processes, or 'auto'. Default: auto",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV files")
    return parser.parse_args()


def run_conversion(args: argparse.Namespace) -> dict[str, Any]:
    input_jsonl = Path(args.input_jsonl).resolve()
    audio_base_dir = Path(args.audio_base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / "wav"
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else output_dir / "metadata.jsonl"
    workers = resolve_workers(str(args.workers))

    records = read_jsonl(input_jsonl)
    errors: list[dict[str, Any]] = []
    written_rows = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    task_iter = iter_tasks(records, audio_base_dir, wav_dir, args.sample_rate, args.overwrite)
    if workers == 1:
        results = map(convert_one, task_iter)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(convert_one, task_iter, chunksize=32)

    try:
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for result in results:
                if result["ok"]:
                    handle.write(json.dumps(result["record"], ensure_ascii=False, sort_keys=True) + "\n")
                    written_rows += 1
                else:
                    errors.append(
                        {
                            "index": result["index"],
                            "record": result["record"],
                            "error": result["error"],
                        }
                    )
    finally:
        if workers != 1:
            pool.shutdown()

    summary = {
        "input_jsonl": str(input_jsonl),
        "audio_base_dir": str(audio_base_dir),
        "output_dir": str(output_dir),
        "wav_dir": str(wav_dir),
        "output_jsonl": str(output_jsonl),
        "sample_rate": args.sample_rate,
        "workers": workers,
        "input_rows": len(records),
        "written_rows": written_rows,
        "error_count": len(errors),
        "errors": errors[:50],
        "overwrite": args.overwrite,
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run_conversion(parse_args())


if __name__ == "__main__":
    main()
