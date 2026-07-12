#!/usr/bin/env python3
"""Create WUW metadata JSONL from a directory of WAV files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def collect_wavs(wav_dir: Path) -> list[Path]:
    return sorted(path for path in wav_dir.rglob("*.wav") if path.is_file())


def audiofile_value(path: Path, wav_dir: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())
    return str(path.relative_to(wav_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create WUW metadata JSONL from a WAV directory.")
    parser.add_argument("--wav-dir", required=True, help="Input directory containing WAV files")
    parser.add_argument("--output-jsonl", required=True, help="Output metadata JSONL path")
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        help="Randomly sample this many WAV files. If omitted, write all WAV files.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of paths relative to --wav-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_dir = Path(args.wav_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0 when provided")

    wavs = collect_wavs(wav_dir)
    total_wavs = len(wavs)
    if args.sample_size is not None and args.sample_size < total_wavs:
        rng = random.Random(args.seed)
        wavs = rng.sample(wavs, args.sample_size)
        wavs.sort()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for path in wavs:
            record = {
                "path": audiofile_value(path, wav_dir, args.absolute_paths),
                "text": "",
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "input_wav_dir": str(wav_dir),
        "output_jsonl": str(output_jsonl),
        "total_wav_files": total_wavs,
        "written_rows": len(wavs),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "absolute_paths": args.absolute_paths,
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
