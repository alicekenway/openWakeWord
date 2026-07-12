#!/usr/bin/env python3
"""Resample WAV files referenced by metadata JSONL into a clean WAV dataset."""

from __future__ import annotations

import argparse

from jsonl_audio_to_wav import run_conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample JSONL-referenced audio to WAV, usually 16 kHz mono PCM.")
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


def main() -> None:
    run_conversion(parse_args())


if __name__ == "__main__":
    main()
