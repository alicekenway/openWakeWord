#!/usr/bin/env python3
"""Convert a Common Voice TSV split into the simple WUW JSONL metadata format."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable


def make_audio_path(path_value: str, audio_prefix: str) -> str:
    path_value = path_value.strip()
    audio_prefix = audio_prefix.strip().strip("/")
    if not audio_prefix:
        return path_value
    return f"{audio_prefix}/{path_value.lstrip('/')}"


def convert_row(row: dict[str, str], path_column: str, text_column: str, audio_prefix: str) -> dict[str, str]:
    if path_column not in row:
        raise KeyError(f"TSV is missing path column: {path_column}")
    if text_column not in row:
        raise KeyError(f"TSV is missing text column: {text_column}")
    return {
        "path": make_audio_path(row[path_column], audio_prefix),
        "text": row[text_column],
    }


def iter_converted_rows(
    tsv_path: Path,
    path_column: str,
    text_column: str,
    audio_prefix: str,
) -> Iterable[dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            yield convert_row(row, path_column, text_column, audio_prefix)


def reservoir_sample(rows: Iterable[dict[str, str]], sample_size: int, seed: int) -> tuple[list[dict[str, str]], int]:
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    total = 0
    for total, row in enumerate(rows, start=1):
        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue
        replace_ndx = rng.randint(1, total)
        if replace_ndx <= sample_size:
            reservoir[replace_ndx - 1] = row
    rng.shuffle(reservoir)
    return reservoir, total


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Common Voice TSV rows to WUW metadata JSONL.")
    parser.add_argument("--tsv", required=True, help="Input Common Voice TSV path, for example en/train.tsv")
    parser.add_argument("--output-jsonl", required=True, help="Output metadata JSONL path")
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        help="Randomly sample this many rows. If omitted, write all TSV rows.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    parser.add_argument("--path-column", default="path", help="TSV column containing the audio filename")
    parser.add_argument("--text-column", default="sentence", help="TSV column containing transcript text")
    parser.add_argument(
        "--audio-prefix",
        default="clips",
        help="Prefix added before the TSV path value. Use '' when paths should stay unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tsv_path = Path(args.tsv).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0 when provided")

    converted_rows = iter_converted_rows(
        tsv_path,
        path_column=args.path_column,
        text_column=args.text_column,
        audio_prefix=args.audio_prefix,
    )

    if args.sample_size is None:
        total_rows = write_jsonl(output_jsonl, converted_rows)
        written_rows = total_rows
    else:
        sampled_rows, total_rows = reservoir_sample(converted_rows, args.sample_size, args.seed)
        written_rows = write_jsonl(output_jsonl, sampled_rows)

    summary = {
        "input_tsv": str(tsv_path),
        "output_jsonl": str(output_jsonl),
        "total_input_rows": total_rows,
        "written_rows": written_rows,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "path_column": args.path_column,
        "text_column": args.text_column,
        "audio_prefix": args.audio_prefix,
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
