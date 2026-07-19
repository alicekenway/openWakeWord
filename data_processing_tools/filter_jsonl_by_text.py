#!/usr/bin/env python3
"""Remove JSONL records whose text field is listed in a plain-text file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_excluded_texts(path: Path) -> set[str]:
    """Read one exact excluded text value per line, ignoring blank lines."""

    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.rstrip("\r\n")
            if value:
                values.add(value)
    if not values:
        raise ValueError(f"Excluded-text file contains no non-empty values: {path}")
    return values


def filter_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    excluded_texts: set[str],
    *,
    text_field: str,
) -> dict[str, Any]:
    if input_jsonl.resolve() == output_jsonl.resolve():
        raise ValueError("--output-jsonl must differ from --input-jsonl")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    removed_rows = 0
    with input_jsonl.open("r", encoding="utf-8") as source, output_jsonl.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {input_jsonl}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL row {line_number} of {input_jsonl} must be an object")
            if text_field not in record:
                raise ValueError(f"JSONL row {line_number} of {input_jsonl} has no {text_field!r} field")
            value = record[text_field]
            if not isinstance(value, str):
                raise ValueError(f"JSONL row {line_number} of {input_jsonl} has non-string {text_field!r} value")
            input_rows += 1
            if value in excluded_texts:
                removed_rows += 1
                continue
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "text_field": text_field,
        "excluded_text_values": len(excluded_texts),
        "input_rows": input_rows,
        "removed_rows": removed_rows,
        "written_rows": input_rows - removed_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove JSONL records whose text matches a listed unwanted value.")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL file")
    parser.add_argument(
        "--exclude-text-file",
        required=True,
        help="UTF-8 text file with one exact unwanted text value per line",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL containing retained records")
    parser.add_argument("--text-field", default="text", help="JSON field compared to unwanted text values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    excluded_path = Path(args.exclude_text_file).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_jsonl}")
    if not excluded_path.is_file():
        raise FileNotFoundError(f"Excluded-text file does not exist: {excluded_path}")
    if not args.text_field:
        raise ValueError("--text-field must not be empty")

    summary = filter_jsonl(
        input_jsonl,
        output_jsonl,
        read_excluded_texts(excluded_path),
        text_field=args.text_field,
    )
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
