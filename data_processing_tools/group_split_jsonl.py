#!/usr/bin/env python3
"""Split a metadata JSONL file by shuffled consecutive groups."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


REST_TOKENS = {"*", "all", "rest", "remaining"}
AUDIO_PATH_KEYS = ["path", "audiofile_path", "audio_file", "audio_path", "file", "filename", "audio_filepath"]


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


def audio_path_key(record: dict[str, Any]) -> str:
    for key in AUDIO_PATH_KEYS:
        if record.get(key):
            return key
    raise ValueError(f"JSONL record has no audio path field: {record}")


def absolutize_audio_paths(
    records: list[dict[str, Any]],
    audio_base_dir: str | None,
    add_path_field: bool,
) -> tuple[list[dict[str, Any]], int]:
    updated: list[dict[str, Any]] = []
    for record in records:
        output_record = dict(record)
        key = audio_path_key(output_record)
        audio_path = os.path.expanduser(str(output_record[key]))
        if audio_base_dir and os.path.isabs(audio_path):
            abs_path = os.path.abspath(audio_path)
        elif audio_base_dir:
            abs_path = os.path.abspath(os.path.join(audio_base_dir, audio_path))
        else:
            abs_path = audio_path
        for path_key in AUDIO_PATH_KEYS:
            output_record.pop(path_key, None)
        output_record.pop("source_path", None)
        vad_trim = output_record.get("vad_trim")
        if isinstance(vad_trim, dict) and "source_path" in vad_trim:
            output_record["vad_trim"] = {item_key: value for item_key, value in vad_trim.items() if item_key != "source_path"}
        output_record["path"] = abs_path
        updated.append(output_record)
    return updated, len(updated)


def chunk_groups(records: list[dict[str, Any]], group_size: int, drop_partial_group: bool) -> list[list[dict[str, Any]]]:
    groups = [records[start:start + group_size] for start in range(0, len(records), group_size)]
    if drop_partial_group and groups and len(groups[-1]) < group_size:
        groups.pop()
    return groups


def parse_colon_list(value: str, arg_name: str) -> list[str]:
    parts = [part.strip() for part in value.split(":")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{arg_name} must be a colon-separated list with no empty parts")
    return parts


def parse_group_counts(value: str, n_sets: int, total_groups: int) -> list[int]:
    raw_counts = parse_colon_list(value, "--group-counts")
    if len(raw_counts) != n_sets:
        raise ValueError("--set-names and --group-counts must have the same number of parts")

    rest_index: int | None = None
    counts: list[int | None] = []
    fixed_total = 0
    for ndx, raw in enumerate(raw_counts):
        lowered = raw.lower()
        if lowered in REST_TOKENS:
            if rest_index is not None:
                raise ValueError("--group-counts can only contain one rest/all/* token")
            rest_index = ndx
            counts.append(None)
            continue
        count = int(raw)
        if count < 0:
            raise ValueError("--group-counts values must be non-negative")
        counts.append(count)
        fixed_total += count

    if fixed_total > total_groups:
        raise ValueError(f"Requested {fixed_total} fixed groups, but input only has {total_groups} groups")

    if rest_index is not None:
        counts[rest_index] = total_groups - fixed_total

    parsed_counts = [int(count) for count in counts]
    requested_total = sum(parsed_counts)
    if requested_total > total_groups:
        raise ValueError(f"Requested {requested_total} groups, but input only has {total_groups} groups")
    return parsed_counts


def flatten(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [record for group in groups for record in group]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split JSONL metadata without breaking consecutive row groups.")
    parser.add_argument("--input-jsonl", required=True, help="Input metadata JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory for split JSONL files")
    parser.add_argument(
        "--set-names",
        default="train:val:test",
        help="Colon-separated output set names, for example train:val:test",
    )
    parser.add_argument(
        "--group-counts",
        required=True,
        help="Colon-separated group counts matching --set-names, for example 100:200:1000 or 100:200:rest",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Number of consecutive rows per group. Default 1 is normal row-level shuffling.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Group shuffle seed")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep groups in original order instead of shuffling groups first",
    )
    parser.add_argument(
        "--drop-partial-group",
        action="store_true",
        help="Drop the final group if it has fewer than --group-size rows",
    )
    parser.add_argument(
        "--unused-name",
        default="unused",
        help="Filename stem for leftover groups when group counts do not consume all groups. Use '' to skip writing leftovers.",
    )
    parser.add_argument(
        "--audio-base-dir",
        "--audio-base-path",
        dest="audio_base_dir",
        help="Resolve relative audio paths against this directory and write absolute paths to the split JSONL files.",
    )
    parser.add_argument(
        "--add-path-field",
        action="store_true",
        help="Deprecated compatibility flag; output always uses one canonical 'path' field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be greater than 0")

    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    set_names = parse_colon_list(args.set_names, "--set-names")

    records = read_jsonl(input_jsonl)
    audio_base_dir = os.path.abspath(os.path.expanduser(args.audio_base_dir)) if args.audio_base_dir else None
    records, paths_rewritten = absolutize_audio_paths(records, audio_base_dir, args.add_path_field)

    groups = chunk_groups(records, args.group_size, args.drop_partial_group)
    if not groups:
        raise ValueError(f"No groups available from {input_jsonl}")
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(groups)

    group_counts = parse_group_counts(args.group_counts, len(set_names), len(groups))
    cursor = 0
    output_counts: dict[str, dict[str, int]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for set_name, group_count in zip(set_names, group_counts):
        selected_groups = groups[cursor:cursor + group_count]
        cursor += group_count
        selected_records = flatten(selected_groups)
        output_path = output_dir / f"{set_name}.jsonl"
        written = write_jsonl(output_path, selected_records)
        output_counts[set_name] = {
            "groups": len(selected_groups),
            "records": written,
        }

    unused_groups = groups[cursor:]
    unused_records = flatten(unused_groups)
    if unused_groups and args.unused_name:
        unused_path = output_dir / f"{args.unused_name}.jsonl"
        write_jsonl(unused_path, unused_records)

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "input_records": len(records),
        "audio_base_dir": str(audio_base_dir) if audio_base_dir else None,
        "absolute_audio_paths": bool(audio_base_dir),
        "paths_rewritten": paths_rewritten,
        "add_path_field": args.add_path_field,
        "group_size": args.group_size,
        "total_groups": len(groups),
        "dropped_partial_group": args.drop_partial_group,
        "shuffle": not args.no_shuffle,
        "seed": args.seed,
        "sets": output_counts,
        "unused_groups": len(unused_groups),
        "unused_records": len(unused_records),
        "unused_name": args.unused_name if unused_groups and args.unused_name else None,
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
