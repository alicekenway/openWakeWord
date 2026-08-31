#!/usr/bin/env python3
"""Build replacement wake-word manifests without copying audio.

The tool starts from an existing ``input_data`` tree, removes one legacy
positive phrase, adds new positive and hard-negative TTS manifests, and makes
a group-safe validation/test split from a generated CosyVoice evaluation set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


AUDIO_PATH_KEYS = (
    "path",
    "audio_filepath",
    "audiofile_path",
    "audio_file",
    "audio_path",
    "file",
    "filename",
)
SPLITS = ("train", "val", "test")
TRIMMED_NAME = re.compile(r"^\d{8}_(\d+)_([0-9a-f]{10})$")


def normalized_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object in {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def audio_path(record: dict[str, Any]) -> str:
    for key in AUDIO_PATH_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError("JSONL record has no audio path field")


def canonicalize(record: dict[str, Any], manifest: Path, *, check_audio: bool) -> dict[str, Any]:
    raw = Path(audio_path(record)).expanduser()
    resolved = raw if raw.is_absolute() else manifest.parent / raw
    resolved = Path(os.path.abspath(os.fspath(resolved)))
    if check_audio and not resolved.is_file():
        raise FileNotFoundError(f"Audio referenced by {manifest} does not exist: {resolved}")
    updated = dict(record)
    for key in AUDIO_PATH_KEYS:
        updated.pop(key, None)
    updated.pop("source_path", None)
    updated["path"] = str(resolved)
    return updated


def no_speech(record: dict[str, Any], manifest: Path) -> bool:
    details = record.get("vad_trim")
    if not isinstance(details, dict) or not isinstance(details.get("no_speech"), bool):
        raise ValueError(f"New TTS row in {manifest} lacks boolean vad_trim.no_speech")
    return bool(details["no_speech"])


def load_variants(path: Path) -> dict[str, tuple[str, str]]:
    raw = read_json(path)
    values = raw.get("variants") if isinstance(raw, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a non-empty variants list")
    variants: dict[str, tuple[str, str]] = {}
    ids: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: variants[{index}] must be an object")
        source_text = " ".join(str(value.get("source_text", "")).split())
        display_text = " ".join(str(value.get("display_text", "")).split())
        keyword_id = str(value.get("id", "")).strip()
        if not source_text or not display_text or not keyword_id:
            raise ValueError(f"{path}: variants[{index}] needs source_text, display_text, and id")
        if source_text in variants or keyword_id in ids:
            raise ValueError(f"{path}: duplicate source_text or id at variants[{index}]")
        variants[source_text] = (keyword_id, display_text)
        ids.add(keyword_id)
    return variants


def failed_candidate_keys(path: Path) -> set[tuple[str, int, int]]:
    values = read_json(path)
    if not isinstance(values, list):
        raise ValueError(f"{path} must contain a JSON list")
    result: set[tuple[str, int, int]] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: failures[{index}] must be an object")
        try:
            result.add(
                (
                    str(value["id"]),
                    int(value["text_index"]),
                    int(value["reference_index"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid failure at index {index}") from exc
    return result


def emitted_group_map(
    generated_path: Path,
    failed_path: Path,
    *,
    output_start: int,
    output_end: int,
) -> tuple[list[str], list[str]]:
    generated = read_json(generated_path)
    if not isinstance(generated, list) or not generated:
        raise ValueError(f"{generated_path} must contain a non-empty group list")
    failures = failed_candidate_keys(failed_path)
    emitted: list[str] = []
    group_ids: list[str] = []
    for group_index, group in enumerate(generated):
        if not isinstance(group, dict) or not isinstance(group.get("output"), list):
            raise ValueError(f"{generated_path}: invalid group at index {group_index}")
        group_id = str(group.get("id", group_index))
        group_ids.append(group_id)
        for output_index, output in enumerate(group["output"]):
            if not output_start <= output_index <= output_end:
                continue
            paths = output.get("candidate_audio_path") if isinstance(output, dict) else None
            if not isinstance(paths, list):
                raise ValueError(
                    f"{generated_path}: group {group_id} output {output_index} lacks candidate_audio_path"
                )
            for candidate_index, _ in enumerate(paths):
                if (group_id, output_index, candidate_index) not in failures:
                    emitted.append(group_id)
    if len(group_ids) != len(set(group_ids)):
        raise ValueError(f"{generated_path} contains duplicate group IDs")
    return emitted, group_ids


def trimmed_source_index(record: dict[str, Any], manifest: Path) -> int:
    stem = Path(audio_path(record)).stem
    match = TRIMMED_NAME.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot recover source index from {manifest} path stem {stem!r}")
    return int(match.group(1))


def prepare_new_positive(
    manifest: Path,
    variants: dict[str, tuple[str, str]],
    *,
    check_audio: bool,
    group_map: list[str] | None = None,
    validation_groups: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    output = {"train": [], "val": [], "test": []}
    dropped = 0
    seen_source_indexes: set[int] = set()
    for row in read_jsonl(manifest):
        source_text = " ".join(str(row.get("text", "")).split())
        if source_text not in variants:
            raise ValueError(f"Unexpected positive text {source_text!r} in {manifest}")
        group_id: str | None = None
        if group_map is not None:
            source_index = trimmed_source_index(row, manifest)
            if source_index in seen_source_indexes:
                raise ValueError(f"Duplicate trimmed source index {source_index} in {manifest}")
            seen_source_indexes.add(source_index)
            if not 0 <= source_index < len(group_map):
                raise ValueError(f"Source index {source_index} is outside provenance map for {manifest}")
            group_id = group_map[source_index]
        if no_speech(row, manifest):
            dropped += 1
            continue
        keyword_id, display_text = variants[source_text]
        updated = canonicalize(row, manifest, check_audio=check_audio)
        updated["text"] = display_text
        updated["expected_keyword_id"] = keyword_id
        split = "train"
        if group_id is not None:
            updated["source_group_id"] = group_id
            split = "val" if group_id in (validation_groups or set()) else "test"
        output[split].append(updated)
    if group_map is not None and len(seen_source_indexes) != len(group_map):
        raise ValueError(
            f"Provenance map has {len(group_map)} rows but {manifest} has "
            f"{len(seen_source_indexes)} unique source indexes"
        )
    return output, dropped


def prepare_new_negative(
    manifest: Path,
    *,
    check_audio: bool,
    group_map: list[str] | None = None,
    validation_groups: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    output = {"train": [], "val": [], "test": []}
    dropped = 0
    seen_source_indexes: set[int] = set()
    for row in read_jsonl(manifest):
        text = " ".join(str(row.get("text", "")).split())
        if not text:
            raise ValueError(f"Negative row in {manifest} has empty text")
        group_id: str | None = None
        if group_map is not None:
            source_index = trimmed_source_index(row, manifest)
            if source_index in seen_source_indexes:
                raise ValueError(f"Duplicate trimmed source index {source_index} in {manifest}")
            seen_source_indexes.add(source_index)
            if not 0 <= source_index < len(group_map):
                raise ValueError(f"Source index {source_index} is outside provenance map for {manifest}")
            group_id = group_map[source_index]
        if no_speech(row, manifest):
            dropped += 1
            continue
        updated = canonicalize(row, manifest, check_audio=check_audio)
        updated["text"] = text
        split = "train"
        if group_id is not None:
            updated["source_group_id"] = group_id
            split = "val" if group_id in (validation_groups or set()) else "test"
        output[split].append(updated)
    if group_map is not None and len(seen_source_indexes) != len(group_map):
        raise ValueError(
            f"Provenance map has {len(group_map)} rows but {manifest} has "
            f"{len(seen_source_indexes)} unique source indexes"
        )
    return output, dropped


def split_paths(rows_by_split: dict[str, list[dict[str, Any]]], label: str) -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        for row in rows_by_split[split]:
            path = str(row["path"])
            previous = seen.get(path)
            if previous is not None:
                raise ValueError(f"Duplicate {label} audio path in {previous} and {split}: {path}")
            seen[path] = split


def counts(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {split: len(rows_by_split[split]) for split in SPLITS}


def text_counts(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(Counter(str(row.get("text", "")) for row in rows_by_split[split]).items()))
        for split in SPLITS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input-dir", required=True, type=Path)
    parser.add_argument("--positive-train-jsonl", required=True, type=Path)
    parser.add_argument("--negative-train-jsonl", required=True, type=Path)
    parser.add_argument("--positive-eval-jsonl", required=True, type=Path)
    parser.add_argument("--negative-eval-jsonl", required=True, type=Path)
    parser.add_argument("--eval-generated-json", required=True, type=Path)
    parser.add_argument("--eval-failed-json", required=True, type=Path)
    parser.add_argument("--variants-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--old-wakeword", default="Hey Siri")
    parser.add_argument("--validation-group-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Do not verify every newly referenced audio file exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Base input directory does not exist: {base}")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    variants = load_variants(args.variants_json.expanduser().resolve())

    positive_map, positive_group_ids = emitted_group_map(
        args.eval_generated_json,
        args.eval_failed_json,
        output_start=0,
        output_end=16,
    )
    negative_map, negative_group_ids = emitted_group_map(
        args.eval_generated_json,
        args.eval_failed_json,
        output_start=17,
        output_end=32,
    )
    if positive_group_ids != negative_group_ids:
        raise ValueError("Positive and negative provenance group order differs")
    if not 0 < args.validation_group_count < len(positive_group_ids):
        raise ValueError("--validation-group-count must be between 1 and group_count - 1")
    validation_groups = set(
        random.Random(args.seed).sample(sorted(positive_group_ids), args.validation_group_count)
    )
    check_audio = not args.skip_audio_check

    old_positive: dict[str, list[dict[str, Any]]] = {}
    removed_old: dict[str, int] = {}
    old_key = normalized_text(args.old_wakeword)
    for split in SPLITS:
        manifest = base / "positive_wuw_audio" / f"{split}.jsonl"
        retained: list[dict[str, Any]] = []
        removed = 0
        for row in read_jsonl(manifest):
            if normalized_text(str(row.get("text", ""))) == old_key:
                removed += 1
            else:
                retained.append(canonicalize(row, manifest, check_audio=False))
        old_positive[split] = retained
        removed_old[split] = removed

    old_negative: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        manifest = base / "negative_non_wuw_audio" / f"{split}.jsonl"
        old_negative[split] = [
            canonicalize(row, manifest, check_audio=False) for row in read_jsonl(manifest)
        ]

    new_positive_train, dropped_positive_train = prepare_new_positive(
        args.positive_train_jsonl,
        variants,
        check_audio=check_audio,
    )
    new_negative_train, dropped_negative_train = prepare_new_negative(
        args.negative_train_jsonl,
        check_audio=check_audio,
    )
    new_positive_eval, dropped_positive_eval = prepare_new_positive(
        args.positive_eval_jsonl,
        variants,
        check_audio=check_audio,
        group_map=positive_map,
        validation_groups=validation_groups,
    )
    new_negative_eval, dropped_negative_eval = prepare_new_negative(
        args.negative_eval_jsonl,
        check_audio=check_audio,
        group_map=negative_map,
        validation_groups=validation_groups,
    )

    positive = {
        split: old_positive[split]
        + new_positive_train[split]
        + new_positive_eval[split]
        for split in SPLITS
    }
    negative = {
        split: old_negative[split]
        + new_negative_train[split]
        + new_negative_eval[split]
        for split in SPLITS
    }
    split_paths(positive, "positive")
    split_paths(negative, "negative")
    if any(
        normalized_text(str(row.get("text", ""))) == old_key
        for rows in positive.values()
        for row in rows
    ):
        raise RuntimeError(f"Legacy wake word {args.old_wakeword!r} remains after filtering")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for category in ("background", "negative_car_control", "negative_cv-en-train_phoneme"):
            source = base / category
            if not source.is_dir():
                raise FileNotFoundError(f"Missing base input category: {source}")
            shutil.copytree(source, temporary / category)
        for split in SPLITS:
            write_jsonl(temporary / "positive_wuw_audio" / f"{split}.jsonl", positive[split])
            write_jsonl(temporary / "negative_non_wuw_audio" / f"{split}.jsonl", negative[split])

        summary = {
            "schema_version": 1,
            "base_input_dir": str(base),
            "output_dir": str(output),
            "old_wakeword_removed": args.old_wakeword,
            "removed_old_positive_rows": removed_old,
            "no_speech_rows_removed": {
                "positive_train": dropped_positive_train,
                "negative_train": dropped_negative_train,
                "positive_eval": dropped_positive_eval,
                "negative_eval": dropped_negative_eval,
            },
            "seed": args.seed,
            "validation_group_count": len(validation_groups),
            "test_group_count": len(positive_group_ids) - len(validation_groups),
            "validation_group_ids": sorted(validation_groups),
            "positive_rows": counts(positive),
            "negative_non_wuw_rows": counts(negative),
            "positive_text_counts": text_counts(positive),
            "new_positive_variant_counts": dict(
                sorted(
                    Counter(
                        str(row["expected_keyword_id"])
                        for rows in (new_positive_train, new_positive_eval)
                        for split_rows in rows.values()
                        for row in split_rows
                    ).items()
                )
            ),
            "canonical_audio_field": "path",
            "absolute_audio_paths": True,
            "audio_files_copied": False,
        }
        (temporary / "preparation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for category, rows in (("positive_wuw_audio", positive), ("negative_non_wuw_audio", negative)):
            (temporary / category / "split_summary.json").write_text(
                json.dumps(
                    {
                        "category": category,
                        "rows": counts(rows),
                        "validation_group_ids": sorted(validation_groups),
                        "seed": args.seed,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
