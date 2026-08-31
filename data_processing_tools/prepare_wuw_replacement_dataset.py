#!/usr/bin/env python3
"""Build replacement wake-word manifests without copying audio.

The tool starts from an existing ``input_data`` tree, removes one legacy
positive phrase, adds new positive and hard-negative TTS manifests, and makes
a group-safe validation/test/holdout split from a generated CosyVoice
evaluation set.  Positive and non-WUW manifests are normalized to the legacy
two-field ``path``/``text`` contract.
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
NEW_SPLITS = (*SPLITS, "holdout")
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
    text = " ".join(str(record.get("text", "")).split())
    if not text:
        raise ValueError(f"JSONL row in {manifest} has empty text")
    return {"path": str(resolved), "text": text}


def load_trimmed_replacements(
    manifest: Path,
) -> dict[str, tuple[dict[str, Any], Path]]:
    """Map original absolute audio paths to their VAD-trimmed manifest rows."""

    summary_path = manifest.with_name("metadata.summary.json")
    summary = read_json(summary_path)
    if not isinstance(summary, dict) or not summary.get("input_jsonl"):
        raise ValueError(f"{summary_path} must identify the original input_jsonl")
    source_manifest = Path(str(summary["input_jsonl"])).expanduser().resolve()
    source_rows = read_jsonl(source_manifest)
    trimmed_rows = read_jsonl(manifest)
    if len(source_rows) != len(trimmed_rows):
        raise ValueError(
            f"Trimmed replacement row count differs from its source: "
            f"{len(trimmed_rows)} != {len(source_rows)}"
        )
    replacements: dict[str, tuple[dict[str, Any], Path]] = {}
    for index, (source_row, trimmed_row) in enumerate(zip(source_rows, trimmed_rows, strict=True)):
        if trimmed_source_index(trimmed_row, manifest) != index:
            raise ValueError(f"Trimmed replacement index mismatch at row {index} in {manifest}")
        source_path = canonicalize(source_row, source_manifest, check_audio=False)["path"]
        if source_path in replacements:
            raise ValueError(f"Duplicate source path in replacement manifest: {source_path}")
        replacements[source_path] = (trimmed_row, manifest)
    return replacements


def prepare_base_rows(
    manifest: Path,
    *,
    excluded_texts: set[str],
    replacement_manifest: Path | None,
    check_audio: bool,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    replacements: dict[str, tuple[dict[str, Any], Path]] = {}
    if replacement_manifest is not None:
        replacements = load_trimmed_replacements(replacement_manifest)
    output: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    dropped_no_speech = 0
    used_replacements: set[str] = set()
    for row in read_jsonl(manifest):
        canonical = canonicalize(row, manifest, check_audio=False)
        replacement = replacements.get(canonical["path"])
        if replacement is not None:
            replacement_row, trimmed_manifest = replacement
            used_replacements.add(canonical["path"])
            candidate = canonicalize(replacement_row, trimmed_manifest, check_audio=check_audio)
            if no_speech(replacement_row, trimmed_manifest):
                dropped_no_speech += 1
                continue
        else:
            candidate = canonical
        text_key = normalized_text(candidate["text"])
        if text_key in excluded_texts:
            removed[candidate["text"]] += 1
            continue
        output.append(candidate)
    if replacements:
        unused_nonexcluded = [
            row
            for source_path, (row, _trimmed_manifest) in replacements.items()
            if source_path not in used_replacements
            and normalized_text(str(row.get("text", ""))) not in excluded_texts
        ]
        if unused_nonexcluded:
            examples = sorted({str(row.get("text", "")) for row in unused_nonexcluded})[:5]
            raise ValueError(
                f"{manifest} does not contain {len(unused_nonexcluded)} non-excluded rows from "
                f"{replacement_manifest}; examples: {examples}"
            )
    return output, removed, dropped_no_speech


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
    group_splits: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int, Counter[str]]:
    output = {split: [] for split in NEW_SPLITS}
    dropped = 0
    variant_counts: Counter[str] = Counter()
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
        variant_counts[keyword_id] += 1
        split = "train"
        if group_id is not None:
            if group_splits is None or group_id not in group_splits:
                raise ValueError(f"No split assignment for evaluation group {group_id}")
            split = group_splits[group_id]
        output[split].append(updated)
    if group_map is not None and len(seen_source_indexes) != len(group_map):
        raise ValueError(
            f"Provenance map has {len(group_map)} rows but {manifest} has "
            f"{len(seen_source_indexes)} unique source indexes"
        )
    return output, dropped, variant_counts


def prepare_new_negative(
    manifest: Path,
    *,
    check_audio: bool,
    group_map: list[str] | None = None,
    group_splits: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    output = {split: [] for split in NEW_SPLITS}
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
            if group_splits is None or group_id not in group_splits:
                raise ValueError(f"No split assignment for evaluation group {group_id}")
            split = group_splits[group_id]
        output[split].append(updated)
    if group_map is not None and len(seen_source_indexes) != len(group_map):
        raise ValueError(
            f"Provenance map has {len(group_map)} rows but {manifest} has "
            f"{len(seen_source_indexes)} unique source indexes"
        )
    return output, dropped


def split_paths(rows_by_split: dict[str, list[dict[str, Any]]], label: str) -> None:
    seen: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
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
    parser.add_argument("--old-wakeword", default="Hey Siri", help=argparse.SUPPRESS)
    parser.add_argument(
        "--exclude-text",
        action="append",
        default=None,
        help="Exact phrase to remove case-insensitively; repeat for multiple phrases",
    )
    parser.add_argument("--positive-test-trimmed-jsonl", type=Path)
    parser.add_argument("--negative-test-trimmed-jsonl", type=Path)
    parser.add_argument("--validation-group-count", type=int, default=10)
    parser.add_argument("--test-group-count", type=int, default=10)
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
    if args.validation_group_count < 1 or args.test_group_count < 1:
        raise ValueError("--validation-group-count and --test-group-count must both be positive")
    if args.validation_group_count + args.test_group_count >= len(positive_group_ids):
        raise ValueError("validation and test groups must leave at least one full-test holdout group")
    ordered_groups = sorted(positive_group_ids)
    rng = random.Random(args.seed)
    validation_groups = set(rng.sample(ordered_groups, args.validation_group_count))
    remaining_groups = [group_id for group_id in ordered_groups if group_id not in validation_groups]
    test_groups = set(rng.sample(remaining_groups, args.test_group_count))
    holdout_groups = set(remaining_groups) - test_groups
    group_splits = {
        group_id: (
            "val" if group_id in validation_groups else "test" if group_id in test_groups else "holdout"
        )
        for group_id in ordered_groups
    }
    check_audio = not args.skip_audio_check
    excluded_values = args.exclude_text if args.exclude_text is not None else [args.old_wakeword]
    excluded_texts = {normalized_text(value) for value in excluded_values}
    if "" in excluded_texts:
        raise ValueError("--exclude-text values must not be empty")

    old_positive: dict[str, list[dict[str, Any]]] = {}
    removed_old_positive: dict[str, dict[str, int]] = {}
    dropped_old_positive_no_speech: dict[str, int] = {}
    for split in SPLITS:
        manifest = base / "positive_wuw_audio" / f"{split}.jsonl"
        retained, removed, dropped = prepare_base_rows(
            manifest,
            excluded_texts=excluded_texts,
            replacement_manifest=(
                args.positive_test_trimmed_jsonl.expanduser().resolve()
                if split == "test" and args.positive_test_trimmed_jsonl is not None
                else None
            ),
            check_audio=check_audio,
        )
        old_positive[split] = retained
        removed_old_positive[split] = dict(sorted(removed.items()))
        dropped_old_positive_no_speech[split] = dropped

    old_negative: dict[str, list[dict[str, Any]]] = {}
    removed_old_negative: dict[str, dict[str, int]] = {}
    dropped_old_negative_no_speech: dict[str, int] = {}
    for split in SPLITS:
        manifest = base / "negative_non_wuw_audio" / f"{split}.jsonl"
        retained, removed, dropped = prepare_base_rows(
            manifest,
            excluded_texts=excluded_texts,
            replacement_manifest=(
                args.negative_test_trimmed_jsonl.expanduser().resolve()
                if split == "test" and args.negative_test_trimmed_jsonl is not None
                else None
            ),
            check_audio=check_audio,
        )
        old_negative[split] = retained
        removed_old_negative[split] = dict(sorted(removed.items()))
        dropped_old_negative_no_speech[split] = dropped

    new_positive_train, dropped_positive_train, train_variant_counts = prepare_new_positive(
        args.positive_train_jsonl,
        variants,
        check_audio=check_audio,
    )
    new_negative_train, dropped_negative_train = prepare_new_negative(
        args.negative_train_jsonl,
        check_audio=check_audio,
    )
    new_positive_eval, dropped_positive_eval, eval_variant_counts = prepare_new_positive(
        args.positive_eval_jsonl,
        variants,
        check_audio=check_audio,
        group_map=positive_map,
        group_splits=group_splits,
    )
    new_negative_eval, dropped_negative_eval = prepare_new_negative(
        args.negative_eval_jsonl,
        check_audio=check_audio,
        group_map=negative_map,
        group_splits=group_splits,
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
    split_paths(
        {**positive, "holdout": new_positive_eval["holdout"]},
        "positive including full-test holdout",
    )
    split_paths(
        {**negative, "holdout": new_negative_eval["holdout"]},
        "negative including full-test holdout",
    )
    if any(
        normalized_text(str(row.get("text", ""))) in excluded_texts
        for rows_by_split in (positive, negative)
        for rows in rows_by_split.values()
        for row in rows
    ):
        raise RuntimeError("An excluded phrase remains after filtering")

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
        write_jsonl(
            temporary / "full_test_holdout" / "positive_wuw_audio.jsonl",
            new_positive_eval["holdout"],
        )
        write_jsonl(
            temporary / "full_test_holdout" / "negative_non_wuw_audio.jsonl",
            new_negative_eval["holdout"],
        )

        summary = {
            "schema_version": 1,
            "base_input_dir": str(base),
            "output_dir": str(output),
            "excluded_texts": excluded_values,
            "removed_old_positive_rows": removed_old_positive,
            "removed_old_negative_rows": removed_old_negative,
            "trimmed_test_replacements": {
                "positive_wuw_audio": (
                    str(args.positive_test_trimmed_jsonl.expanduser().resolve())
                    if args.positive_test_trimmed_jsonl is not None
                    else None
                ),
                "negative_non_wuw_audio": (
                    str(args.negative_test_trimmed_jsonl.expanduser().resolve())
                    if args.negative_test_trimmed_jsonl is not None
                    else None
                ),
            },
            "no_speech_rows_removed": {
                "base_positive_test_replacement": dropped_old_positive_no_speech["test"],
                "base_negative_test_replacement": dropped_old_negative_no_speech["test"],
                "positive_train": dropped_positive_train,
                "negative_train": dropped_negative_train,
                "positive_eval": dropped_positive_eval,
                "negative_eval": dropped_negative_eval,
            },
            "seed": args.seed,
            "validation_group_count": len(validation_groups),
            "test_group_count": len(test_groups),
            "full_test_holdout_group_count": len(holdout_groups),
            "validation_group_ids": sorted(validation_groups),
            "test_group_ids": sorted(test_groups),
            "full_test_holdout_group_ids": sorted(holdout_groups),
            "positive_rows": counts(positive),
            "negative_non_wuw_rows": counts(negative),
            "positive_text_counts": text_counts(positive),
            "full_test_holdout_rows": {
                "positive_wuw_audio": len(new_positive_eval["holdout"]),
                "negative_non_wuw_audio": len(new_negative_eval["holdout"]),
            },
            "new_positive_source_variant_counts": dict(
                sorted((train_variant_counts + eval_variant_counts).items())
            ),
            "canonical_audio_field": "path",
            "manifest_fields": ["path", "text"],
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
                        "test_group_ids": sorted(test_groups),
                        "full_test_holdout_group_ids": sorted(holdout_groups),
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
