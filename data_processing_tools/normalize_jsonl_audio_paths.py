#!/usr/bin/env python3
"""Normalize JSONL rows to one canonical ``path`` audio field.

This is a metadata-only migration tool.  It never reads, writes, augments, or
trims audio.  JSONL files are rewritten atomically after every row validates.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


AUDIO_PATH_KEYS = ("path", "audiofile_path", "audio_file", "audio_path", "file", "filename")


@dataclass
class FileStats:
    path: Path
    rows: int = 0
    audio_rows: int = 0
    changed_rows: int = 0


def detect_audio_path(record: dict[str, Any]) -> str | None:
    for key in AUDIO_PATH_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    return None


def absolute_path(path: Path) -> Path:
    """Make a path absolute lexically, without one filesystem lookup per row."""
    return Path(os.path.abspath(os.fspath(path)))


@lru_cache(maxsize=16)
def root_child_names(root: str) -> frozenset[str]:
    return frozenset(path.name for path in Path(root).iterdir())


def resolve_audio_path(
    raw_path: str,
    record: dict[str, Any],
    manifest: Path,
    relative_to: Path | None,
    fallback_root: Path | None,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return absolute_path(path)
    first_part = path.parts[0] if path.parts else ""
    if relative_to is not None:
        root = absolute_path(relative_to)
        base = root if first_part in root_child_names(str(root)) else manifest.parent
        return absolute_path(base / path)
    input_jsonl = record.get("input_jsonl")
    if input_jsonl:
        input_manifest = Path(str(input_jsonl)).expanduser()
        if input_manifest.is_absolute():
            return absolute_path(input_manifest.parent / path)
    if fallback_root is not None:
        root = absolute_path(fallback_root)
        if first_part in root_child_names(str(root)):
            return absolute_path(root / path)
    return absolute_path(manifest.parent / path)


def canonicalize_record(
    record: dict[str, Any],
    manifest: Path,
    *,
    absolute: bool,
    relative_to: Path | None,
    fallback_root: Path | None,
) -> tuple[dict[str, Any], bool]:
    raw_path = detect_audio_path(record)
    if raw_path is None:
        return dict(record), False

    resolved = resolve_audio_path(raw_path, record, manifest, relative_to, fallback_root)
    if relative_to is not None:
        try:
            output_path = resolved.relative_to(absolute_path(relative_to))
        except ValueError as exc:
            raise ValueError(
                f"Audio path {resolved} from {manifest} is outside relative root {relative_to}"
            ) from exc
    elif absolute:
        output_path = resolved
    else:
        output_path = Path(raw_path)

    updated = dict(record)
    for key in AUDIO_PATH_KEYS:
        updated.pop(key, None)
    updated.pop("source_path", None)
    vad_trim = updated.get("vad_trim")
    if isinstance(vad_trim, dict) and "source_path" in vad_trim:
        updated["vad_trim"] = {key: value for key, value in vad_trim.items() if key != "source_path"}
    updated["path"] = str(output_path)
    return updated, updated != record


def jsonl_files(targets: Iterable[Path], recursive: bool, materialize_symlinks: bool = False) -> list[Path]:
    files: set[Path] = set()
    for target in targets:
        resolved = absolute_path(target.expanduser())
        if resolved.is_symlink():
            if materialize_symlinks and resolved.suffix == ".jsonl" and resolved.is_file():
                files.add(resolved)
            continue
        if resolved.is_file():
            if resolved.suffix == ".jsonl":
                files.add(resolved)
            continue
        if not resolved.is_dir():
            raise FileNotFoundError(f"JSONL target does not exist: {resolved}")
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        files.update(
            absolute_path(path)
            for path in resolved.glob(pattern)
            if path.is_file() and (materialize_symlinks or not path.is_symlink())
        )
    return sorted(files)


def normalize_file(
    path: Path,
    *,
    absolute: bool,
    relative_to: Path | None,
    fallback_root: Path | None,
    dry_run: bool,
) -> FileStats:
    stats = FileStats(path=path)
    temporary_path: Path | None = None
    handle = None
    if not dry_run:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
        temporary_path = Path(handle.name)
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL row must be an object in {path}:{line_number}")
                stats.rows += 1
                updated, changed = canonicalize_record(
                    record,
                    path,
                    absolute=absolute,
                    relative_to=relative_to,
                    fallback_root=fallback_root,
                )
                if detect_audio_path(record) is not None:
                    stats.audio_rows += 1
                stats.changed_rows += int(changed)
                if handle is not None:
                    handle.write(json.dumps(updated, ensure_ascii=False, sort_keys=True) + "\n")
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            assert temporary_path is not None
            if stats.changed_rows:
                os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
                os.replace(temporary_path, path)
            else:
                temporary_path.unlink()
            temporary_path = None
    except Exception:
        if handle is not None:
            handle.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path, help="JSONL file or directory")
    parser.add_argument("--recursive", action="store_true", help="Find JSONL files recursively in directories")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--absolute", action="store_true", help="Write absolute canonical audio paths")
    mode.add_argument(
        "--relative-to",
        type=Path,
        help="Write canonical audio paths relative to this root",
    )
    parser.add_argument(
        "--fallback-root",
        type=Path,
        help="Last base to try when resolving an existing relative path",
    )
    parser.add_argument(
        "--materialize-symlinks",
        action="store_true",
        help="Replace JSONL symlinks with normalized regular files; symlinks are skipped by default",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without changing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = jsonl_files(args.targets, args.recursive, args.materialize_symlinks)
    if not files:
        raise FileNotFoundError("No JSONL files found")
    totals = FileStats(path=Path("<total>"))
    for path in files:
        result = normalize_file(
            path,
            absolute=bool(args.absolute),
            relative_to=args.relative_to.resolve() if args.relative_to else None,
            fallback_root=args.fallback_root.resolve() if args.fallback_root else None,
            dry_run=bool(args.dry_run),
        )
        totals.rows += result.rows
        totals.audio_rows += result.audio_rows
        totals.changed_rows += result.changed_rows
        print(
            f"{path}: rows={result.rows} audio_rows={result.audio_rows} "
            f"changed_rows={result.changed_rows}"
        )
    action = "would change" if args.dry_run else "changed"
    print(
        f"Processed {len(files)} file(s), {totals.rows} row(s), "
        f"{totals.audio_rows} audio row(s); {action} {totals.changed_rows} row(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
