"""Small, dependency-light artifact and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ConfigurationError, IniConfig, parse_json


AUDIO_PATH_KEYS = ("path", "audiofile_path", "audio_file", "audio_path", "file", "filename")


def replace_audio_path(record: dict[str, Any], value: str | Path) -> dict[str, Any]:
    """Return a record with exactly one canonical audio path field.

    Legacy manifests used several aliases at once and retained the previous
    audio as ``source_path``.  Generated manifests only need the audio that the
    next stage should consume, so replace all aliases and historical audio
    paths with a single ``path`` value.
    """
    updated = dict(record)
    for key in AUDIO_PATH_KEYS:
        updated.pop(key, None)
    updated.pop("source_path", None)
    vad_trim = updated.get("vad_trim")
    if isinstance(vad_trim, dict) and "source_path" in vad_trim:
        updated["vad_trim"] = {key: item for key, item in vad_trim.items() if key != "source_path"}
    updated["path"] = str(value)
    return updated


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        return value.item()
    except AttributeError:
        return str(value)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ConfigurationError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ConfigurationError(f"JSONL record in {path}:{line_number} must be an object")
                records.append(record)
    except OSError as exc:
        raise ConfigurationError(f"Could not read JSONL manifest {path}: {exc}") from exc
    if not records and not allow_empty:
        raise ConfigurationError(f"JSONL manifest is empty: {path}")
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")
            count += 1
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return count


def stable_id(value: str, index: int = 0) -> str:
    return hashlib.sha1(f"{index}:{value}".encode("utf-8")).hexdigest()[:16]


def hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=json_default, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


@dataclass(frozen=True)
class ManifestInput:
    path: Path
    audio_base_dir: Path | None


def _parse_manifest_value(config: IniConfig, section: str, key: str) -> list[str | dict[str, Any]]:
    raw = config.get(section, key)
    assert raw is not None
    stripped = raw.strip()
    if stripped.startswith("["):
        value = parse_json(stripped, f"[{section}] {key}", list)
        return value
    return [raw]


def parse_manifest_inputs(
    config: IniConfig,
    section: str,
    *,
    key: str = "input_jsonl",
    base_key: str = "audio_base_dir",
    required: bool = True,
) -> list[ManifestInput]:
    if not config.parser.has_option(section, key):
        if required:
            raise ConfigurationError(f"Missing required option [{section}] {key}")
        return []
    default_base_text = config.get(section, base_key, required=False)
    default_base = config.resolve_path(default_base_text) if default_base_text else None
    items = _parse_manifest_value(config, section, key)
    if not items:
        raise ConfigurationError(f"[{section}] {key} cannot be empty")

    result: list[ManifestInput] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            result.append(ManifestInput(config.resolve_path(item), default_base))
            continue
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            base_text = item.get("audio_base_dir")
            base = config.resolve_path(base_text) if base_text else default_base
            result.append(ManifestInput(config.resolve_path(item["path"]), base))
            continue
        raise ConfigurationError(
            f"[{section}] {key}[{index}] must be a JSONL path or {{\"path\": ..., \"audio_base_dir\": ...}}"
        )
    return result


def resolve_record_path(record: dict[str, Any], manifest: Path, audio_base_dir: Path | None) -> Path:
    for key in AUDIO_PATH_KEYS:
        value = record.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (audio_base_dir or manifest.parent) / path
            return path.resolve()
    raise ConfigurationError(f"JSONL record in {manifest} has no supported audio path key ({', '.join(AUDIO_PATH_KEYS)})")


def normalise_manifest_inputs(
    inputs: list[ManifestInput],
    output_path: Path,
    *,
    default_placement: str | None = None,
    label: int | None = None,
    source: str | None = None,
) -> tuple[Path, int]:
    """Combine manifests and turn all audio paths into absolute ``path`` values."""
    records: list[dict[str, Any]] = []
    for dataset_index, manifest_input in enumerate(inputs):
        raw_records = read_jsonl(manifest_input.path)
        for record_index, raw_record in enumerate(raw_records):
            path = resolve_record_path(raw_record, manifest_input.path, manifest_input.audio_base_dir)
            record = replace_audio_path(raw_record, path)
            record.setdefault("id", stable_id(str(path), record_index))
            record["input_jsonl"] = str(manifest_input.path)
            record["dataset_index"] = dataset_index
            if default_placement and not record.get("placement"):
                record["placement"] = default_placement
            if label is not None:
                record["label"] = int(label)
            if source and not record.get("source"):
                record["source"] = source
            records.append(record)
    write_jsonl(output_path, records)
    return output_path, len(records)


def input_signatures(inputs: list[ManifestInput]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for item in inputs:
        signature = file_signature(item.path)
        signature["audio_base_dir"] = str(item.audio_base_dir) if item.audio_base_dir else None
        signatures.append(signature)
    return signatures
