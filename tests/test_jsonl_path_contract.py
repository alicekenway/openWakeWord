"""Regression tests for the one-audio-path JSONL contract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TOOL_PATH = Path(__file__).resolve().parents[1] / "data_processing_tools" / "normalize_jsonl_audio_paths.py"
SPEC = importlib.util.spec_from_file_location("normalize_jsonl_audio_paths", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
normalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = normalizer
SPEC.loader.exec_module(normalizer)


def test_relative_normalization_is_canonical_and_idempotent(tmp_path: Path) -> None:
    data_root = tmp_path / "wuw_data"
    dataset = data_root / "wuw_audio"
    audio = dataset / "wav" / "clip.wav"
    audio.parent.mkdir(parents=True)
    audio.touch()
    manifest = dataset / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audiofile_path": "wav/clip.wav",
                "path": str(audio),
                "source_path": "/old/audio.wav",
                "vad_trim": {"source_path": "/older/audio.wav", "threshold": 0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = normalizer.normalize_file(
        manifest,
        absolute=False,
        relative_to=data_root,
        fallback_root=None,
        dry_run=False,
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert first.changed_rows == 1
    assert record == {
        "path": "wuw_audio/wav/clip.wav",
        "vad_trim": {"threshold": 0.5},
    }

    second = normalizer.normalize_file(
        manifest,
        absolute=False,
        relative_to=data_root,
        fallback_root=None,
        dry_run=False,
    )
    assert second.changed_rows == 0


def test_training_normalization_writes_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    audio = source / "wav" / "clip.wav"
    audio.parent.mkdir(parents=True)
    audio.touch()
    source_manifest = source / "train.jsonl"
    source_manifest.touch()
    training_manifest = tmp_path / "training" / "train.jsonl"
    training_manifest.parent.mkdir()
    training_manifest.write_text(
        json.dumps(
            {
                "audiofile_path": "wav/clip.wav",
                "input_jsonl": str(source_manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    normalizer.normalize_file(
        training_manifest,
        absolute=True,
        relative_to=None,
        fallback_root=None,
        dry_run=False,
    )

    assert json.loads(training_manifest.read_text(encoding="utf-8"))["path"] == str(audio.resolve())


def test_jsonl_discovery_does_not_follow_manifest_symlinks(tmp_path: Path) -> None:
    external = tmp_path / "external.jsonl"
    external.write_text('{"path": "audio.wav"}\n', encoding="utf-8")
    training = tmp_path / "training"
    training.mkdir()
    (training / "linked.jsonl").symlink_to(external)
    local = training / "local.jsonl"
    local.write_text('{"path": "/audio.wav"}\n', encoding="utf-8")

    assert normalizer.jsonl_files([training], recursive=True) == [local]
    assert normalizer.jsonl_files([training], recursive=True, materialize_symlinks=True) == [
        training / "linked.jsonl",
        local,
    ]
