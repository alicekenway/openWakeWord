"""Tests for the group-safe JSONL splitting command."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "data_processing_tools" / "group_split_jsonl.py"
SPEC = importlib.util.spec_from_file_location("group_split_jsonl", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
group_split = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = group_split
SPEC.loader.exec_module(group_split)


def test_audio_base_path_alias_writes_joined_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_base = tmp_path / "audio"
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps({"path": "clips/example.wav", "text": "hello"}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "split"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "group_split_jsonl.py",
            "--input-jsonl",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--set-names",
            "train",
            "--group-counts",
            "1",
            "--no-shuffle",
            "--audio-base-path",
            str(audio_base),
        ],
    )

    group_split.main()

    output = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    assert output == {"path": str(audio_base / "clips/example.wav"), "text": "hello"}
