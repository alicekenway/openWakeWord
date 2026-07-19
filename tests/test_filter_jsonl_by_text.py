"""Tests for exact-text JSONL filtering."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "data_processing_tools" / "filter_jsonl_by_text.py"
SPEC = importlib.util.spec_from_file_location("filter_jsonl_by_text", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
text_filter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = text_filter
SPEC.loader.exec_module(text_filter)


def test_filter_keeps_only_records_without_exact_excluded_text(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"path": "one.wav", "text": "keep me"}),
                json.dumps({"path": "two.wav", "text": "discard me"}),
                json.dumps({"path": "three.wav", "text": "discard me "}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("discard me\n\n", encoding="utf-8")
    output_jsonl = tmp_path / "output.jsonl"

    summary = text_filter.filter_jsonl(
        input_jsonl,
        output_jsonl,
        text_filter.read_excluded_texts(excluded),
        text_field="text",
    )

    retained = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [record["path"] for record in retained] == ["one.wav", "three.wav"]
    assert summary["input_rows"] == 3
    assert summary["removed_rows"] == 1
    assert summary["written_rows"] == 2


def test_filter_rejects_records_without_the_requested_text_field(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text(json.dumps({"path": "one.wav"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has no 'text' field"):
        text_filter.filter_jsonl(input_jsonl, tmp_path / "output.jsonl", {"discard me"}, text_field="text")
