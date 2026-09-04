from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "data_processing_tools" / "decode_stage1_token_sequences.py"
SPEC = importlib.util.spec_from_file_location("decode_stage1_token_sequences", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_greedy_ctc_collapse_and_emission_spans() -> None:
    frame_ids = [0, 1, 1, 0, 1, 2, 2, 0]
    scores = np.full((len(frame_ids), 3), -10.0, dtype=np.float32)
    scores[np.arange(len(frame_ids)), frame_ids] = 0.0

    token_ids, emissions = MODULE.greedy_ctc_with_emissions(
        scores, blank_id=0, frame_shift_ms=40.0
    )

    assert token_ids == (1, 1, 2)
    assert [(row["start_frame"], row["end_frame_exclusive"], row["frame_count"]) for row in emissions] == [
        (1, 3, 2),
        (4, 5, 1),
        (5, 7, 2),
    ]
    assert [row["duration_ms"] for row in emissions] == [80.0, 40.0, 80.0]


def test_sentencepiece_rendering_omits_special_tokens() -> None:
    symbols = ("<blank>", "<unk>", "▁HEL", "LO", "▁LONCIN")
    assert MODULE.visible_pieces([0, 2, 3, 4], symbols) == ["▁HEL", "LO", "▁LONCIN"]
    assert MODULE.pieces_to_text(["▁HEL", "LO", "▁LONCIN"]) == "HELLO LONCIN"


def test_select_records_filters_then_samples_deterministically() -> None:
    rows = [{"id": index, "text": " Hello   Loncin " if index < 8 else "Other"} for index in range(10)]
    first = MODULE.select_records(rows, text="hello loncin", text_field="text", sample_size=4, seed=7)
    second = MODULE.select_records(rows, text="HELLO LONCIN", text_field="text", sample_size=4, seed=7)
    assert first == second
    assert len(first) == 4
    assert all(index < 8 for index, _record in first)


def test_load_token_table_requires_complete_contiguous_ids(tmp_path: Path) -> None:
    units = tmp_path / "units.txt"
    units.write_text("<blank> 0\n▁HEL 1\nLO 2\n", encoding="utf-8")
    assert MODULE.load_token_table(units, vocabulary_size=3) == ("<blank>", "▁HEL", "LO")
