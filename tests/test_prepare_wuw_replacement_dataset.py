from __future__ import annotations

import json
import sys
from pathlib import Path

from data_processing_tools import prepare_wuw_replacement_dataset as tool


VARIANTS = [
    {
        "id": "hello_loncin_aa_n",
        "display_text": "Hello Loncin",
        "source_text": "Hello [L][AA1][N][S][IH0][N]",
        "arpabet": "[HH][AH0][L][OW1][L][AA1][N][S][IH0][N]",
    },
    {
        "id": "hello_loncin_ao_n",
        "display_text": "Hello Loncin",
        "source_text": "Hello [L][AO1][N][S][IH0][N]",
        "arpabet": "[HH][AH0][L][OW1][L][AO1][N][S][IH0][N]",
    },
    {
        "id": "hello_loncin_ao_ng",
        "display_text": "Hello Loncin",
        "source_text": "Hello [L][AO1][NG][S][IH0][N]",
        "arpabet": "[HH][AH0][L][OW1][L][AO1][NG][S][IH0][N]",
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def audio_row(path: Path, text: str, *, no_speech: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return {
        "path": str(path),
        "text": text,
        "vad_trim": {"no_speech": no_speech, "trimmed_duration_seconds": 1.0},
    }


def test_builds_group_safe_replacement_dataset(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base"
    for category in ("background", "negative_car_control", "negative_cv-en-train_phoneme"):
        directory = base / category
        directory.mkdir(parents=True)
        (directory / "marker.txt").write_text(category, encoding="utf-8")
    for split in tool.SPLITS:
        write_jsonl(
            base / "positive_wuw_audio" / f"{split}.jsonl",
            [
                {"path": f"/old/{split}_hey.wav", "text": "Hey Siri"},
                {"path": f"/old/{split}_command.wav", "text": "Open Homepage"},
            ],
        )
        write_jsonl(
            base / "negative_non_wuw_audio" / f"{split}.jsonl",
            [{"path": f"/old/{split}_negative.wav", "text": "Home page"}],
        )

    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"variants": VARIANTS}), encoding="utf-8")
    positive_train = write_jsonl(
        tmp_path / "positive_train" / "metadata.jsonl",
        [
            audio_row(tmp_path / "positive_train/wav/a.wav", VARIANTS[0]["source_text"]),
            audio_row(
                tmp_path / "positive_train/wav/silent.wav",
                VARIANTS[1]["source_text"],
                no_speech=True,
            ),
        ],
    )
    negative_train = write_jsonl(
        tmp_path / "negative_train" / "metadata.jsonl",
        [audio_row(tmp_path / "negative_train/wav/a.wav", "Hello London")],
    )

    groups = []
    failures = []
    for group_index in range(4):
        outputs = []
        for output_index in range(33):
            outputs.append(
                {
                    "text": (
                        VARIANTS[output_index % 3]["source_text"]
                        if output_index < 17
                        else "Hello London"
                    ),
                    "candidate_audio_path": [f"missing/{group_index}_{output_index}.wav"],
                }
            )
        groups.append({"id": f"group_{group_index:06d}", "output": outputs})
    failures.extend(
        [
            {"id": "group_000001", "text_index": 3, "reference_index": 0},
            {"id": "group_000002", "text_index": 20, "reference_index": 0},
        ]
    )
    generated_path = tmp_path / "generated.json"
    failed_path = tmp_path / "failed.json"
    generated_path.write_text(json.dumps(groups), encoding="utf-8")
    failed_path.write_text(json.dumps(failures), encoding="utf-8")

    positive_map, _ = tool.emitted_group_map(generated_path, failed_path, output_start=0, output_end=16)
    negative_map, _ = tool.emitted_group_map(generated_path, failed_path, output_start=17, output_end=32)

    def trimmed_manifest(directory: Path, mapping: list[str], positive: bool) -> Path:
        rows = []
        for index, _group_id in enumerate(mapping):
            audio = directory / "wav" / f"00000000_{index:09d}_aaaaaaaaaa.wav"
            text = VARIANTS[index % 3]["source_text"] if positive else "Hello London"
            rows.append(audio_row(audio, text, no_speech=index == 0))
        return write_jsonl(directory / "metadata.jsonl", rows)

    positive_eval = trimmed_manifest(tmp_path / "positive_eval", positive_map, True)
    negative_eval = trimmed_manifest(tmp_path / "negative_eval", negative_map, False)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_wuw_replacement_dataset.py",
            "--base-input-dir", str(base),
            "--positive-train-jsonl", str(positive_train),
            "--negative-train-jsonl", str(negative_train),
            "--positive-eval-jsonl", str(positive_eval),
            "--negative-eval-jsonl", str(negative_eval),
            "--eval-generated-json", str(generated_path),
            "--eval-failed-json", str(failed_path),
            "--variants-json", str(variants_path),
            "--output-dir", str(output),
            "--validation-group-count", "1",
            "--seed", "1337",
        ],
    )

    assert tool.main() == 0
    summary = json.loads((output / "preparation_summary.json").read_text(encoding="utf-8"))
    assert summary["removed_old_positive_rows"] == {"train": 1, "val": 1, "test": 1}
    assert summary["validation_group_count"] == 1
    assert summary["test_group_count"] == 3
    assert sum(summary["no_speech_rows_removed"].values()) == 3

    positive_rows = {
        split: tool.read_jsonl(output / "positive_wuw_audio" / f"{split}.jsonl")
        for split in tool.SPLITS
    }
    assert all(row["text"] != "Hey Siri" for rows in positive_rows.values() for row in rows)
    new_rows = [
        row for rows in positive_rows.values() for row in rows if row["text"] == "Hello Loncin"
    ]
    assert new_rows
    assert all("expected_keyword_id" in row for row in new_rows)
    eval_groups = {
        split: {row["source_group_id"] for row in positive_rows[split] if "source_group_id" in row}
        for split in ("val", "test")
    }
    assert not (eval_groups["val"] & eval_groups["test"])
