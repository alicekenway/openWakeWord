#!/usr/bin/env python3
"""Decode a JSON/JSONL audio dataset with a streaming WeNet CTC Stage 1 model.

The tool is intended for discovering stable keyword token paths before a
``keyword_mappings`` file is written.  It performs unconstrained full-utterance
greedy and prefix-beam decoding, preserves per-token greedy emission spans, and
writes both per-record JSONL and aggregate JSON statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


AUDIO_PATH_KEYS = (
    "path",
    "audiofile_path",
    "audio_file",
    "audio_path",
    "file",
    "filename",
    "audio_filepath",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_stage1_runtime() -> tuple[Any, ...]:
    """Import the pipeline runtime while keeping this script directly runnable."""

    source_dir = _repo_root() / "training_pipline" / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from wuw_training.ctc_wac import (  # pylint: disable=import-outside-toplevel
        Stage1Contract,
        StreamingCtcStage1,
        audio_to_fbank,
        ctc_prefix_beam_search,
        load_audio,
    )

    return Stage1Contract, StreamingCtcStage1, audio_to_fbank, ctc_prefix_beam_search, load_audio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, a JSON array, or a JSON object containing a record list."""

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if not value:
                    continue
                record = json.loads(value)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(record)
        return records

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        raw_records = value
    elif isinstance(value, dict):
        candidates = [value.get(key) for key in ("records", "items", "data")]
        raw_records = next((item for item in candidates if isinstance(item, list)), None)
        if raw_records is None:
            raise ValueError(f"{path} must be a JSON array or contain records/items/data")
    else:
        raise ValueError(f"{path} must contain JSON objects")
    if not all(isinstance(record, dict) for record in raw_records):
        raise ValueError(f"{path} contains a non-object record")
    return list(raw_records)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalized_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def select_records(
    records: Sequence[dict[str, Any]],
    *,
    text: str | None,
    text_field: str,
    sample_size: int | None,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    selected = [
        (index, record)
        for index, record in enumerate(records)
        if text is None or normalized_text(record.get(text_field, "")) == normalized_text(text)
    ]
    if sample_size is None:
        return selected
    if sample_size < 1:
        raise ValueError("--sample-size must be at least 1")
    if len(selected) < sample_size:
        raise ValueError(
            f"Requested {sample_size} rows, but only {len(selected)} match the input filters"
        )
    sampled_positions = sorted(random.Random(seed).sample(range(len(selected)), sample_size))
    return [selected[position] for position in sampled_positions]


def audio_path(record: dict[str, Any], *, dataset_path: Path, audio_base_dir: Path | None) -> Path:
    raw: Any = None
    for key in AUDIO_PATH_KEYS:
        if record.get(key) not in (None, ""):
            raw = record[key]
            break
    if raw is None:
        raise ValueError(f"record has none of the supported audio keys: {', '.join(AUDIO_PATH_KEYS)}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (audio_base_dir if audio_base_dir is not None else dataset_path.parent) / path
    return path.resolve()


def load_token_table(path: Path, *, vocabulary_size: int) -> tuple[str, ...]:
    symbols: dict[int, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        fields = value.rsplit(maxsplit=1)
        if len(fields) != 2 or not fields[0]:
            raise ValueError(f"{path}:{line_number} must use 'symbol id' format")
        try:
            token_id = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} has a non-integer token ID") from exc
        if token_id in symbols:
            raise ValueError(f"{path} repeats token ID {token_id}")
        symbols[token_id] = unicodedata.normalize("NFC", fields[0])
    expected = set(range(vocabulary_size))
    if set(symbols) != expected:
        missing = sorted(expected - set(symbols))[:10]
        extra = sorted(set(symbols) - expected)[:10]
        raise ValueError(
            f"{path} does not match vocabulary_size={vocabulary_size}; "
            f"missing IDs={missing}, extra IDs={extra}"
        )
    return tuple(symbols[index] for index in range(vocabulary_size))


def visible_pieces(token_ids: Sequence[int], symbols: Sequence[str]) -> list[str]:
    pieces: list[str] = []
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        if token_id < 0 or token_id >= len(symbols):
            raise ValueError(f"Token ID {token_id} is outside the token table")
        symbol = symbols[token_id]
        if not (symbol.startswith("<") and symbol.endswith(">")):
            pieces.append(symbol)
    return pieces


def pieces_to_text(pieces: Sequence[str]) -> str:
    """Render SentencePiece-style pieces for diagnostics, without retokenizing."""

    return "".join(pieces).replace("▁", " ").strip()


def greedy_ctc_with_emissions(
    log_probs: np.ndarray, *, blank_id: int, frame_shift_ms: float
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    frame_tokens = np.argmax(np.asarray(log_probs), axis=1)
    collapsed: list[int] = []
    emissions: list[dict[str, Any]] = []
    previous: int | None = None
    active_index: int | None = None
    for frame_index, raw_token in enumerate(frame_tokens):
        token = int(raw_token)
        if token != previous:
            active_index = None
            if token != blank_id:
                collapsed.append(token)
                emissions.append(
                    {
                        "token_id": token,
                        "start_frame": frame_index,
                        "end_frame_exclusive": frame_index + 1,
                    }
                )
                active_index = len(emissions) - 1
        elif token != blank_id and active_index is not None:
            emissions[active_index]["end_frame_exclusive"] = frame_index + 1
        previous = token
    for emission in emissions:
        frame_count = int(emission["end_frame_exclusive"]) - int(emission["start_frame"])
        emission["frame_count"] = frame_count
        emission["start_ms"] = round(float(emission["start_frame"]) * frame_shift_ms, 3)
        emission["end_ms"] = round(float(emission["end_frame_exclusive"]) * frame_shift_ms, 3)
        emission["duration_ms"] = round(frame_count * frame_shift_ms, 3)
    return tuple(collapsed), emissions


def render_sequence(token_ids: Sequence[int], symbols: Sequence[str]) -> dict[str, Any]:
    ids = [int(item) for item in token_ids]
    pieces = visible_pieces(ids, symbols)
    return {"token_ids": ids, "token_pieces": pieces, "text": pieces_to_text(pieces)}


def sequence_counts_json(counts: Counter[tuple[int, ...]], symbols: Sequence[str], total: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (token_ids, count) in enumerate(
        sorted(counts.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        row = {"rank": rank, "count": count, "share": count / total if total else 0.0}
        row.update(render_sequence(token_ids, symbols))
        rows.append(row)
    return rows


def histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def infer_contract_path(model_path: Path) -> Path:
    inferred = model_path.parent / "stage1-wuw.contract.json"
    if not inferred.is_file():
        raise FileNotFoundError(
            f"Could not infer Stage 1 contract at {inferred}; pass --stage1-contract explicitly"
        )
    return inferred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL or JSON dataset")
    parser.add_argument("--audio-base-dir", type=Path, help="Base directory for relative audio paths")
    parser.add_argument("--stage1-model", required=True, type=Path)
    parser.add_argument(
        "--stage1-contract",
        type=Path,
        help="Contract JSON; defaults to stage1-wuw.contract.json beside the model",
    )
    parser.add_argument("--token-units", required=True, type=Path, help="Complete WeNet 'symbol id' table")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text", help="Keep only rows whose normalized text matches this value")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sample-size", type=int, help="Deterministically sample this many filtered rows")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--token-prune", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.stage1_model.expanduser().resolve()
    contract_path = (
        args.stage1_contract.expanduser().resolve()
        if args.stage1_contract is not None
        else infer_contract_path(model_path)
    )
    units_path = args.token_units.expanduser().resolve()
    audio_base_dir = args.audio_base_dir.expanduser().resolve() if args.audio_base_dir else None
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    Stage1Contract, StreamingCtcStage1, audio_to_fbank, prefix_beam, load_audio = load_stage1_runtime()
    contract = Stage1Contract.from_json(contract_path)
    if contract.vocab_size is None:
        raise ValueError("Stage 1 contract must declare vocab_size")
    symbols = load_token_table(units_path, vocabulary_size=int(contract.vocab_size))
    records = read_dataset(input_path)
    selected = select_records(
        records,
        text=args.text,
        text_field=args.text_field,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True)
    sampled_rows = []
    for source_index, record in selected:
        copied = dict(record)
        copied["_source_index"] = source_index
        sampled_rows.append(copied)
    write_jsonl(output_dir / "sampled_input.jsonl", sampled_rows)

    stage1 = StreamingCtcStage1(model_path, contract, device=args.device)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    greedy_counts: Counter[tuple[int, ...]] = Counter()
    beam_counts: Counter[tuple[int, ...]] = Counter()
    token_counts: Counter[int] = Counter()
    emission_frame_counts: list[int] = []

    for selected_index, (source_index, record) in enumerate(selected):
        try:
            path = audio_path(record, dataset_path=input_path, audio_base_dir=audio_base_dir)
            audio = load_audio(path, int(contract.sample_rate))
            fbank = audio_to_fbank(audio, contract)
            _encoder, ctc = stage1.infer_fbank(fbank)
            greedy_ids, emissions = greedy_ctc_with_emissions(
                ctc,
                blank_id=int(contract.blank_id),
                frame_shift_ms=float(contract.encoder_frame_shift_ms),
            )
            for emission in emissions:
                emission["token"] = symbols[int(emission["token_id"])]
                emission_frame_counts.append(int(emission["frame_count"]))
            hypotheses = prefix_beam(
                ctc,
                blank_id=int(contract.blank_id),
                beam_size=int(args.beam_size),
                token_prune=int(args.token_prune),
            )
            if not hypotheses:
                raise RuntimeError("CTC prefix beam search returned no hypotheses")
            beam_rows = []
            for hypothesis in hypotheses[: int(args.top_k)]:
                rendered = render_sequence(hypothesis.token_ids, symbols)
                rendered["log_score"] = float(hypothesis.log_score)
                rendered["log_score_per_frame"] = float(hypothesis.log_score) / int(ctc.shape[0])
                beam_rows.append(rendered)
            greedy = render_sequence(greedy_ids, symbols)
            greedy["emissions"] = emissions
            result = {
                "selected_index": selected_index,
                "source_index": source_index,
                "id": record.get("id"),
                "path": str(path),
                "text": record.get(args.text_field),
                "duration_seconds": len(audio) / int(contract.sample_rate),
                "encoder_frames": int(ctc.shape[0]),
                "greedy": greedy,
                "beam_best": beam_rows[0],
                "beam_top_k": beam_rows,
            }
            results.append(result)
            greedy_counts[greedy_ids] += 1
            beam_counts[tuple(hypotheses[0].token_ids)] += 1
            token_counts.update(greedy_ids)
        except Exception as exc:  # preserve other usable records and report the exact row
            errors.append(
                {
                    "selected_index": selected_index,
                    "source_index": source_index,
                    "id": record.get("id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (selected_index + 1) % 100 == 0:
            print(
                f"processed={selected_index + 1}/{len(selected)} "
                f"decoded={len(results)} errors={len(errors)}",
                flush=True,
            )

    write_jsonl(output_dir / "results.jsonl", results)
    write_jsonl(output_dir / "errors.jsonl", errors)
    decoded_count = len(results)
    greedy_sequence_counts = sequence_counts_json(greedy_counts, symbols, decoded_count)
    beam_sequence_counts = sequence_counts_json(beam_counts, symbols, decoded_count)
    (output_dir / "greedy_sequence_counts.json").write_text(
        json.dumps(greedy_sequence_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "beam_sequence_counts.json").write_text(
        json.dumps(beam_sequence_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    statistics = {
        "schema_version": 1,
        "input": str(input_path),
        "input_count": len(records),
        "text_filter": args.text,
        "matching_count": sum(
            1
            for record in records
            if args.text is None
            or normalized_text(record.get(args.text_field, "")) == normalized_text(args.text)
        ),
        "sample_size": len(selected),
        "seed": int(args.seed),
        "decoded_count": decoded_count,
        "error_count": len(errors),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "token_units": str(units_path),
        "token_units_sha256": sha256(units_path),
        "vocab_size": int(contract.vocab_size),
        "blank_id": int(contract.blank_id),
        "encoder_frame_shift_ms": float(contract.encoder_frame_shift_ms),
        "providers": stage1.providers,
        "beam_size": int(args.beam_size),
        "token_prune": int(args.token_prune),
        "top_k": int(args.top_k),
        "greedy": {
            "unique_sequence_count": len(greedy_counts),
            "sequence_length_histogram": histogram(len(row["greedy"]["token_ids"]) for row in results),
            "empty_sequence_count": greedy_counts[()],
            "top_sequences": greedy_sequence_counts[:100],
        },
        "beam_best": {
            "unique_sequence_count": len(beam_counts),
            "sequence_length_histogram": histogram(len(row["beam_best"]["token_ids"]) for row in results),
            "empty_sequence_count": beam_counts[()],
            "top_sequences": beam_sequence_counts[:100],
        },
        "greedy_token_counts": [
            {"token_id": token_id, "token": symbols[token_id], "count": count}
            for token_id, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "greedy_emission_frame_count_histogram": histogram(emission_frame_counts),
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_size": len(selected),
                "decoded_count": decoded_count,
                "error_count": len(errors),
                "greedy_unique_sequences": len(greedy_counts),
                "beam_unique_sequences": len(beam_counts),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_error and errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
