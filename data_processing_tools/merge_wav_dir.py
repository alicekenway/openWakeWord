#!/usr/bin/env python3
"""Concatenate WAV files into duration-sized WAV files and write metadata."""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


COPY_CHUNK_FRAMES = 1024 * 1024


@dataclass(frozen=True)
class WavFormat:
    channels: int
    sample_width: int
    sample_rate: int
    compression_type: str
    compression_name: str


@dataclass(frozen=True)
class WavInfo:
    path: Path
    frames: int
    wav_format: WavFormat


def collect_wavs(wav_dir: Path, excluded_dir: Path | None = None) -> list[Path]:
    wavs: list[Path] = []
    for path in wav_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        resolved_path = path.resolve()
        if excluded_dir is not None and resolved_path.is_relative_to(excluded_dir):
            continue
        wavs.append(resolved_path)
    return sorted(wavs)


def inspect_wav(path: Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as reader:
            wav_format = WavFormat(
                channels=reader.getnchannels(),
                sample_width=reader.getsampwidth(),
                sample_rate=reader.getframerate(),
                compression_type=reader.getcomptype(),
                compression_name=reader.getcompname(),
            )
            return WavInfo(path=path, frames=reader.getnframes(), wav_format=wav_format)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Cannot read WAV header from {path}: {exc}") from exc


def validate_formats(wavs: list[WavInfo]) -> WavFormat:
    expected = wavs[0].wav_format
    if expected.compression_type != "NONE":
        raise ValueError(
            f"Compressed WAV input is not supported: {wavs[0].path} "
            f"uses compression type {expected.compression_type!r}"
        )

    for wav in wavs[1:]:
        if wav.wav_format != expected:
            raise ValueError(
                "All input WAV files must have the same channels, sample width, "
                "sample rate, and compression. "
                f"Expected {expected}, but {wav.path} has {wav.wav_format}."
            )
    return expected


def group_wavs(wavs: list[WavInfo], target_frames: int) -> list[list[WavInfo]]:
    groups: list[list[WavInfo]] = []
    current_group: list[WavInfo] = []
    current_frames = 0

    for wav in wavs:
        current_group.append(wav)
        current_frames += wav.frames
        if current_frames >= target_frames:
            groups.append(current_group)
            current_group = []
            current_frames = 0

    if current_group:
        groups.append(current_group)
    return groups


def copy_wav_frames(reader: wave.Wave_read, writer: wave.Wave_write, source_path: Path) -> int:
    frames_copied = 0
    bytes_per_frame = reader.getnchannels() * reader.getsampwidth()
    while True:
        data = reader.readframes(COPY_CHUNK_FRAMES)
        if not data:
            break
        if len(data) % bytes_per_frame != 0:
            raise ValueError(f"WAV data ends with an incomplete sample frame: {source_path}")
        writer.writeframesraw(data)
        frames_copied += len(data) // bytes_per_frame
    return frames_copied


def merge_group(group: list[WavInfo], output_path: Path, wav_format: WavFormat) -> int:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    frames_written = 0
    try:
        with wave.open(str(temporary_path), "wb") as writer:
            writer.setnchannels(wav_format.channels)
            writer.setsampwidth(wav_format.sample_width)
            writer.setframerate(wav_format.sample_rate)
            writer.setcomptype(wav_format.compression_type, wav_format.compression_name)
            for wav in group:
                with wave.open(str(wav.path), "rb") as reader:
                    copied = copy_wav_frames(reader, writer, wav.path)
                if copied != wav.frames:
                    raise ValueError(
                        f"WAV frame count changed while reading {wav.path}: "
                        f"header says {wav.frames}, read {copied}"
                    )
                frames_written += copied
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return frames_written


def metadata_path_value(output_path: Path, output_dir: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(output_path.resolve())
    return output_path.relative_to(output_dir).as_posix()


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    written = 0
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate a WAV directory into files that reach a target duration."
    )
    parser.add_argument("--wav-dir", required=True, help="Input directory containing WAV files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Merged audio is written to output-dir/wav",
    )
    parser.add_argument(
        "--length",
        type=float,
        required=True,
        help="Minimum target duration per merged WAV, in seconds",
    )
    parser.add_argument(
        "--max-output-files",
        type=int,
        default=None,
        help=(
            "Optional upper limit on merged WAV files to write. "
            "For example, --max-output-files 10 writes only the first 10 groups."
        ),
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute WAV paths in metadata.jsonl instead of paths relative to --output-dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace merged WAV and metadata files from an earlier run",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    wav_dir = Path(args.wav_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_wav_dir = output_dir / "wav"
    metadata_path = output_dir / "metadata.jsonl"
    summary_path = output_dir / "merge_summary.json"

    if not wav_dir.is_dir():
        raise ValueError(f"--wav-dir is not a directory: {wav_dir}")
    if not math.isfinite(args.length) or args.length <= 0:
        raise ValueError("--length must be a finite number greater than 0")
    if args.max_output_files is not None and args.max_output_files < 1:
        raise ValueError("--max-output-files must be an integer greater than 0")

    excluded_dir = output_wav_dir if output_wav_dir.is_relative_to(wav_dir) else None
    wav_paths = collect_wavs(wav_dir, excluded_dir)
    if not wav_paths:
        raise ValueError(f"No WAV files found under {wav_dir}")

    wavs = [inspect_wav(path) for path in wav_paths]
    wav_format = validate_formats(wavs)
    target_frames = math.ceil(args.length * wav_format.sample_rate)
    all_groups = group_wavs(wavs, target_frames)
    groups = all_groups[:args.max_output_files] if args.max_output_files is not None else all_groups
    output_paths = [output_wav_dir / f"merged_{index:08d}.wav" for index in range(len(groups))]

    previous_merged_outputs = sorted(output_wav_dir.glob("merged_*.wav"))
    existing_outputs = [
        path
        for path in [metadata_path, summary_path, *previous_merged_outputs]
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        preview = ", ".join(str(path) for path in existing_outputs[:3])
        raise FileExistsError(f"Output files already exist ({preview}); use --overwrite to replace them")

    output_wav_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    output_durations: list[float] = []
    for group, output_path in zip(groups, output_paths):
        frames_written = merge_group(group, output_path, wav_format)
        duration = frames_written / wav_format.sample_rate
        output_durations.append(duration)
        records.append(
            {
                "path": metadata_path_value(output_path, output_dir, args.absolute_paths),
                "text": "",
                "duration": round(duration, 3),
            }
        )

    written_rows = write_jsonl(metadata_path, records)
    expected_output_paths = set(output_paths)
    for stale_path in previous_merged_outputs:
        if stale_path not in expected_output_paths:
            stale_path.unlink()

    total_input_frames = sum(wav.frames for wav in wavs)
    merged_input_frames = sum(wav.frames for group in groups for wav in group)
    summary: dict[str, object] = {
        "input_wav_dir": str(wav_dir),
        "output_dir": str(output_dir),
        "output_wav_dir": str(output_wav_dir),
        "metadata_jsonl": str(metadata_path),
        "target_length_seconds": args.length,
        "sample_rate": wav_format.sample_rate,
        "channels": wav_format.channels,
        "sample_width_bytes": wav_format.sample_width,
        "input_wav_files": len(wavs),
        "input_duration_seconds": round(total_input_frames / wav_format.sample_rate, 3),
        "merged_input_wav_files": sum(len(group) for group in groups),
        "merged_input_duration_seconds": round(merged_input_frames / wav_format.sample_rate, 3),
        "merged_wav_files": len(groups),
        "max_output_files": args.max_output_files,
        "output_limit_reached": len(groups) < len(all_groups),
        "written_rows": written_rows,
        "output_durations_seconds": [round(duration, 3) for duration in output_durations],
        "final_output_below_target": bool(output_durations and output_durations[-1] < args.length),
        "absolute_paths": bool(args.absolute_paths),
        "overwrite": bool(args.overwrite),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
