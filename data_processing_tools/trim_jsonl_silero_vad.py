#!/usr/bin/env python3
"""Trim leading and trailing silence from JSONL audio using Silero VAD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torchaudio


OPENWAKEWORD_ROOT = Path(__file__).resolve().parents[1]
if str(OPENWAKEWORD_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENWAKEWORD_ROOT))

from openwakeword.vad import VAD  # noqa: E402


AUDIO_PATH_KEYS = ["path", "audiofile_path", "audio_file", "audio_path", "file", "filename"]
_WORKER_CONFIG: dict[str, Any] = {}
_WORKER_VAD: VAD | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def detect_audio_path(record: dict[str, Any]) -> tuple[str, str]:
    for key in AUDIO_PATH_KEYS:
        value = record.get(key)
        if value:
            return key, str(value)
    raise ValueError(f"Record has no supported audio path key: {record}")


def resolve_audio_path(audio_path: str, audio_base_dir: Path) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return audio_base_dir / path


def resolve_workers(value: str) -> int:
    if value.lower() == "auto":
        cpu_count = os.cpu_count() or 2
        return max(1, min(cpu_count, 16))
    workers = int(value)
    if workers <= 0:
        raise ValueError("--workers must be 'auto' or a positive integer")
    return workers


def safe_output_name(index: int, source_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem).strip("._") or "audio"
    stem = stem[:80]
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{index:08d}_{stem}_{digest}.wav"


def waveform_to_mono_float(wav: torch.Tensor, source_sample_rate: int, sample_rate: int) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(torch.float32).clamp(-1.0, 1.0)
    if source_sample_rate != sample_rate:
        wav = torchaudio.functional.resample(wav, source_sample_rate, sample_rate)
    return wav.squeeze(0)


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    wav, source_sample_rate = torchaudio.load(str(path))
    return waveform_to_mono_float(wav, source_sample_rate, sample_rate)


def vad_frame_scores(vad: VAD, audio: torch.Tensor, frame_size: int) -> list[float]:
    if audio.numel() == 0:
        return []
    vad.reset_states()
    pcm = (audio.clamp(-1.0, 1.0).numpy() * 32767).astype(np.int16)
    scores: list[float] = []
    for start in range(0, pcm.shape[0], frame_size):
        frame = pcm[start:start + frame_size]
        if frame.shape[0] < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.shape[0]))
        scores.append(float(vad.predict(frame, frame_size=frame_size)))
    return scores


def speech_bounds(
    scores: list[float],
    *,
    threshold: float,
    frame_size: int,
    sample_count: int,
    pre_pad_samples: int,
    post_pad_samples: int,
) -> tuple[int, int] | None:
    speech_frames = [index for index, score in enumerate(scores) if score >= threshold]
    if not speech_frames:
        return None
    start = max(0, speech_frames[0] * frame_size - pre_pad_samples)
    end = min(sample_count, (speech_frames[-1] + 1) * frame_size + post_pad_samples)
    if end <= start:
        return None
    return start, end


def output_path_for_jsonl(output_path: Path, output_jsonl: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(output_path.resolve())
    try:
        return str(output_path.relative_to(output_jsonl.parent))
    except ValueError:
        return str(output_path.resolve())


def update_record_paths(record: dict[str, Any], output_audio_path: str) -> dict[str, Any]:
    updated = dict(record)
    for key in AUDIO_PATH_KEYS:
        updated.pop(key, None)
    updated.pop("source_path", None)
    previous_vad = updated.get("vad_trim")
    if isinstance(previous_vad, dict) and "source_path" in previous_vad:
        updated["vad_trim"] = {key: value for key, value in previous_vad.items() if key != "source_path"}
    updated["path"] = output_audio_path
    return updated


def iter_tasks(records: list[dict[str, Any]], config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for index, record in enumerate(records):
        yield {
            "index": index,
            "record": record,
            "config": config,
        }


def init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _WORKER_VAD
    _WORKER_CONFIG = config
    _WORKER_VAD = VAD(model_path=config["vad_model"], n_threads=int(config["vad_threads"]))


def trim_one(task: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_VAD
    config = task.get("config") or _WORKER_CONFIG
    if _WORKER_VAD is None:
        _WORKER_VAD = VAD(model_path=config["vad_model"], n_threads=int(config["vad_threads"]))

    index = int(task["index"])
    record = dict(task["record"])
    audio_base_dir = Path(config["audio_base_dir"])
    wav_dir = Path(config["wav_dir"])
    output_jsonl = Path(config["output_jsonl"])
    sample_rate = int(config["sample_rate"])
    frame_size = int(config["frame_size"])
    pre_pad_samples = int(config["pre_pad_samples"])
    post_pad_samples = int(config["post_pad_samples"])
    threshold = float(config["threshold"])
    overwrite = bool(config["overwrite"])
    absolute_paths = bool(config["absolute_paths"])
    no_speech_policy = str(config["no_speech_policy"])

    try:
        _, raw_audio_path = detect_audio_path(record)
        source_path = resolve_audio_path(raw_audio_path, audio_base_dir).resolve()
        output_path = wav_dir / safe_output_name(index, source_path)

        audio = load_audio(source_path, sample_rate)
        scores = vad_frame_scores(_WORKER_VAD, audio, frame_size)
        bounds = speech_bounds(
            scores,
            threshold=threshold,
            frame_size=frame_size,
            sample_count=int(audio.numel()),
            pre_pad_samples=pre_pad_samples,
            post_pad_samples=post_pad_samples,
        )
        no_speech = bounds is None
        if no_speech:
            if no_speech_policy == "skip":
                return {
                    "ok": False,
                    "skipped": True,
                    "index": index,
                    "record": record,
                    "source_path": str(source_path),
                    "error": "No speech detected by VAD",
                }
            if no_speech_policy == "error":
                raise RuntimeError("No speech detected by VAD")
            start_sample = 0
            end_sample = int(audio.numel())
            trimmed = audio
        else:
            start_sample, end_sample = bounds
            trimmed = audio[start_sample:end_sample]

        if overwrite or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(output_path),
                trimmed.unsqueeze(0).clamp(-1.0, 1.0),
                sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )

        output_audio_path = output_path_for_jsonl(output_path, output_jsonl, absolute_paths)
        updated = update_record_paths(record, output_audio_path)
        updated["vad_trim"] = {
            "sample_rate": sample_rate,
            "threshold": threshold,
            "frame_ms": config["frame_ms"],
            "pad_ms": config["pad_ms"],
            "pre_pad_ms": config["pre_pad_ms"],
            "post_pad_ms": config["post_pad_ms"],
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_seconds": round(start_sample / sample_rate, 6),
            "end_seconds": round(end_sample / sample_rate, 6),
            "source_duration_seconds": round(float(audio.numel()) / sample_rate, 6),
            "trimmed_duration_seconds": round(float(trimmed.numel()) / sample_rate, 6),
            "speech_frame_count": sum(1 for score in scores if score >= threshold),
            "frame_count": len(scores),
            "no_speech": no_speech,
        }
        return {
            "ok": True,
            "skipped": False,
            "index": index,
            "record": updated,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "no_speech": no_speech,
            "source_duration_seconds": float(audio.numel()) / sample_rate,
            "trimmed_duration_seconds": float(trimmed.numel()) / sample_rate,
        }
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "index": index,
            "record": record,
            "error": repr(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim leading and trailing silence from JSONL audio with Silero VAD."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL with audio path fields")
    parser.add_argument(
        "--audio-base-dir",
        default=".",
        help="Base directory for relative audio paths in the JSONL. Absolute paths ignore this.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL and trimmed WAV directory")
    parser.add_argument("--wav-dir-name", default="wav", help="Trimmed WAV subdirectory name under output-dir")
    parser.add_argument("--output-jsonl", help="Output JSONL path. Default: output-dir/metadata.jsonl")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output WAV sample rate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Silero VAD speech threshold")
    parser.add_argument("--frame-ms", type=float, default=30.0, help="VAD frame size in milliseconds")
    parser.add_argument("--pad-ms", type=float, default=100.0, help="Default audio to keep before/after speech")
    parser.add_argument("--pre-pad-ms", type=float, help="Audio to keep before first speech frame. Defaults to --pad-ms")
    parser.add_argument("--post-pad-ms", type=float, help="Audio to keep after last speech frame. Defaults to --pad-ms")
    parser.add_argument(
        "--vad-model",
        default=str(OPENWAKEWORD_ROOT / "openwakeword" / "resources" / "models" / "silero_vad.onnx"),
        help="Path to silero_vad.onnx",
    )
    parser.add_argument("--vad-threads", type=int, default=1, help="ONNXRuntime threads per worker")
    parser.add_argument(
        "--workers",
        "--max-workers",
        default="auto",
        help="Number of worker processes, or 'auto'. Default: auto",
    )
    parser.add_argument(
        "--no-speech-policy",
        choices=["copy", "skip", "error"],
        default="copy",
        help="What to do when VAD finds no speech. Default: copy full audio.",
    )
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute output paths in JSONL")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV files")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / args.wav_dir_name
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else output_dir / "metadata.jsonl"
    audio_base_dir = Path(args.audio_base_dir).resolve()
    workers = resolve_workers(str(args.workers))
    frame_size = max(1, int(round(args.sample_rate * args.frame_ms / 1000.0)))
    pre_pad_ms = args.pad_ms if args.pre_pad_ms is None else args.pre_pad_ms
    post_pad_ms = args.pad_ms if args.post_pad_ms is None else args.post_pad_ms
    pre_pad_samples = max(0, int(round(args.sample_rate * pre_pad_ms / 1000.0)))
    post_pad_samples = max(0, int(round(args.sample_rate * post_pad_ms / 1000.0)))

    records = read_jsonl(input_jsonl)
    config = {
        "audio_base_dir": str(audio_base_dir),
        "wav_dir": str(wav_dir),
        "output_jsonl": str(output_jsonl),
        "sample_rate": int(args.sample_rate),
        "threshold": float(args.threshold),
        "frame_ms": float(args.frame_ms),
        "frame_size": frame_size,
        "pad_ms": float(args.pad_ms),
        "pre_pad_ms": float(pre_pad_ms),
        "post_pad_ms": float(post_pad_ms),
        "pre_pad_samples": pre_pad_samples,
        "post_pad_samples": post_pad_samples,
        "wav_dir_name": str(args.wav_dir_name),
        "vad_model": str(Path(args.vad_model).resolve()),
        "vad_threads": int(args.vad_threads),
        "overwrite": bool(args.overwrite),
        "absolute_paths": bool(args.absolute_paths),
        "no_speech_policy": str(args.no_speech_policy),
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    written_rows = 0
    no_speech_count = 0
    source_seconds = 0.0
    trimmed_seconds = 0.0

    task_iter = iter_tasks(records, config)
    if workers == 1:
        init_worker(config)
        results = map(trim_one, task_iter)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(config,))
        results = pool.map(trim_one, task_iter, chunksize=16)

    try:
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for result in results:
                if result["ok"]:
                    handle.write(json.dumps(result["record"], ensure_ascii=False, sort_keys=True) + "\n")
                    written_rows += 1
                    no_speech_count += int(bool(result.get("no_speech")))
                    source_seconds += float(result.get("source_duration_seconds") or 0.0)
                    trimmed_seconds += float(result.get("trimmed_duration_seconds") or 0.0)
                elif result.get("skipped"):
                    skipped.append(
                        {
                            "index": result["index"],
                            "source_path": result.get("source_path"),
                            "error": result.get("error"),
                        }
                    )
                else:
                    errors.append(
                        {
                            "index": result["index"],
                            "record": result.get("record"),
                            "error": result.get("error"),
                        }
                    )
    finally:
        if pool is not None:
            pool.shutdown()

    summary = {
        "input_jsonl": str(input_jsonl),
        "audio_base_dir": str(audio_base_dir),
        "output_dir": str(output_dir),
        "wav_dir": str(wav_dir),
        "wav_dir_name": args.wav_dir_name,
        "output_jsonl": str(output_jsonl),
        "sample_rate": args.sample_rate,
        "threshold": args.threshold,
        "frame_ms": args.frame_ms,
        "pad_ms": args.pad_ms,
        "pre_pad_ms": pre_pad_ms,
        "post_pad_ms": post_pad_ms,
        "vad_model": str(Path(args.vad_model).resolve()),
        "vad_threads": args.vad_threads,
        "workers": workers,
        "input_rows": len(records),
        "written_rows": written_rows,
        "skipped_rows": len(skipped),
        "no_speech_rows": no_speech_count,
        "error_count": len(errors),
        "source_duration_seconds": round(source_seconds, 6),
        "trimmed_duration_seconds": round(trimmed_seconds, 6),
        "removed_duration_seconds": round(max(0.0, source_seconds - trimmed_seconds), 6),
        "no_speech_policy": args.no_speech_policy,
        "absolute_paths": args.absolute_paths,
        "overwrite": args.overwrite,
        "errors": errors[:50],
        "skipped": skipped[:50],
    }
    write_json(output_jsonl.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
