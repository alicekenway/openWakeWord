#!/usr/bin/env python3
"""Reusable training pipeline helpers for the local openWakeWord checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torchaudio
from numpy.lib.format import open_memmap
from tqdm import tqdm


os.environ.setdefault("MPLCONFIGDIR", "/tmp/wuw_mpl_config")

OPENWAKEWORD_ROOT = Path(__file__).resolve().parents[2]
if str(OPENWAKEWORD_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENWAKEWORD_ROOT))

# acoustics imports scipy.special.sph_harm at import time. Newer SciPy builds
# expose sph_harm_y instead, and this pipeline only needs acoustics.generator.
try:
    import scipy.special as scipy_special

    if not hasattr(scipy_special, "sph_harm") and hasattr(scipy_special, "sph_harm_y"):
        def _sph_harm_compat(m: int, n: int, theta: Any, phi: Any) -> Any:
            return scipy_special.sph_harm_y(n, m, phi, theta)

        scipy_special.sph_harm = _sph_harm_compat  # type: ignore[attr-defined]
except Exception:
    pass

import openwakeword  # noqa: E402
from openwakeword.train import Model as TrainModel  # noqa: E402
from openwakeword.data import mmap_batch_generator  # noqa: E402
from openwakeword.utils import AudioFeatures, download_models  # noqa: E402


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
DEFAULT_SR = 16000
DEFAULT_IO_WORKERS = max(1, min(8, os.cpu_count() or 2))
DEFAULT_IO_CHUNKSIZE = 64
VALID_PLACEMENTS = {"start", "end", "center", "random"}
AUDIO_PATH_KEYS = ("path", "audiofile_path", "audio_file", "audio_path", "file", "filename")
EVALUATION_DEFAULTS: dict[str, Any] = {
    "threshold": 0.5,
    "debounce_seconds": 1.0,
    "positive_padding": 1,
    "negative_padding": 0,
    "limit_positive": None,
    "limit_negative_seconds": None,
    "sample_rate": DEFAULT_SR,
    "chunk_size": 1280,
    "model_window_seconds": 2.0,
    "record_window_scores": True,
}
EVALUATION_CONFIG_KEYS = {
    "model",
    "model_dir",
    "positive_manifest",
    "negative_manifest",
    "negative_speech_manifest",
    "background_manifest",
    "output_json",
    "details_jsonl",
    "abnormal_jsonl",
    "output_config_json",
    *EVALUATION_DEFAULTS.keys(),
}
PREPARE_MANIFEST_DEFAULTS: dict[str, Any] = {
    "output_dir": None,
    "positive_jsonl": None,
    "positive_audio_base_path": None,
    "negative_jsonl": None,
    "negative_audio_base_path": None,
    "background_jsonl": None,
    "background_audio_base_path": None,
    "positive_train_count": None,
    "positive_dev_count": 300,
    "positive_test_count": 300,
    "negative_train_count": 200_000,
    "negative_dev_count": 300,
    "negative_test_seconds": 3600,
    "background_train_count": 5000,
    "background_dev_count": 300,
    "background_test_seconds": 3600,
    "clip_seconds": 2.0,
    "sample_rate": DEFAULT_SR,
    "seed": 1337,
}
RUN_EXPERIMENT_DEFAULTS: dict[str, Any] = {
    **PREPARE_MANIFEST_DEFAULTS,
    "experiment_dir": "/home/alicekenway/Dev/project/WUW/training/expts1",
    "model_name": "turn_on_the_office_lights",
    "snr_low": -5.0,
    "snr_high": 15.0,
    "artificial_prob": 0.15,
    "augmentation_rounds": 1,
    "batch_size": 64,
    "audio_loader_workers": 1,
    "prefetch_batches": 1,
    "batch_positive": 64,
    "batch_negative": 256,
    "convert_workers": DEFAULT_IO_WORKERS,
    "augment_workers": DEFAULT_IO_WORKERS,
    "steps": 2000,
    "layer_size": 64,
    "model_type": "dnn",
    "max_negative_weight": 500.0,
    "target_false_positives_per_hour": 0.5,
    "evaluation_config": None,
    "threshold": None,
    "debounce_seconds": None,
    "eval_chunk_size": None,
    "model_window_seconds": None,
    "record_window_scores": None,
    "ncpu": max(1, (os.cpu_count() or 2) // 2),
    "device": "auto",
    "require_cuda": False,
    "skip_download": False,
    "overwrite": False,
    "quick": False,
}
_CONVERT_WORKER_CONFIG: dict[str, Any] = {}
_AUGMENT_WORKER_CONFIG: dict[str, Any] = {}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")
            count += 1
    return count


def existing_complete_summary(summary_path: Path, expected_count: int, count_key: str) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = read_json(summary_path)
    except Exception:
        return False
    return (
        int(summary.get(count_key, -1)) == int(expected_count)
        and int(summary.get("error_count", 0)) == 0
    )


def feature_file_complete(output_file: Path, expected_count: int, expected_summary: dict[str, Any] | None = None) -> bool:
    summary_path = output_file.with_suffix(".summary.json")
    if not output_file.exists() or not summary_path.exists():
        return False
    try:
        summary = read_json(summary_path)
        if int(summary.get("feature_count", -1)) != int(expected_count):
            return False
        if int(summary.get("error_count", 0)) != 0:
            return False
        for key, expected_value in (expected_summary or {}).items():
            if summary.get(key) != expected_value:
                return False
        return int(np.load(output_file, mmap_mode="r").shape[0]) == int(expected_count)
    except Exception:
        return False


def augmented_manifest_complete(
    output_manifest: Path,
    expected_count: int,
    expected_summary: dict[str, Any] | None = None,
) -> bool:
    summary_path = output_manifest.with_suffix(".summary.json")
    if not output_manifest.exists() or not summary_path.exists():
        return False
    try:
        summary = read_json(summary_path)
        if int(summary.get("output_count", -1)) != int(expected_count):
            return False
        if int(summary.get("error_count", 0)) != 0:
            return False
        for key, expected_value in (expected_summary or {}).items():
            if summary.get(key) != expected_value:
                return False
        return True
    except Exception:
        return False


def collect_audio_files(root: Path, extensions: set[str] = AUDIO_EXTENSIONS) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions])


def stable_id(path: str, index: int = 0) -> str:
    digest = hashlib.sha1(f"{index}:{path}".encode("utf-8")).hexdigest()
    return digest[:16]


def load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return config


def merge_config(args: argparse.Namespace, defaults: dict[str, Any]) -> argparse.Namespace:
    config = dict(defaults)
    config.update(load_config_file(getattr(args, "config", None)))
    for key, value in vars(args).items():
        if key in {"config", "func"} or callable(value):
            continue
        if value is not None:
            config[key] = value
    return argparse.Namespace(**config)


def parse_dataset_specs(value: Any, audio_base_path: str | None = None) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return parse_dataset_specs(json.loads(stripped), audio_base_path=audio_base_path)
        spec: dict[str, Any] = {"jsonl_path": value}
        if audio_base_path:
            spec["audio_base_path"] = audio_base_path
        return [spec]
    if isinstance(value, dict):
        if "jsonl_path" not in value:
            raise ValueError(f"Dataset spec is missing jsonl_path: {value}")
        spec = dict(value)
        if audio_base_path and not spec.get("audio_base_path"):
            spec["audio_base_path"] = audio_base_path
        return [spec]
    if isinstance(value, list):
        specs: list[dict[str, Any]] = []
        for item in value:
            specs.extend(parse_dataset_specs(item, audio_base_path=audio_base_path))
        return specs
    raise TypeError(f"Unsupported dataset spec type: {type(value).__name__}")


def replace_audio_path(record: dict[str, Any], value: str | Path) -> dict[str, Any]:
    """Replace legacy/current audio references with one canonical ``path``."""
    updated = dict(record)
    for key in AUDIO_PATH_KEYS:
        updated.pop(key, None)
    updated.pop("source_path", None)
    vad_trim = updated.get("vad_trim")
    if isinstance(vad_trim, dict) and "source_path" in vad_trim:
        updated["vad_trim"] = {key: item for key, item in vad_trim.items() if key != "source_path"}
    updated["path"] = str(value)
    return updated


def record_source_path(record: dict[str, Any], jsonl_path: Path, audio_base_path: Path | None) -> Path:
    for key in AUDIO_PATH_KEYS:
        value = record.get(key)
        if value:
            path = Path(str(value))
            if not path.is_absolute():
                path = (audio_base_path or jsonl_path.parent) / path
            return Path(os.path.abspath(os.fspath(path)))
    raise ValueError(f"JSONL record has no audio path field: {record}")


def load_dataset_records(
    value: Any,
    label: int,
    source: str,
    audio_base_path: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    specs = parse_dataset_specs(value, audio_base_path=audio_base_path)
    for dataset_ndx, spec in enumerate(specs):
        jsonl_path = Path(spec["jsonl_path"]).expanduser().resolve()
        base_value = spec.get("audio_base_path")
        base_path = Path(base_value).expanduser().resolve() if base_value else None
        dataset_source = spec.get("source") or source
        raw_records = read_jsonl(jsonl_path)
        sources.append(
            {
                "jsonl_path": str(jsonl_path),
                "audio_base_path": str(base_path) if base_path else None,
                "source": dataset_source,
                "records": len(raw_records),
            }
        )
        for ndx, raw_record in enumerate(raw_records):
            path = record_source_path(raw_record, jsonl_path, base_path)
            record = replace_audio_path(raw_record, path)
            record["label"] = label
            record["source"] = record.get("source") or dataset_source
            record["id"] = record.get("id") or stable_id(str(path), ndx)
            record["input_jsonl"] = str(jsonl_path)
            record["dataset_index"] = dataset_ndx
            records.append(record)
    return records, sources


def split_records(
    records: list[dict[str, Any]],
    *,
    train_count: int | None,
    dev_count: int,
    test_count: int,
    seed: int,
    test_split_name: str = "test",
) -> dict[str, list[dict[str, Any]]]:
    shuffled = [dict(record) for record in records]
    random.Random(seed).shuffle(shuffled)
    test_count = min(test_count, len(shuffled))
    dev_count = min(dev_count, max(0, len(shuffled) - test_count))
    remaining = shuffled[test_count + dev_count:]
    if train_count is None:
        train_count = len(remaining)
    train_count = min(train_count, len(remaining))
    splits = {
        "train": remaining[:train_count],
        "dev": shuffled[test_count:test_count + dev_count],
        test_split_name: shuffled[:test_count],
    }
    for split_name, split_records_ in splits.items():
        for ndx, record in enumerate(split_records_):
            record["split"] = split_name
            record["id"] = stable_id(record["path"], ndx)
    return splits


def write_split_manifests(manifest_dir: Path, prefix: str, splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name, records in splits.items():
        output_name = f"{prefix}_{split_name}.jsonl"
        counts[f"{prefix}_{split_name}"] = write_jsonl(manifest_dir / output_name, records)
    return counts


def require_dataset(value: Any, name: str) -> None:
    if not parse_dataset_specs(value):
        raise ValueError(
            f"Provide {name}_jsonl as a JSONL path, a dataset object, or a list of dataset objects."
        )


def waveform_to_float(wav: torch.Tensor, sample_rate: int, sr: int = DEFAULT_SR) -> torch.Tensor:
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if sample_rate != sr:
        wav = torchaudio.functional.resample(wav, sample_rate, sr)
    return wav.squeeze(0).to(torch.float32).clamp(-1.0, 1.0)


def load_audio_float(path: Path, sr: int = DEFAULT_SR) -> torch.Tensor:
    wav, sample_rate = torchaudio.load(str(path))
    return waveform_to_float(wav, sample_rate, sr=sr)


def fixed_length(
    audio: torch.Tensor,
    target_samples: int,
    rng: random.Random,
    placement: str,
    end_jitter_seconds: float = 0.2,
    sr: int = DEFAULT_SR,
) -> torch.Tensor:
    if audio.numel() == target_samples:
        return audio
    if audio.numel() > target_samples:
        if placement == "end":
            start = audio.numel() - target_samples
        elif placement == "center":
            start = max(0, (audio.numel() - target_samples) // 2)
        else:
            start = rng.randint(0, audio.numel() - target_samples)
        return audio[start:start + target_samples]

    out = torch.zeros(target_samples, dtype=torch.float32)
    if placement == "start":
        start = 0
    elif placement == "center":
        start = (target_samples - audio.numel()) // 2
    elif placement == "random":
        start = rng.randint(0, target_samples - audio.numel())
    else:
        max_jitter = min(target_samples - audio.numel(), int(end_jitter_seconds * sr))
        jitter = rng.randint(0, max_jitter) if max_jitter > 0 else 0
        start = target_samples - audio.numel() - jitter
    out[start:start + audio.numel()] = audio
    return out


def load_fixed_length_audio(
    path: Path,
    target_samples: int,
    rng: random.Random,
    placement: str,
    sr: int = DEFAULT_SR,
) -> torch.Tensor:
    try:
        info = torchaudio.info(str(path))
        source_sr = info.sample_rate or sr
        target_source_frames = max(1, math.ceil(target_samples * source_sr / sr))
        if info.num_frames > target_source_frames:
            if placement == "start":
                frame_offset = 0
            elif placement == "end":
                frame_offset = info.num_frames - target_source_frames
            elif placement == "center":
                frame_offset = max(0, (info.num_frames - target_source_frames) // 2)
            else:
                frame_offset = rng.randint(0, info.num_frames - target_source_frames)
            wav, sample_rate = torchaudio.load(
                str(path),
                frame_offset=frame_offset,
                num_frames=target_source_frames,
            )
            audio = waveform_to_float(wav, sample_rate, sr=sr)
            return fixed_length(
                audio,
                target_samples,
                rng,
                "end" if placement == "end" else "start",
                sr=sr,
            )
    except Exception:
        pass

    return fixed_length(load_audio_float(path, sr=sr), target_samples, rng, placement, sr=sr)


def load_feature_audio_item(item: tuple[int, Path, int, str, int, int]) -> dict[str, Any]:
    index, path, target_samples, placement, sample_rate, seed = item
    try:
        audio = load_fixed_length_audio(
            path,
            target_samples,
            random.Random(seed + index),
            placement,
            sr=sample_rate,
        )
        return {
            "index": index,
            "audio": (audio.numpy() * 32767).astype(np.int16),
            "error": None,
        }
    except Exception as exc:
        return {
            "index": index,
            "audio": None,
            "error": {"path": str(path), "error": repr(exc)},
        }


def save_wav(path: Path, audio: torch.Tensor, sr: int = DEFAULT_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(path),
        audio.unsqueeze(0).clamp(-1.0, 1.0),
        sr,
        encoding="PCM_S",
        bits_per_sample=16,
    )


def rms(audio: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(audio.to(torch.float32) ** 2) + 1e-10)


def mix_with_noise(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_rms = rms(signal)
    noise_rms = rms(noise)
    scale = signal_rms / (noise_rms * (10 ** (snr_db / 20.0)))
    mixed = signal + noise * scale
    peak = mixed.abs().max().item()
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed


def add_artificial_noise(signal: torch.Tensor, rng: random.Random, snr_low: float, snr_high: float) -> torch.Tensor:
    noise = torch.randn_like(signal)
    snr = rng.uniform(snr_low, snr_high)
    return mix_with_noise(signal, noise, snr)


def manifest_paths(records: list[dict[str, Any]]) -> list[Path]:
    return [Path(record["path"]) for record in records]


def count_duration_seconds(paths: Iterable[Path], sr: int = DEFAULT_SR) -> float:
    total = 0.0
    for path in paths:
        try:
            info = torchaudio.info(str(path))
            total += info.num_frames / info.sample_rate
        except Exception:
            audio = load_audio_float(path, sr=sr)
            total += audio.numel() / sr
    return total


def _init_convert_worker(config: dict[str, Any]) -> None:
    global _CONVERT_WORKER_CONFIG
    _CONVERT_WORKER_CONFIG = config


def _convert_manifest_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    ndx, record = item
    cfg = _CONVERT_WORKER_CONFIG
    src = Path(record["path"])
    out_name = f"{ndx:08d}_{stable_id(str(src), ndx)}.wav"
    out_path = Path(cfg["output_dir"]) / out_name
    rng = random.Random(cfg["seed"] + ndx)
    placement = validate_placement(record.get("placement", cfg["placement"]))
    try:
        if not out_path.exists() or cfg["overwrite"]:
            audio = load_fixed_length_audio(
                src,
                cfg["target_samples"],
                rng,
                placement,
                sr=cfg["sample_rate"],
            )
            save_wav(out_path, audio, sr=cfg["sample_rate"])
        new_record = replace_audio_path(record, out_path.resolve())
        new_record.update({"converted": True, "placement": placement})
        return {"record": new_record, "error": None}
    except Exception as exc:
        return {"record": None, "error": {"path": str(src), "error": repr(exc)}}


def _init_augment_worker(config: dict[str, Any]) -> None:
    global _AUGMENT_WORKER_CONFIG
    _AUGMENT_WORKER_CONFIG = config


def _augment_audio_worker(item: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
    round_ndx, ndx, record = item
    cfg = _AUGMENT_WORKER_CONFIG
    rng = random.Random(cfg["seed"] + round_ndx * 1_000_003 + ndx)
    src = Path(record["path"])
    out_name = f"{round_ndx:02d}_{ndx:08d}_{stable_id(str(src), ndx)}.wav"
    out_path = Path(cfg["output_dir"]) / out_name
    placement = validate_placement(record.get("placement", cfg["placement"]))
    try:
        if not out_path.exists() or cfg["overwrite"]:
            audio = load_fixed_length_audio(
                src,
                cfg["target_samples"],
                rng,
                placement,
                sr=cfg["sample_rate"],
            )
            noise_src = Path(rng.choice(cfg["noise_paths"]))
            noise = load_fixed_length_audio(
                noise_src,
                cfg["target_samples"],
                rng,
                "random",
                sr=cfg["sample_rate"],
            )
            snr = rng.uniform(cfg["snr_low"], cfg["snr_high"])
            mixed = mix_with_noise(audio, noise, snr)
            if rng.random() < cfg["artificial_prob"]:
                mixed = add_artificial_noise(mixed, rng, cfg["snr_low"], cfg["snr_high"])
            if cfg["random_gain_db"]:
                gain_db = rng.uniform(-abs(cfg["random_gain_db"]), abs(cfg["random_gain_db"]))
                mixed = mixed * (10 ** (gain_db / 20.0))
            save_wav(out_path, mixed, sr=cfg["sample_rate"])
        new_record = replace_audio_path(record, out_path.resolve())
        new_record.update(
            {
                "augmented": True,
                "augmentation_round": round_ndx,
                "placement": placement,
            }
        )
        return {"record": new_record, "error": None}
    except Exception as exc:
        return {"record": None, "error": {"path": str(src), "error": repr(exc)}}


def command_download_models(args: argparse.Namespace) -> None:
    target = Path(args.output_dir).resolve()
    models = [] if args.models == ["all"] else args.models
    download_models(model_names=models, target_directory=str(target))
    files = sorted([p.name for p in target.glob("*") if p.is_file()])
    write_json(target / "download_manifest.json", {"target_directory": str(target), "requested_models": args.models, "files": files})
    print(f"Downloaded/verified {len(files)} model files in {target}")


def command_index_audio(args: argparse.Namespace) -> None:
    audio_dir = Path(args.audio_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    paths = collect_audio_files(audio_dir)
    if args.limit:
        paths = paths[:args.limit]
    records = []
    for ndx, path in enumerate(paths):
        audio_path = path.relative_to(audio_dir) if args.relative_paths else path
        records.append(
            {
                "id": stable_id(str(path), ndx),
                "path": str(audio_path),
                "source": args.source,
                "label": args.label,
            }
        )
    count = write_jsonl(output_jsonl, records)
    write_json(
        output_jsonl.with_suffix(".summary.json"),
        {
            "audio_dir": str(audio_dir),
            "output_jsonl": str(output_jsonl),
            "count": count,
            "source": args.source,
            "label": args.label,
            "relative_paths": args.relative_paths,
        },
    )
    print(f"Indexed {count} files from {audio_dir} into {output_jsonl}")


def command_prepare_manifests(args: argparse.Namespace) -> None:
    args = merge_config(args, PREPARE_MANIFEST_DEFAULTS)
    require_dataset(args.positive_jsonl, "positive")
    require_dataset(args.negative_jsonl, "negative")
    require_dataset(args.background_jsonl, "background")

    output_value = args.output_dir or getattr(args, "experiment_dir", None)
    if not output_value:
        raise ValueError("Provide --output-dir or experiment_dir in the config.")
    output_dir = Path(output_value).resolve()
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    positive_records, positive_sources = load_dataset_records(
        args.positive_jsonl,
        label=1,
        source="positive",
        audio_base_path=args.positive_audio_base_path,
    )
    negative_records, negative_sources = load_dataset_records(
        args.negative_jsonl,
        label=0,
        source="negative",
        audio_base_path=args.negative_audio_base_path,
    )
    background_records, background_sources = load_dataset_records(
        args.background_jsonl,
        label=0,
        source="background",
        audio_base_path=args.background_audio_base_path,
    )

    positive_splits = split_records(
        positive_records,
        train_count=args.positive_train_count,
        dev_count=int(args.positive_dev_count),
        test_count=int(args.positive_test_count),
        seed=int(args.seed),
    )
    negative_test_count = math.ceil(float(args.negative_test_seconds) / float(args.clip_seconds))
    negative_splits = split_records(
        negative_records,
        train_count=int(args.negative_train_count),
        dev_count=int(args.negative_dev_count),
        test_count=negative_test_count,
        seed=int(args.seed) + 1,
    )
    background_test_count = math.ceil(args.background_test_seconds / args.clip_seconds)
    background_splits = split_records(
        background_records,
        train_count=int(args.background_train_count),
        dev_count=int(args.background_dev_count),
        test_count=background_test_count,
        seed=int(args.seed) + 2,
    )

    positive_counts = write_split_manifests(manifest_dir, "positive", positive_splits)
    negative_counts = write_split_manifests(manifest_dir, "negative", negative_splits)
    background_counts = write_split_manifests(manifest_dir, "background", background_splits)

    summary = {
        "positive": positive_counts,
        "negative": negative_counts,
        "background": background_counts,
        "clip_seconds": args.clip_seconds,
        "seed": args.seed,
        "sources": {
            "positive": positive_sources,
            "negative": negative_sources,
            "background": background_sources,
        },
    }
    write_json(manifest_dir / "manifest_summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default))


def command_convert_manifest(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_manifest = Path(args.output_manifest).resolve() if args.output_manifest else output_dir / f"{manifest.stem}_converted.jsonl"
    records = read_jsonl(manifest)
    if args.limit:
        records = records[:args.limit]
    target_samples = int(args.clip_seconds * args.sample_rate)

    converted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    worker_config = {
        "output_dir": str(output_dir),
        "target_samples": target_samples,
        "clip_seconds": args.clip_seconds,
        "sample_rate": args.sample_rate,
        "placement": args.placement,
        "seed": args.seed,
        "overwrite": args.overwrite,
    }
    workers = max(1, args.workers)
    tasks = list(enumerate(records))
    if workers == 1:
        _init_convert_worker(worker_config)
        results = map(_convert_manifest_worker, tasks)
        iterator = tqdm(results, total=len(tasks), desc=f"Converting {manifest.name}")
        for result in iterator:
            if result["record"]:
                converted.append(result["record"])
            else:
                errors.append(result["error"])
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_convert_worker, initargs=(worker_config,)) as pool:
            results = pool.map(_convert_manifest_worker, tasks, chunksize=DEFAULT_IO_CHUNKSIZE)
            iterator = tqdm(results, total=len(tasks), desc=f"Converting {manifest.name}")
            for result in iterator:
                if result["record"]:
                    converted.append(result["record"])
                else:
                    errors.append(result["error"])

    count = write_jsonl(output_manifest, converted)
    write_json(
        output_manifest.with_suffix(".summary.json"),
        {
            "input_manifest": str(manifest),
            "output_manifest": str(output_manifest),
            "output_dir": str(output_dir),
            "count": count,
            "errors": errors[:50],
            "error_count": len(errors),
            "clip_seconds": args.clip_seconds,
            "placement": args.placement,
            "workers": workers,
        },
    )
    print(f"Converted {count} clips to {output_dir}; errors={len(errors)}")


def command_augment_audio(args: argparse.Namespace) -> None:
    input_records: list[dict[str, Any]] = []
    if args.input_manifest:
        input_records.extend(read_jsonl(Path(args.input_manifest)))
    if args.input_dir:
        input_records.extend({"path": str(path.resolve())} for path in collect_audio_files(Path(args.input_dir), {".wav"}))
    if not input_records:
        raise ValueError("Provide --input-manifest or --input-dir")

    default_placement = validate_placement(args.placement)
    for record in input_records:
        record["placement"] = validate_placement(record.get("placement", default_placement))
    input_items = [(Path(record["path"]), record["placement"]) for record in input_records]
    input_signature = feature_input_signature(input_items)
    input_placement_counts = placement_counts(record["placement"] for record in input_records)

    output_dir = Path(args.output_dir).resolve()
    output_manifest = Path(args.output_manifest).resolve() if args.output_manifest else output_dir / "augmented.jsonl"
    total = args.rounds * len(input_records)
    expected_summary = {
        "input_signature": input_signature,
        "placement_counts": input_placement_counts,
    }
    if not getattr(args, "overwrite", False) and augmented_manifest_complete(output_manifest, total, expected_summary):
        print(f"Skipping augmentation; complete output already exists: {output_manifest}")
        return
    force_output_overwrite = False
    summary_path = output_manifest.with_suffix(".summary.json")
    if summary_path.exists():
        try:
            existing_summary = read_json(summary_path)
            force_output_overwrite = any(existing_summary.get(key) != value for key, value in expected_summary.items())
        except Exception:
            force_output_overwrite = False

    noise_paths: list[Path] = []
    for noise_manifest in getattr(args, "noise_manifest", None) or []:
        noise_paths.extend(manifest_paths(read_jsonl(Path(noise_manifest))))
    for noise_dir in getattr(args, "noise_dir", None) or []:
        noise_paths.extend(collect_audio_files(Path(noise_dir)))
    if not noise_paths:
        raise ValueError("No noise files found from --noise-manifest or --noise-dir")

    target_samples = int(args.clip_seconds * args.sample_rate)
    augmented: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    worker_config = {
        "output_dir": str(output_dir),
        "noise_paths": [str(path) for path in noise_paths],
        "target_samples": target_samples,
        "snr_low": args.snr_low,
        "snr_high": args.snr_high,
        "artificial_prob": args.artificial_prob,
        "random_gain_db": args.random_gain_db,
        "clip_seconds": args.clip_seconds,
        "sample_rate": args.sample_rate,
        "placement": default_placement,
        "seed": args.seed,
        "overwrite": args.overwrite or force_output_overwrite,
    }
    workers = max(1, args.workers)
    tasks = (
        (round_ndx, ndx, record)
        for round_ndx in range(args.rounds)
        for ndx, record in enumerate(input_records)
    )
    if workers == 1:
        _init_augment_worker(worker_config)
        results = map(_augment_audio_worker, tasks)
        iterator = tqdm(results, total=total, desc=f"Augment {args.rounds} round(s)")
        for result in iterator:
            if result["record"]:
                augmented.append(result["record"])
            else:
                errors.append(result["error"])
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_augment_worker, initargs=(worker_config,)) as pool:
            results = pool.map(_augment_audio_worker, tasks, chunksize=DEFAULT_IO_CHUNKSIZE)
            iterator = tqdm(results, total=total, desc=f"Augment {args.rounds} round(s)")
            for result in iterator:
                if result["record"]:
                    augmented.append(result["record"])
                else:
                    errors.append(result["error"])

    count = write_jsonl(output_manifest, augmented)
    write_json(
        output_manifest.with_suffix(".summary.json"),
        {
            "input_count": len(input_records),
            "output_count": count,
            "rounds": args.rounds,
            "noise_count": len(noise_paths),
            "snr_low": args.snr_low,
            "snr_high": args.snr_high,
            "artificial_prob": args.artificial_prob,
            "placement": default_placement,
            "placement_counts": input_placement_counts,
            "input_signature": input_signature,
            "workers": workers,
            "errors": errors[:50],
            "error_count": len(errors),
        },
    )
    print(f"Augmented {count} clips into {output_dir}; errors={len(errors)}")


def paths_from_feature_inputs(manifests: list[str], dirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for manifest in manifests:
        paths.extend(manifest_paths(read_jsonl(Path(manifest))))
    for directory in dirs:
        paths.extend(collect_audio_files(Path(directory), {".wav"}))
    return paths


def feature_items_from_feature_inputs(
    manifests: list[str],
    dirs: list[str],
    default_placement: str,
) -> list[tuple[Path, str]]:
    default_placement = validate_placement(default_placement)
    items: list[tuple[Path, str]] = []
    for manifest in manifests:
        manifest_path = Path(manifest).expanduser().resolve()
        for record in read_jsonl(manifest_path):
            path = record_source_path(record, manifest_path, None)
            placement = validate_placement(record.get("placement", default_placement))
            items.append((path, placement))
    for directory in dirs:
        items.extend((path, default_placement) for path in collect_audio_files(Path(directory), {".wav"}))
    return items


def feature_model_paths(model_dir: str | None) -> dict[str, str]:
    if not model_dir:
        return {}
    root = Path(model_dir)
    paths = {
        "melspec_model_path": str(root / "melspectrogram.onnx"),
        "embedding_model_path": str(root / "embedding_model.onnx"),
    }
    missing = [path for path in paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature model files: {missing}")
    return paths


def resolve_feature_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"

    torch_cuda = torch.cuda.is_available()
    try:
        import onnxruntime as ort

        ort_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as exc:
        if requested_device == "gpu":
            raise RuntimeError(f"Could not import ONNX Runtime while checking GPU support: {exc!r}") from exc
        ort_cuda = False

    if requested_device == "gpu":
        if not torch_cuda:
            raise RuntimeError("GPU was requested, but torch.cuda.is_available() is False in this process.")
        if not ort_cuda:
            raise RuntimeError("GPU was requested, but ONNX Runtime does not expose CUDAExecutionProvider.")
        return "gpu"

    return "gpu" if torch_cuda and ort_cuda else "cpu"


def command_generate_features(args: argparse.Namespace) -> None:
    default_placement = validate_placement(args.placement)
    feature_items = feature_items_from_feature_inputs(args.audio_manifest, args.audio_dir, default_placement)
    if args.limit:
        feature_items = feature_items[:args.limit]
    if not feature_items:
        raise ValueError("No audio files provided")
    input_signature = feature_input_signature(feature_items)
    input_placement_counts = placement_counts(placement for _, placement in feature_items)
    expected_summary = {
        "input_signature": input_signature,
        "placement_counts": input_placement_counts,
    }

    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_file.with_suffix(".summary.json")
    progress_path = output_file.with_suffix(".progress.json")
    if not getattr(args, "overwrite", False) and feature_file_complete(output_file, len(feature_items), expected_summary):
        print(f"Skipping feature extraction; complete output already exists: {output_file}")
        return

    target_samples = int(args.clip_seconds * args.sample_rate)
    device = resolve_feature_device(args.device)
    model_kwargs = feature_model_paths(args.model_dir)
    features = AudioFeatures(device=device, ncpu=args.ncpu, **model_kwargs)
    feature_shape = features.get_embedding_shape(args.clip_seconds, sr=args.sample_rate)
    audio_loader_workers = max(1, int(getattr(args, "audio_loader_workers", 1) or 1))
    prefetch_batches = max(1, int(getattr(args, "prefetch_batches", 1) or 1))

    start_from = 0
    row = 0
    errors: list[dict[str, str]] = []
    error_count = 0
    resume = False
    if not getattr(args, "overwrite", False) and output_file.exists() and progress_path.exists():
        try:
            progress = read_json(progress_path)
            existing = np.load(output_file, mmap_mode="r")
            resume = (
                int(progress.get("input_count", -1)) == len(feature_items)
                and progress.get("input_signature") == input_signature
                and progress.get("placement_counts") == input_placement_counts
                and list(progress.get("feature_shape", [])) == list(feature_shape)
                and tuple(existing.shape) == (len(feature_items), feature_shape[0], feature_shape[1])
            )
            if resume:
                start_from = int(progress.get("next_index", 0))
                row = int(progress.get("row_count", 0))
                errors = list(progress.get("errors", []))[:50]
                error_count = int(progress.get("error_count", len(errors)))
        except Exception:
            resume = False

    if resume:
        mmap = open_memmap(output_file, mode="r+")
        print(f"Resuming feature extraction at input index {start_from}, output row {row}: {output_file}")
    else:
        if output_file.exists() and not summary_path.exists() and not getattr(args, "overwrite", False):
            print(f"Existing feature file has no complete summary/progress; recomputing: {output_file}")
        mmap = open_memmap(output_file, mode="w+", dtype=np.float32, shape=(len(feature_items), feature_shape[0], feature_shape[1]))

    def write_progress(next_index: int) -> None:
        write_json(
            progress_path,
            {
                "output_file": str(output_file),
                "input_count": len(feature_items),
                "input_signature": input_signature,
                "next_index": next_index,
                "row_count": row,
                "feature_shape": list(feature_shape),
                "clip_seconds": args.clip_seconds,
                "requested_device": args.device,
                "device": device,
                "ncpu": args.ncpu,
                "batch_size": args.batch_size,
                "audio_loader_workers": audio_loader_workers,
                "prefetch_batches": prefetch_batches,
                "placement": default_placement,
                "placement_counts": input_placement_counts,
                "seed": args.seed,
                "sample_rate": args.sample_rate,
                "errors": errors[:50],
                "error_count": error_count,
            },
        )

    def feature_load_items(start: int, batch_items: list[tuple[Path, str]]) -> list[tuple[int, Path, int, str, int, int]]:
        return [
            (start + offset, path, target_samples, placement, args.sample_rate, args.seed)
            for offset, (path, placement) in enumerate(batch_items)
        ]

    def process_loaded_batch(start: int, batch_count: int, loaded: list[dict[str, Any]]) -> None:
        nonlocal row, error_count, errors
        batch_audio: list[np.ndarray] = []
        for result in loaded:
            if result["error"]:
                error_count += 1
                if len(errors) < 50:
                    errors.append(result["error"])
            else:
                batch_audio.append(result["audio"])

        if not batch_audio:
            write_progress(start + batch_count)
            return

        batch = np.stack(batch_audio, axis=0)
        batch_features = features.embed_clips(batch, batch_size=batch.shape[0], ncpu=args.ncpu)
        mmap[row:row + batch_features.shape[0]] = batch_features.astype(np.float32)
        row += batch_features.shape[0]
        mmap.flush()
        write_progress(start + batch_count)

    batch_starts = list(range(start_from, len(feature_items), args.batch_size))
    progress_bar = tqdm(batch_starts, desc=f"Features {output_file.name}")
    if audio_loader_workers == 1 and prefetch_batches == 1:
        for start in progress_bar:
            batch_items = feature_items[start:start + args.batch_size]
            loaded = [load_feature_audio_item(item) for item in feature_load_items(start, batch_items)]
            process_loaded_batch(start, len(batch_items), loaded)
    else:
        with ThreadPoolExecutor(max_workers=audio_loader_workers) as loader_pool:
            starts_iter = iter(batch_starts)
            pending: list[tuple[int, int, list[Any]]] = []

            def submit_batch(batch_start: int) -> tuple[int, int, list[Any]]:
                batch_items = feature_items[batch_start:batch_start + args.batch_size]
                futures = [loader_pool.submit(load_feature_audio_item, item) for item in feature_load_items(batch_start, batch_items)]
                return batch_start, len(batch_items), futures

            for _ in range(prefetch_batches):
                try:
                    pending.append(submit_batch(next(starts_iter)))
                except StopIteration:
                    break

            while pending:
                start, batch_count, futures = pending.pop(0)
                loaded = [future.result() for future in futures]
                while len(pending) < prefetch_batches:
                    try:
                        pending.append(submit_batch(next(starts_iter)))
                    except StopIteration:
                        break
                process_loaded_batch(start, batch_count, loaded)
                progress_bar.update(1)
            progress_bar.close()

    if row != len(feature_items):
        trimmed = output_file.with_name(output_file.stem + "_trimmed.npy")
        trimmed_mmap = open_memmap(trimmed, mode="w+", dtype=np.float32, shape=(row, feature_shape[0], feature_shape[1]))
        trimmed_mmap[:] = mmap[:row]
        trimmed_mmap.flush()
        output_file.unlink()
        trimmed.rename(output_file)

    write_json(
        summary_path,
        {
            "output_file": str(output_file),
            "input_count": len(feature_items),
            "input_signature": input_signature,
            "feature_count": row,
            "feature_shape": list(feature_shape),
            "clip_seconds": args.clip_seconds,
            "requested_device": args.device,
            "device": device,
            "ncpu": args.ncpu,
            "batch_size": args.batch_size,
            "audio_loader_workers": audio_loader_workers,
            "prefetch_batches": prefetch_batches,
            "placement": default_placement,
            "placement_counts": input_placement_counts,
            "errors": errors[:50],
            "error_count": error_count,
        },
    )
    if progress_path.exists():
        progress_path.unlink()
    print(f"Wrote features {output_file} with {row} rows; errors={error_count}")


class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator: Iterable[Any]):
        self.generator = generator

    def __iter__(self) -> Iterable[Any]:
        return self.generator


def normalize_training_group_key(value: Any) -> str:
    key = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def validate_placement(value: Any, field_name: str = "placement") -> str:
    placement = str(value).strip().lower()
    if placement not in VALID_PLACEMENTS:
        valid = ", ".join(sorted(VALID_PLACEMENTS))
        raise ValueError(f"{field_name} must be one of {valid}, got {value!r}")
    return placement


def placement_path_key(path: Path) -> str:
    return f"path:{path.expanduser().resolve(strict=False)}"


def placement_rule_key(value: Any, base_dir: Path | None = None) -> str:
    raw_key = str(value).strip()
    if not raw_key:
        raise ValueError("placements contains an empty key")
    if raw_key in {"positive", "negative", "background", "default"}:
        return raw_key
    looks_like_jsonl_path = raw_key.endswith(".jsonl") or "/" in raw_key or "\\" in raw_key
    if looks_like_jsonl_path:
        path = Path(raw_key).expanduser()
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return placement_path_key(path)
    raise ValueError(
        f"Unsupported placements key {raw_key!r}. Use positive, negative, background, default, or a JSONL path."
    )


def parse_placements(value: Any, base_dir: Path | None = None) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        if stripped.startswith("{"):
            return parse_placements(json.loads(stripped), base_dir=base_dir)
        path = Path(stripped).expanduser()
        if path.exists():
            return parse_placements(read_json(path), base_dir=path.parent)
        raise ValueError("placements strings must be JSON objects or JSON file paths")
    if not isinstance(value, dict):
        raise TypeError(f"placements must be an object, JSON string, or JSON file path, got {type(value).__name__}")

    placements: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_value, str):
            raise TypeError(f"placements.{raw_key} must be a placement string")
        key = placement_rule_key(raw_key, base_dir=base_dir)
        placements[key] = validate_placement(raw_value, f"placements.{raw_key}")
    return placements


def resolve_placement(placements: dict[str, str], default: str, aliases: Iterable[Any]) -> str:
    for alias in aliases:
        raw_key = str(alias).strip()
        if raw_key and raw_key in placements:
            return placements[raw_key]
        key = normalize_training_group_key(alias)
        if key and key in placements:
            return placements[key]
    return validate_placement(default)


def placement_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        placement = validate_placement(value)
        counts[placement] = counts.get(placement, 0) + 1
    return dict(sorted(counts.items()))


def feature_input_signature(items: Iterable[tuple[Path, str]]) -> str:
    digest = hashlib.sha1()
    for path, placement in items:
        digest.update(os.fsencode(path))
        digest.update(b"\0")
        digest.update(validate_placement(placement).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_batch_counts(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        if stripped.startswith("{"):
            raw_value = json.loads(stripped)
        else:
            path = Path(stripped).expanduser()
            if path.exists():
                raw_value = read_json(path)
            else:
                raw_value = {}
                for item in stripped.split(","):
                    if not item.strip():
                        continue
                    if "=" not in item:
                        raise ValueError(
                            "Batch count strings must be JSON, a JSON file path, or comma-separated key=value pairs."
                        )
                    key, count = item.split("=", 1)
                    raw_value[key.strip()] = count.strip()
    elif isinstance(value, dict):
        raw_value = value
    else:
        raise TypeError(f"batch_counts must be an object, JSON string, or key=value string, got {type(value).__name__}")

    if not isinstance(raw_value, dict):
        raise ValueError("batch_counts must resolve to a JSON object")

    counts: dict[str, int] = {}
    for key, count in raw_value.items():
        normalized_key = normalize_training_group_key(key)
        if not normalized_key:
            raise ValueError(f"Invalid empty batch_counts key: {key!r}")
        int_count = int(count)
        if int_count < 0:
            raise ValueError(f"Batch count for {key!r} must be >= 0, got {count}")
        counts[normalized_key] = int_count
    return counts


def train_subset_name(record: dict[str, Any], fallback: str) -> str:
    subset = record.get("subset")
    if subset:
        return normalize_training_group_key(subset)
    input_jsonl = record.get("input_jsonl")
    if input_jsonl:
        parent = Path(str(input_jsonl)).parent.name
        if parent:
            return normalize_training_group_key(parent)
    source = record.get("source") or fallback
    return normalize_training_group_key(source)


def manifest_subset_name(jsonl_path: Path, source: str, dataset_index: int) -> str:
    if source == "positive":
        return "positive"
    if source == "background":
        return "background"
    path_text = str(jsonl_path).lower()
    parent = normalize_training_group_key(jsonl_path.parent.name)
    stem = normalize_training_group_key(jsonl_path.stem)
    if source == "negative":
        if "non_wuw" in path_text:
            return "negative_non_wuw"
        if "negative_cv" in path_text or "common_voice" in path_text or parent == "negative_cv":
            return "negative_cv"
        if parent.startswith("negative_"):
            return parent
        if stem.startswith("negative_"):
            return stem
        return f"negative_{dataset_index}"
    return normalize_training_group_key(source)


def train_group_aliases(group_name: str, subset_name: str) -> list[str]:
    aliases = {group_name, subset_name}
    if group_name.endswith("_train"):
        aliases.add(group_name[: -len("_train")])
    if subset_name.startswith("negative_"):
        short_name = subset_name[len("negative_"):]
        aliases.add(short_name)
        aliases.add(f"{short_name}_train")
    return sorted(normalize_training_group_key(alias) for alias in aliases if alias)


def add_feature_group_segment(
    groups: dict[str, dict[str, Any]],
    *,
    feature_file: Path,
    label: int,
    subset_name: str,
    start: int,
    stop: int,
) -> None:
    if stop <= start:
        return
    group_name = normalize_training_group_key(f"{subset_name}_train")
    group = groups.setdefault(
        group_name,
        {
            "name": group_name,
            "label": int(label),
            "aliases": [],
            "sources": [],
            "rows": 0,
        },
    )
    group["aliases"] = sorted(set(group["aliases"]) | set(train_group_aliases(group_name, subset_name)))
    group["sources"].append(
        {
            "path": str(feature_file),
            "start": int(start),
            "stop": int(stop),
            "rows": int(stop - start),
        }
    )
    group["rows"] = int(group["rows"]) + int(stop - start)


def feature_groups_from_manifests(
    feature_file: Path,
    input_manifests: list[Path],
    *,
    label: int,
    fallback_subset: str,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    row = 0
    current_subset: str | None = None
    current_start = 0

    def flush_current(stop: int) -> None:
        if current_subset is not None:
            add_feature_group_segment(
                groups,
                feature_file=feature_file,
                label=label,
                subset_name=current_subset,
                start=current_start,
                stop=stop,
            )

    for manifest in input_manifests:
        for record in read_jsonl(manifest):
            subset = train_subset_name(record, fallback_subset)
            if current_subset is None:
                current_subset = subset
                current_start = row
            elif subset != current_subset:
                flush_current(row)
                current_subset = subset
                current_start = row
            row += 1
    flush_current(row)

    feature_rows = int(np.load(feature_file, mmap_mode="r").shape[0])
    if row != feature_rows:
        raise ValueError(
            f"Feature row mismatch for {feature_file}: manifests describe {row} rows, "
            f"but the feature file has {feature_rows}. Regenerate features or point to matching manifests."
        )
    return list(groups.values())


def build_train_feature_groups(
    feature_files: dict[str, Path],
    train_manifests: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    groups.extend(
        feature_groups_from_manifests(
            feature_files["positive_train"],
            train_manifests["positive_train"],
            label=1,
            fallback_subset="positive",
        )
    )
    groups.extend(
        feature_groups_from_manifests(
            feature_files["negative_train"],
            train_manifests["negative_train"],
            label=0,
            fallback_subset="negative",
        )
    )
    groups.extend(
        feature_groups_from_manifests(
            feature_files["background_train"],
            train_manifests["background_train"],
            label=0,
            fallback_subset="background",
        )
    )
    return groups


def build_train_feature_groups_from_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for spec in specs:
        groups.extend(
            feature_groups_from_manifests(
                Path(spec["feature_file"]),
                [Path(path) for path in spec["manifests"]],
                label=int(spec["label"]),
                fallback_subset=str(spec["fallback_subset"]),
            )
        )
    return groups


def apply_feature_group_batch_counts(
    groups: list[dict[str, Any]],
    batch_counts: dict[str, int],
) -> list[dict[str, Any]]:
    unmatched = set(batch_counts)
    resolved: list[dict[str, Any]] = []
    available_aliases: set[str] = set()
    for group in groups:
        aliases = {normalize_training_group_key(group["name"])}
        aliases.update(normalize_training_group_key(alias) for alias in group.get("aliases", []))
        available_aliases.update(aliases)
        matched_key = next((alias for alias in aliases if alias in batch_counts), None)
        resolved_group = dict(group)
        resolved_group["batch_count"] = int(batch_counts[matched_key]) if matched_key else 0
        resolved_group["matched_batch_count_key"] = matched_key
        if matched_key:
            unmatched.discard(matched_key)
        resolved.append(resolved_group)

    if unmatched:
        available = ", ".join(sorted(available_aliases))
        unknown = ", ".join(sorted(unmatched))
        raise ValueError(f"Unknown batch_counts key(s): {unknown}. Available keys: {available}")
    if not any(group["label"] == 1 and int(group.get("batch_count", 0)) > 0 for group in resolved):
        raise ValueError("batch_counts must include a positive group, for example positive_train.")
    if not any(group["label"] == 0 and int(group.get("batch_count", 0)) > 0 for group in resolved):
        raise ValueError("batch_counts must include at least one negative group.")
    return resolved


def apply_data_file_batch_counts(
    data_files: dict[str, str],
    batch_counts: dict[str, int],
) -> dict[str, int]:
    aliases_by_key: dict[str, set[str]] = {}
    for key, path in data_files.items():
        aliases = {normalize_training_group_key(key), normalize_training_group_key(Path(path).stem)}
        if key == "positive":
            aliases.add("positive_train")
        aliases_by_key[key] = aliases

    unmatched = set(batch_counts)
    n_per_class: dict[str, int] = {}
    available_aliases = set().union(*aliases_by_key.values())
    for key, aliases in aliases_by_key.items():
        matched_key = next((alias for alias in aliases if alias in batch_counts), None)
        if matched_key:
            n_per_class[key] = int(batch_counts[matched_key])
            unmatched.discard(matched_key)

    if unmatched:
        available = ", ".join(sorted(available_aliases))
        unknown = ", ".join(sorted(unmatched))
        raise ValueError(f"Unknown batch_counts key(s): {unknown}. Available keys: {available}")
    if n_per_class.get("positive", 0) <= 0:
        raise ValueError("batch_counts must include positive or positive_train when using train-model directly.")
    if not any(key.startswith("negative_") and count > 0 for key, count in n_per_class.items()):
        raise ValueError("batch_counts must include at least one negative feature file, for example negative_0.")
    return {key: count for key, count in n_per_class.items() if count > 0}


class FeatureBatchSampler:
    def __init__(self, groups: list[dict[str, Any]], seed: int):
        self.rng = np.random.default_rng(seed)
        self.data: dict[str, Any] = {}
        self.groups: list[dict[str, Any]] = []
        for group in groups:
            batch_count = int(group.get("batch_count", 0))
            if batch_count <= 0:
                continue
            sources: list[dict[str, Any]] = []
            offset = 0
            for source in group.get("sources", []):
                path = str(Path(source["path"]).resolve())
                if path not in self.data:
                    self.data[path] = np.load(path, mmap_mode="r")
                start = int(source["start"])
                stop = int(source["stop"])
                rows = stop - start
                if start < 0 or stop > int(self.data[path].shape[0]) or rows <= 0:
                    raise ValueError(f"Invalid feature slice for {group['name']}: {source}")
                sources.append(
                    {
                        "path": path,
                        "start": start,
                        "stop": stop,
                        "offset_start": offset,
                        "offset_stop": offset + rows,
                    }
                )
                offset += rows
            if offset <= 0:
                raise ValueError(f"Feature group {group['name']} has no rows to sample")
            self.groups.append(
                {
                    "name": group["name"],
                    "label": int(group["label"]),
                    "batch_count": batch_count,
                    "rows": offset,
                    "sources": sources,
                }
            )
        if not self.groups:
            raise ValueError("No train feature groups have a positive batch_count")

    def __iter__(self) -> "FeatureBatchSampler":
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for group in self.groups:
            offsets = self.rng.integers(0, int(group["rows"]), size=int(group["batch_count"]))
            group_parts: list[np.ndarray] = []
            for source in group["sources"]:
                mask = (offsets >= source["offset_start"]) & (offsets < source["offset_stop"])
                if not np.any(mask):
                    continue
                indices = source["start"] + (offsets[mask] - source["offset_start"])
                group_parts.append(np.asarray(self.data[source["path"]][indices], dtype=np.float32))
            if not group_parts:
                continue
            x_group = np.vstack(group_parts).astype(np.float32, copy=False)
            x_parts.append(x_group)
            y_parts.append(np.full(x_group.shape[0], int(group["label"]), dtype=np.float32))

        x = np.vstack(x_parts).astype(np.float32, copy=False)
        y = np.concatenate(y_parts).astype(np.float32, copy=False)
        order = self.rng.permutation(y.shape[0])
        return x[order], y[order]


def summarize_feature_batch_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for group in groups:
        summary.append(
            {
                "name": group["name"],
                "label": group["label"],
                "batch_count": int(group.get("batch_count", 0)),
                "rows": int(group.get("rows", 0)),
                "aliases": group.get("aliases", []),
                "sources": group.get("sources", []),
            }
        )
    return summary


def make_val_loader(positive_features: Path, negative_features: Path | list[Path]) -> torch.utils.data.DataLoader:
    x_pos = np.load(positive_features)
    negative_files = negative_features if isinstance(negative_features, list) else [negative_features]
    x_neg = np.vstack([np.load(path) for path in negative_files]).astype(np.float32)
    labels = np.hstack((np.ones(x_pos.shape[0]), np.zeros(x_neg.shape[0]))).astype(np.float32)
    x = np.vstack((x_pos, x_neg)).astype(np.float32)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(labels)),
        batch_size=len(labels),
    )


def make_false_positive_loader(feature_files: list[Path]) -> torch.utils.data.DataLoader:
    arrays = [np.load(path) for path in feature_files]
    x = np.vstack(arrays).astype(np.float32)
    labels = np.zeros(x.shape[0]).astype(np.float32)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(labels)),
        batch_size=max(1, min(x.shape[0], 8192)),
    )


def command_train_model(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_model = output_dir / f"{args.model_name}.onnx"
    torch_model = output_dir / f"{args.model_name}.pt"
    training_summary = output_dir / "training_summary.json"
    if not getattr(args, "overwrite", False) and onnx_model.exists() and torch_model.exists() and training_summary.exists():
        print(f"Skipping training; complete model already exists: {onnx_model}")
        return

    positive_train = Path(args.positive_train_features).resolve()
    negative_train_files = [Path(path).resolve() for path in args.negative_train_features]
    positive_dev = Path(args.positive_dev_features).resolve()
    negative_dev_values = args.negative_dev_features if isinstance(args.negative_dev_features, list) else [args.negative_dev_features]
    negative_dev = [Path(path).resolve() for path in negative_dev_values]
    fp_files = [Path(path).resolve() for path in (args.false_positive_features or [])]
    if not fp_files:
        fp_files = negative_dev
    raw_batch_counts = getattr(args, "batch_counts", None)
    if raw_batch_counts is None:
        raw_batch_counts = getattr(args, "batch_counts_json", None)
    batch_counts = parse_batch_counts(raw_batch_counts)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required for training, but torch.cuda.is_available() is False in this process.")

    input_shape = np.load(positive_dev, mmap_mode="r").shape[1:]
    model = TrainModel(
        n_classes=1,
        input_shape=input_shape,
        model_type=args.model_type,
        layer_dim=args.layer_size,
        seconds_per_example=1280 * input_shape[0] / DEFAULT_SR,
    )

    data_files = {"positive": str(positive_train)}
    for ndx, path in enumerate(negative_train_files):
        data_files[f"negative_{ndx}"] = str(path)

    train_feature_groups = getattr(args, "train_feature_groups", None)
    resolved_feature_groups: list[dict[str, Any]] = []
    if batch_counts and train_feature_groups:
        resolved_feature_groups = apply_feature_group_batch_counts(train_feature_groups, batch_counts)
        batch_generator = FeatureBatchSampler(resolved_feature_groups, seed=args.seed)
        n_per_class = {
            group["name"]: int(group.get("batch_count", 0))
            for group in resolved_feature_groups
            if int(group.get("batch_count", 0)) > 0
        }
        train_batch_mode = "feature_groups"
    else:
        negative_keys = [key for key in data_files if key.startswith("negative_")]
        if batch_counts:
            n_per_class = apply_data_file_batch_counts(data_files, batch_counts)
        else:
            per_negative = max(1, args.batch_negative // max(1, len(negative_keys)))
            n_per_class = {"positive": args.batch_positive}
            n_per_class.update({key: per_negative for key in negative_keys})
        label_transforms = {"positive": lambda x: [1 for _ in x]}
        label_transforms.update({key: (lambda x: [0 for _ in x]) for key in negative_keys})

        batch_generator = mmap_batch_generator(
            data_files,
            n_per_class=n_per_class,
            label_transform_funcs=label_transforms,
        )
        train_batch_mode = "feature_files"
    x_train = torch.utils.data.DataLoader(IterDataset(batch_generator), batch_size=None, num_workers=0)
    x_val = make_val_loader(positive_dev, negative_dev)
    x_fp = make_false_positive_loader(fp_files)

    started = time.time()
    best_model = model.auto_train(
        X_train=x_train,
        X_val=x_val,
        false_positive_val_data=x_fp,
        steps=args.steps,
        max_negative_weight=args.max_negative_weight,
        target_fp_per_hour=args.target_false_positives_per_hour,
    )
    elapsed = time.time() - started
    model.export_model(model=best_model, model_name=args.model_name, output_dir=str(output_dir))
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_shape": tuple(input_shape),
            "model_type": args.model_type,
            "layer_size": args.layer_size,
            "model_name": args.model_name,
        },
        output_dir / f"{args.model_name}.pt",
    )

    summary = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "onnx_model": str(output_dir / f"{args.model_name}.onnx"),
        "torch_model": str(output_dir / f"{args.model_name}.pt"),
        "input_shape": list(input_shape),
        "elapsed_seconds": elapsed,
        "steps": args.steps,
        "batch_positive": args.batch_positive,
        "batch_negative": args.batch_negative,
        "batch_counts": batch_counts or None,
        "effective_batch_counts": n_per_class,
        "train_batch_mode": train_batch_mode,
        "train_feature_groups": summarize_feature_batch_groups(resolved_feature_groups) if resolved_feature_groups else None,
        "layer_size": args.layer_size,
        "model_type": args.model_type,
        "max_negative_weight": args.max_negative_weight,
        "target_false_positives_per_hour": args.target_false_positives_per_hour,
        "require_cuda": args.require_cuda,
        "seed": args.seed,
        "negative_train_features": [str(path) for path in negative_train_files],
        "positive_train_features": str(positive_train),
        "positive_dev_features": str(positive_dev),
        "negative_dev_features": [str(path) for path in negative_dev],
        "false_positive_features": [str(path) for path in fp_files],
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "history": dict(model.history),
        "best_model_scores": model.best_model_scores,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(f"Trained {args.model_name}; ONNX saved to {summary['onnx_model']}")


def prediction_scores(
    model: openwakeword.Model,
    audio: torch.Tensor,
    model_label: str,
    padding: int,
    chunk_size: int = 1280,
) -> list[float]:
    pcm = (audio.clamp(-1.0, 1.0).numpy() * 32767).astype(np.int16)
    model.reset()
    predictions = model.predict_clip(pcm, padding=padding, chunk_size=chunk_size)
    return [float(prediction.get(model_label, next(iter(prediction.values())))) for prediction in predictions]


def round_metric(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def score_summary(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {
            "score_count": 0,
            "max_score": None,
            "min_score": None,
            "average_score": None,
            "mean_score": None,
            "median_score": None,
        }
    return {
        "score_count": len(scores),
        "max_score": round_metric(max(scores)),
        "min_score": round_metric(min(scores)),
        "average_score": round_metric(float(np.mean(scores))),
        "mean_score": round_metric(float(np.mean(scores))),
        "median_score": round_metric(float(np.median(scores))),
    }


def score_windows(
    scores: list[float],
    duration_seconds: float,
    padding_seconds: float,
    chunk_size: int,
    sample_rate: int,
    model_window_seconds: float,
) -> list[dict[str, Any]]:
    frame_seconds = chunk_size / sample_rate
    windows: list[dict[str, Any]] = []
    for frame_ndx, score in enumerate(scores):
        chunk_start = frame_ndx * frame_seconds - padding_seconds
        chunk_end = chunk_start + frame_seconds
        raw_start = chunk_end - model_window_seconds
        raw_end = chunk_end
        start = max(0.0, min(duration_seconds, raw_start))
        end = max(0.0, min(duration_seconds, raw_end))
        if end < start:
            start, end = end, start
        windows.append(
            {
                "frame_index": frame_ndx,
                "start_time": round_metric(start),
                "end_time": round_metric(end),
                "raw_start_time": round_metric(raw_start),
                "raw_end_time": round_metric(raw_end),
                "score": round_metric(score),
            }
        )
    return windows


def threshold_events(windows: list[dict[str, Any]], threshold: float, debounce_seconds: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    last_event_time = -1e9
    for window in windows:
        score = float(window["score"])
        event_time = float(window["end_time"])
        if score >= threshold and event_time - last_event_time >= debounce_seconds:
            event = {
                "event_index": len(events),
                "frame_index": window["frame_index"],
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "score": window["score"],
            }
            events.append(event)
            last_event_time = event_time
    return events


def load_evaluation_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = dict(EVALUATION_DEFAULTS)
    config_path = getattr(args, "evaluation_config", None)
    if config_path:
        raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if "evaluation" in raw_config and isinstance(raw_config["evaluation"], dict):
            raw_config = raw_config["evaluation"]
        for key, value in raw_config.items():
            if key in EVALUATION_CONFIG_KEYS:
                config[key] = value

    for key in EVALUATION_CONFIG_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if not config.get("model"):
        raise ValueError("Evaluation requires a model path from --model or evaluation_config.model.")
    if not config.get("output_json"):
        raise ValueError("Evaluation requires --output-json or evaluation_config.output_json.")

    output_json = Path(config["output_json"]).resolve()
    config["model"] = str(Path(config["model"]).resolve())
    if config.get("model_dir"):
        config["model_dir"] = str(Path(config["model_dir"]).resolve())
    if not config.get("negative_manifest") and config.get("negative_speech_manifest"):
        config["negative_manifest"] = config["negative_speech_manifest"]

    for key in ["positive_manifest", "negative_manifest", "negative_speech_manifest", "background_manifest"]:
        if config.get(key):
            config[key] = str(Path(config[key]).resolve())
    config["output_json"] = str(output_json)
    config["details_jsonl"] = str(Path(config.get("details_jsonl") or output_json.with_name(f"{output_json.stem}_details.jsonl")).resolve())
    config["abnormal_jsonl"] = str(Path(config.get("abnormal_jsonl") or output_json.with_name(f"{output_json.stem}_abnormal.jsonl")).resolve())
    config["output_config_json"] = str(Path(config.get("output_config_json") or output_json.with_name("evaluation_config.json")).resolve())
    return config


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def evaluate_one_record_scores(
    model: openwakeword.Model,
    model_label: str,
    record: dict[str, Any],
    set_name: str,
    index: int,
    expected_label: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_rate = int(config["sample_rate"])
    chunk_size = int(config["chunk_size"])
    audio = load_audio_float(Path(record["path"]), sr=sample_rate)
    duration_seconds = audio.numel() / sample_rate
    padding_seconds = float(config["positive_padding"] if expected_label == 1 else config["negative_padding"])
    scores = prediction_scores(
        model,
        audio,
        model_label,
        padding=int(padding_seconds),
        chunk_size=chunk_size,
    )
    windows = score_windows(
        scores,
        duration_seconds=duration_seconds,
        padding_seconds=padding_seconds,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        model_window_seconds=float(config["model_window_seconds"]),
    )
    best_window = max(windows, key=lambda window: float(window["score"])) if windows else None
    detail: dict[str, Any] = {
        "set": set_name,
        "index": index,
        "id": record.get("id"),
        "path": str(Path(record["path"]).resolve()),
        "source": record.get("source"),
        "input_jsonl": record.get("input_jsonl"),
        "dataset_index": record.get("dataset_index"),
        "subset": evaluation_subset_name(record, set_name),
        "sentence": record.get("sentence"),
        "expected_label": expected_label,
        "padding_seconds": padding_seconds,
        "chunk_size": chunk_size,
        "frame_seconds": round_metric(chunk_size / sample_rate),
        "model_window_seconds": float(config["model_window_seconds"]),
        "duration_seconds": round_metric(duration_seconds),
        "best_window": best_window,
    }
    detail.update(score_summary(scores))
    if config["record_window_scores"]:
        detail["sliding_windows"] = windows
    return detail


def evaluate_one_record(
    model: openwakeword.Model,
    model_label: str,
    record: dict[str, Any],
    set_name: str,
    index: int,
    expected_label: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Legacy single-threshold detail built from threshold-independent scores."""
    detail = evaluate_one_record_scores(model, model_label, record, set_name, index, expected_label, config)
    windows = detail.get("sliding_windows", [])
    events = threshold_events(windows, float(config["threshold"]), float(config["debounce_seconds"]))
    windows_above_threshold = [window for window in windows if float(window["score"]) >= float(config["threshold"])]
    detected = bool(events)
    detail.update(
        {
            "threshold": float(config["threshold"]),
            "debounce_seconds": float(config["debounce_seconds"]),
            "detected": detected,
            "false_reject": expected_label == 1 and not detected,
            "false_accept": expected_label == 0 and detected,
            "events": events,
            "event_count": len(events),
            "windows_above_threshold": windows_above_threshold,
            "windows_above_threshold_count": len(windows_above_threshold),
        }
    )
    return detail


def sanitize_summary_key(value: Any) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def evaluation_subset_name(record: dict[str, Any], set_name: str) -> str:
    input_jsonl = record.get("input_jsonl")
    if input_jsonl:
        input_path = Path(str(input_jsonl))
        return sanitize_summary_key(input_path.parent.name or input_path.stem)

    dataset_index = record.get("dataset_index")
    if dataset_index is not None:
        return sanitize_summary_key(f"{set_name}_{dataset_index}")
    return sanitize_summary_key(set_name)


def new_eval_accumulator() -> dict[str, Any]:
    return {
        "evaluated": 0,
        "seconds": 0.0,
        "false_rejects": 0,
        "false_accept_clips": 0,
        "false_accept_events": 0,
        "max_scores": [],
        "average_scores": [],
        "errors": [],
    }


def add_detail_to_accumulator(accumulator: dict[str, Any], detail: dict[str, Any]) -> None:
    accumulator["evaluated"] += 1
    accumulator["seconds"] += float(detail["duration_seconds"] or 0.0)
    if detail["max_score"] is not None:
        accumulator["max_scores"].append(float(detail["max_score"]))
    if detail["average_score"] is not None:
        accumulator["average_scores"].append(float(detail["average_score"]))
    if detail["false_reject"]:
        accumulator["false_rejects"] += 1
    if detail["false_accept"]:
        accumulator["false_accept_clips"] += 1
        accumulator["false_accept_events"] += int(detail["event_count"])


def add_error_to_accumulator(accumulator: dict[str, Any], error: dict[str, Any]) -> None:
    accumulator["errors"].append(error)


def finalize_eval_metrics(
    accumulator: dict[str, Any],
    *,
    clips_requested: int,
    expected_label: int,
) -> dict[str, Any]:
    evaluated = int(accumulator["evaluated"])
    seconds = float(accumulator["seconds"])
    hours = seconds / 3600.0
    max_scores = accumulator["max_scores"]
    average_scores = accumulator["average_scores"]
    errors = accumulator["errors"]
    metrics: dict[str, Any] = {
        "clips": evaluated,
        "clips_evaluated": evaluated,
        "clips_requested": clips_requested,
        "evaluated_seconds": round_metric(seconds),
        "evaluated_hours": round_metric(hours),
        "error_count": len(errors),
        "errors": errors[:50],
        "mean_max_score": round_metric(float(np.mean(max_scores))) if max_scores else None,
        "median_max_score": round_metric(float(np.median(max_scores))) if max_scores else None,
        "mean_average_score": round_metric(float(np.mean(average_scores))) if average_scores else None,
        "average_score": round_metric(float(np.mean(average_scores))) if average_scores else None,
    }
    if expected_label == 1:
        false_rejects = int(accumulator["false_rejects"])
        metrics.update(
            {
                "misses": false_rejects,
                "false_rejects": false_rejects,
                "detected_clips": evaluated - false_rejects,
                "false_reject_rate": false_rejects / evaluated if evaluated else None,
            }
        )
    else:
        false_accept_events = int(accumulator["false_accept_events"])
        metrics.update(
            {
                "false_accept_clips": int(accumulator["false_accept_clips"]),
                "false_accept_events": false_accept_events,
                "false_accepts_per_hour": false_accept_events / hours if hours else None,
            }
        )
    return metrics


def add_set_metrics(output: dict[str, Any], set_name: str, metrics: dict[str, Any]) -> None:
    output["sets"][set_name] = metrics
    for subset, subset_metrics in metrics.get("subsets", {}).items():
        if subset == sanitize_summary_key(set_name):
            continue
        output_key = subset
        if output_key in output["sets"]:
            output_key = sanitize_summary_key(f"{set_name}_{subset}")
        output["sets"][output_key] = subset_metrics


def evaluate_manifest_set(
    model: openwakeword.Model,
    model_label: str,
    set_name: str,
    manifest: str,
    expected_label: int,
    config: dict[str, Any],
    detail_handle: Any,
    abnormal_handle: Any,
) -> dict[str, Any]:
    records = read_jsonl(Path(manifest))
    if expected_label == 1 and config.get("limit_positive"):
        records = records[:int(config["limit_positive"])]

    aggregate = new_eval_accumulator()
    subset_accumulators: dict[str, dict[str, Any]] = {}
    subset_requested: dict[str, int] = {}
    subset_inputs: dict[str, set[str]] = {}
    for record in records:
        subset = evaluation_subset_name(record, set_name)
        subset_accumulators.setdefault(subset, new_eval_accumulator())
        subset_requested[subset] = subset_requested.get(subset, 0) + 1
        if record.get("input_jsonl"):
            subset_inputs.setdefault(subset, set()).add(str(record["input_jsonl"]))

    for ndx, record in enumerate(tqdm(records, desc=f"Evaluate {set_name}")):
        if (
            expected_label == 0
            and config.get("limit_negative_seconds")
            and float(aggregate["seconds"]) >= float(config["limit_negative_seconds"])
        ):
            break
        subset = evaluation_subset_name(record, set_name)
        subset_accumulator = subset_accumulators.setdefault(subset, new_eval_accumulator())
        try:
            detail = evaluate_one_record(model, model_label, record, set_name, ndx, expected_label, config)
            write_jsonl_record(detail_handle, detail)
            if detail["false_reject"] or detail["false_accept"]:
                write_jsonl_record(abnormal_handle, detail)
            add_detail_to_accumulator(aggregate, detail)
            add_detail_to_accumulator(subset_accumulator, detail)
        except Exception as exc:
            error = {
                "set": set_name,
                "subset": subset,
                "index": ndx,
                "id": record.get("id"),
                "path": record.get("path"),
                "input_jsonl": record.get("input_jsonl"),
                "dataset_index": record.get("dataset_index"),
                "error": repr(exc),
            }
            add_error_to_accumulator(aggregate, error)
            add_error_to_accumulator(subset_accumulator, error)
            write_jsonl_record(abnormal_handle, error)

    metrics = finalize_eval_metrics(aggregate, clips_requested=len(records), expected_label=expected_label)
    subsets: dict[str, Any] = {}
    for subset, accumulator in sorted(subset_accumulators.items()):
        subset_metrics = finalize_eval_metrics(
            accumulator,
            clips_requested=subset_requested.get(subset, 0),
            expected_label=expected_label,
        )
        subset_metrics["parent_set"] = set_name
        subset_metrics["subset"] = subset
        if subset_inputs.get(subset):
            subset_metrics["input_jsonl"] = sorted(subset_inputs[subset])
        subsets[subset] = subset_metrics
    if len(subsets) > 1 or any(subset != sanitize_summary_key(set_name) for subset in subsets):
        metrics["subsets"] = subsets
    return metrics


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_evaluation_config(args)
    if not getattr(args, "overwrite", False) and Path(config["output_json"]).exists():
        print(f"Skipping evaluation; complete output already exists: {config['output_json']}")
        return

    model_path = Path(config["model"]).resolve()
    model_label = model_path.stem
    model_kwargs = feature_model_paths(config.get("model_dir"))
    model = openwakeword.Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        **model_kwargs,
    )
    output: dict[str, Any] = {
        "model": str(model_path),
        "threshold": config["threshold"],
        "debounce_seconds": config["debounce_seconds"],
        "evaluation_config": config["output_config_json"],
        "detail_jsonl": config["details_jsonl"],
        "abnormal_jsonl": config["abnormal_jsonl"],
        "sets": {},
    }

    details_path = Path(config["details_jsonl"])
    abnormal_path = Path(config["abnormal_jsonl"])
    details_path.parent.mkdir(parents=True, exist_ok=True)
    abnormal_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(config["output_config_json"]), config)
    with details_path.open("w", encoding="utf-8") as detail_handle, abnormal_path.open("w", encoding="utf-8") as abnormal_handle:
        if config.get("positive_manifest"):
            add_set_metrics(
                output,
                "positive",
                evaluate_manifest_set(
                    model,
                    model_label,
                    "positive",
                    config["positive_manifest"],
                    1,
                    config,
                    detail_handle,
                    abnormal_handle,
                ),
            )

        if config.get("negative_manifest"):
            add_set_metrics(
                output,
                "negative",
                evaluate_manifest_set(
                    model,
                    model_label,
                    "negative",
                    config["negative_manifest"],
                    0,
                    config,
                    detail_handle,
                    abnormal_handle,
                ),
            )
        if config.get("background_manifest"):
            add_set_metrics(
                output,
                "background",
                evaluate_manifest_set(
                    model,
                    model_label,
                    "background",
                    config["background_manifest"],
                    0,
                    config,
                    detail_handle,
                    abnormal_handle,
                ),
            )

    output_path = Path(config["output_json"]).resolve()
    write_json(output_path, output)
    print(json.dumps(output, indent=2, default=json_default))


def run_step(name: str, func: Any, namespace: argparse.Namespace, status: dict[str, Any]) -> None:
    started = time.time()
    print(f"\n### {name}", flush=True)
    func(namespace)
    status[name] = {"status": "done", "elapsed_seconds": time.time() - started}


def listify_paths(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        paths: list[str] = []
        for item in value:
            paths.extend(listify_paths(item, field_name))
        return paths
    raise TypeError(f"{field_name} must be a string path or a list of string paths")


def split_manifest_paths(split_manifests: dict[str, Any], kind: str, split_name: str) -> list[str]:
    kind_value = split_manifests.get(kind)
    if not isinstance(kind_value, dict):
        raise ValueError(f"split_manifests.{kind} must be an object")
    if split_name not in kind_value:
        raise ValueError(f"Missing split_manifests.{kind}.{split_name}")
    paths = listify_paths(kind_value[split_name], f"split_manifests.{kind}.{split_name}")
    if not paths:
        raise ValueError(f"split_manifests.{kind}.{split_name} is empty")
    return paths


def materialize_split_manifest(
    input_manifests: list[str],
    output_path: Path,
    *,
    label: int,
    source: str,
    placements: dict[str, str] | None = None,
    default_placement: str = "random",
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    all_placements: list[str] = []
    for dataset_index, manifest in enumerate(input_manifests):
        jsonl_path = Path(manifest).expanduser().resolve()
        raw_records = read_jsonl(jsonl_path)
        source_summary = {
            "jsonl_path": str(jsonl_path),
            "records": len(raw_records),
            "source": source,
            "subset": manifest_subset_name(jsonl_path, source, dataset_index),
            "placement_counts": {},
        }
        subset = source_summary["subset"]
        source_placements: list[str] = []
        for raw_record in raw_records:
            path = record_source_path(raw_record, jsonl_path, None)
            record = replace_audio_path(raw_record, path)
            record["label"] = label
            record["source"] = record.get("source") or source
            record["id"] = record.get("id") or stable_id(str(path), len(records))
            record["input_jsonl"] = str(jsonl_path)
            record["dataset_index"] = dataset_index
            record["subset"] = record.get("subset") or subset
            record_default_placement = record.get("placement", default_placement)
            record["placement"] = resolve_placement(
                placements or {},
                record_default_placement,
                [
                    placement_path_key(jsonl_path),
                    source,
                    "default",
                ],
            )
            source_placements.append(record["placement"])
            all_placements.append(record["placement"])
            records.append(record)
        source_summary["placement_counts"] = placement_counts(source_placements)
        sources.append(source_summary)

    count = write_jsonl(output_path, records)
    return {
        "output_manifest": str(output_path),
        "count": count,
        "label": label,
        "source": source,
        "placement_counts": placement_counts(all_placements),
        "inputs": sources,
    }


def write_feature_input_manifests_by_subset(
    input_manifests: list[Path],
    output_dir: Path,
    *,
    split_name: str,
    fallback_subset: str,
) -> dict[str, Path]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for manifest in input_manifests:
        for record in read_jsonl(manifest):
            subset = train_subset_name(record, fallback_subset)
            name = normalize_training_group_key(f"{subset}_{split_name}")
            grouped.setdefault(name, []).append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    summary: dict[str, Any] = {"split_name": split_name, "sets": {}}
    for name, records in sorted(grouped.items()):
        output_path = output_dir / f"{name}.jsonl"
        count = write_jsonl(output_path, records)
        outputs[name] = output_path
        summary["sets"][name] = {
            "output_manifest": str(output_path),
            "count": count,
            "input_manifests": [str(path) for path in input_manifests],
        }
    write_json(output_dir / f"{split_name}_feature_inputs.summary.json", summary)
    return outputs


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config field {name} must be an object")
    return value


def command_run_from_splits(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config_file(str(config_path))
    if not config:
        raise ValueError("run-from-splits requires --config")

    if args.experiment_dir:
        config["experiment_dir"] = args.experiment_dir
    if args.model_name:
        config["model_name"] = args.model_name
    if args.skip_download is not None:
        config["skip_download"] = args.skip_download
    if args.overwrite is not None:
        config["overwrite"] = args.overwrite

    training_cfg = config_section(config, "training")
    feature_cfg = config_section(config, "features")
    augmentation_cfg = config_section(config, "augmentation")
    evaluation_cfg = config_section(config, "evaluation")
    output_cfg = config_section(config, "outputs")

    if args.steps is not None:
        training_cfg["steps"] = args.steps
    if args.device is not None:
        feature_cfg["device"] = args.device
    if args.require_cuda is not None:
        training_cfg["require_cuda"] = args.require_cuda

    split_manifests = config_section(config, "split_manifests")
    exp = Path(config.get("experiment_dir") or RUN_EXPERIMENT_DEFAULTS["experiment_dir"]).expanduser().resolve()
    exp.mkdir(parents=True, exist_ok=True)
    model_name = str(config.get("model_name") or RUN_EXPERIMENT_DEFAULTS["model_name"])
    sample_rate = int(config.get("sample_rate", DEFAULT_SR))
    clip_seconds = float(config.get("clip_seconds", 2.0))
    seed = int(config.get("seed", 1337))
    overwrite = bool(config.get("overwrite", False))

    status: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "split_manifests",
        "torch_cuda_available": torch.cuda.is_available(),
    }
    write_json(exp / "experiment_config.resolved.json", config)

    model_assets_dir = Path(config.get("model_dir") or exp / "models").expanduser().resolve()
    if not bool(config.get("skip_download", False)):
        run_step(
            "download_models",
            command_download_models,
            argparse.Namespace(output_dir=str(model_assets_dir), models=["all"]),
            status,
        )

    manifests_dir = Path(output_cfg.get("manifests_dir") or exp / "manifests").expanduser().resolve()
    augmented_dir = Path(output_cfg.get("augmented_dir") or exp / "audio" / "augmented").expanduser().resolve()
    features_dir = Path(output_cfg.get("features_dir") or exp / "features").expanduser().resolve()
    model_out = Path(output_cfg.get("model_dir") or exp / "trained_model").expanduser().resolve()
    evaluation_dir = Path(output_cfg.get("evaluation_dir") or exp / "evaluation").expanduser().resolve()
    conversion_cfg = config_section(config, "conversion")
    placements = parse_placements(conversion_cfg.get("placements", {}), base_dir=config_path.parent)

    def placement_for_source(source: str, fallback: str) -> str:
        return resolve_placement(placements, fallback, [source, "default"])

    started = time.time()
    manifest_jobs = [
        ("positive_train", "train", split_manifest_paths(split_manifests, "positive", "train"), 1, "positive", "end"),
        ("positive_dev", "dev", split_manifest_paths(split_manifests, "positive", "dev"), 1, "positive", "end"),
        ("positive_test", "test", split_manifest_paths(split_manifests, "positive", "test"), 1, "positive", "end"),
        ("negative_train", "train", split_manifest_paths(split_manifests, "negative", "train"), 0, "negative", "random"),
        ("negative_dev", "dev", split_manifest_paths(split_manifests, "negative", "dev"), 0, "negative", "random"),
        ("negative_test", "test", split_manifest_paths(split_manifests, "negative", "test"), 0, "negative", "random"),
        ("background_train", "train", split_manifest_paths(split_manifests, "background", "train"), 0, "background", "random"),
        ("background_dev", "dev", split_manifest_paths(split_manifests, "background", "dev"), 0, "background", "random"),
        ("background_test", "test", split_manifest_paths(split_manifests, "background", "test"), 0, "background", "random"),
    ]
    materialized_manifests: dict[str, Path] = {}
    manifest_summary: dict[str, Any] = {"sets": {}, "placements": placements}
    for name, _split_name, input_paths, label, source, default_placement in manifest_jobs:
        output_manifest = manifests_dir / f"{name}.jsonl"
        materialized_manifests[name] = output_manifest
        manifest_summary["sets"][name] = materialize_split_manifest(
            input_paths,
            output_manifest,
            label=label,
            source=source,
            placements=placements,
            default_placement=default_placement,
        )
    write_json(manifests_dir / "manifest_summary.json", manifest_summary)
    status["materialize_manifests"] = {"status": "done", "elapsed_seconds": time.time() - started}

    positive_placement = placement_for_source("positive", "end")
    negative_placement = placement_for_source("negative", "random")

    augmented_manifests: dict[str, Path] = {}
    if bool(augmentation_cfg.get("enabled", True)):
        augmentation_jobs = [
            ("positive_train", positive_placement, bool(augmentation_cfg.get("augment_positive_train", True))),
            ("negative_train", negative_placement, bool(augmentation_cfg.get("augment_negative_train", True))),
        ]
        for name, placement, enabled in augmentation_jobs:
            if not enabled:
                continue
            out_manifest = augmented_dir / f"{name}.jsonl"
            augmented_manifests[name] = out_manifest
            run_step(
                f"augment_{name}",
                command_augment_audio,
                argparse.Namespace(
                    input_manifest=str(materialized_manifests[name]),
                    input_dir=None,
                    noise_manifest=[str(materialized_manifests["background_train"])],
                    noise_dir=[],
                    output_dir=str(augmented_dir / name),
                    output_manifest=str(out_manifest),
                    rounds=int(augmentation_cfg.get("rounds", 1)),
                    snr_low=float(augmentation_cfg.get("snr_low", -5.0)),
                    snr_high=float(augmentation_cfg.get("snr_high", 15.0)),
                    artificial_prob=float(augmentation_cfg.get("artificial_prob", 0.15)),
                    random_gain_db=float(augmentation_cfg.get("random_gain_db", 3.0)),
                    clip_seconds=clip_seconds,
                    sample_rate=sample_rate,
                    placement=placement,
                    seed=seed,
                    overwrite=bool(augmentation_cfg.get("overwrite", overwrite)),
                    workers=int(augmentation_cfg.get("workers", RUN_EXPERIMENT_DEFAULTS["augment_workers"])),
                ),
                status,
            )

    feature_input_dir = manifests_dir / "feature_inputs"
    feature_jobs: dict[str, dict[str, Any]] = {}

    def add_feature_jobs(
        *,
        split_name: str,
        source: str,
        label: int,
        input_manifests: list[Path],
        placement: str,
    ) -> None:
        split_inputs = write_feature_input_manifests_by_subset(
            input_manifests,
            feature_input_dir / split_name,
            split_name=split_name,
            fallback_subset=source,
        )
        for name, manifest in split_inputs.items():
            feature_jobs[name] = {
                "manifests": [manifest],
                "placement": placement,
                "label": int(label),
                "source": source,
                "split": split_name,
                "fallback_subset": name[: -len(f"_{split_name}")] if name.endswith(f"_{split_name}") else source,
            }

    add_feature_jobs(
        split_name="train",
        source="positive",
        label=1,
        input_manifests=[materialized_manifests["positive_train"]]
        + ([augmented_manifests["positive_train"]] if "positive_train" in augmented_manifests else []),
        placement=placement_for_source("positive", "end"),
    )
    add_feature_jobs(
        split_name="dev",
        source="positive",
        label=1,
        input_manifests=[materialized_manifests["positive_dev"]],
        placement=placement_for_source("positive", "end"),
    )
    add_feature_jobs(
        split_name="test",
        source="positive",
        label=1,
        input_manifests=[materialized_manifests["positive_test"]],
        placement=placement_for_source("positive", "end"),
    )
    add_feature_jobs(
        split_name="train",
        source="negative",
        label=0,
        input_manifests=[materialized_manifests["negative_train"]]
        + ([augmented_manifests["negative_train"]] if "negative_train" in augmented_manifests else []),
        placement=placement_for_source("negative", "random"),
    )
    add_feature_jobs(
        split_name="dev",
        source="negative",
        label=0,
        input_manifests=[materialized_manifests["negative_dev"]],
        placement=placement_for_source("negative", "random"),
    )
    add_feature_jobs(
        split_name="test",
        source="negative",
        label=0,
        input_manifests=[materialized_manifests["negative_test"]],
        placement=placement_for_source("negative", "random"),
    )
    add_feature_jobs(
        split_name="train",
        source="background",
        label=0,
        input_manifests=[materialized_manifests["background_train"]],
        placement=placement_for_source("background", "random"),
    )
    add_feature_jobs(
        split_name="dev",
        source="background",
        label=0,
        input_manifests=[materialized_manifests["background_dev"]],
        placement=placement_for_source("background", "random"),
    )
    add_feature_jobs(
        split_name="test",
        source="background",
        label=0,
        input_manifests=[materialized_manifests["background_test"]],
        placement=placement_for_source("background", "random"),
    )

    feature_files: dict[str, Path] = {}
    for name, spec in sorted(feature_jobs.items()):
        out_file = features_dir / f"{name}.npy"
        feature_files[name] = out_file
        spec["feature_file"] = out_file
        run_step(
            f"features_{name}",
            command_generate_features,
            argparse.Namespace(
                audio_manifest=[str(path) for path in spec["manifests"]],
                audio_dir=[],
                output_file=str(out_file),
                model_dir=str(model_assets_dir),
                batch_size=int(feature_cfg.get("batch_size", RUN_EXPERIMENT_DEFAULTS["batch_size"])),
                audio_loader_workers=int(feature_cfg.get("audio_loader_workers", RUN_EXPERIMENT_DEFAULTS["audio_loader_workers"])),
                prefetch_batches=int(feature_cfg.get("prefetch_batches", RUN_EXPERIMENT_DEFAULTS["prefetch_batches"])),
                ncpu=int(feature_cfg.get("ncpu", RUN_EXPERIMENT_DEFAULTS["ncpu"])),
                device=str(feature_cfg.get("device", RUN_EXPERIMENT_DEFAULTS["device"])),
                limit=None,
                clip_seconds=clip_seconds,
                sample_rate=sample_rate,
                placement=spec["placement"],
                seed=seed,
                overwrite=overwrite,
            ),
            status,
        )

    training_batch_counts = training_cfg.get("batch_counts", training_cfg.get("batch_sample_counts"))
    train_feature_groups = None
    if training_batch_counts is not None:
        train_feature_groups = build_train_feature_groups_from_specs(
            [spec for spec in feature_jobs.values() if spec["split"] == "train"]
        )

    negative_train_feature_names = [
        name
        for name, spec in sorted(feature_jobs.items())
        if spec["split"] == "train" and int(spec["label"]) == 0
    ]
    negative_dev_feature_names = [
        name
        for name, spec in sorted(feature_jobs.items())
        if spec["split"] == "dev" and spec["source"] == "negative"
    ]

    run_step(
        "train_model",
        command_train_model,
        argparse.Namespace(
            positive_train_features=str(feature_files["positive_train"]),
            negative_train_features=[str(feature_files[name]) for name in negative_train_feature_names],
            positive_dev_features=str(feature_files["positive_dev"]),
            negative_dev_features=[str(feature_files[name]) for name in negative_dev_feature_names],
            false_positive_features=[str(feature_files["background_dev"])],
            output_dir=str(model_out),
            model_name=model_name,
            steps=int(training_cfg.get("steps", RUN_EXPERIMENT_DEFAULTS["steps"])),
            batch_positive=int(training_cfg.get("batch_positive", RUN_EXPERIMENT_DEFAULTS["batch_positive"])),
            batch_negative=int(training_cfg.get("batch_negative", RUN_EXPERIMENT_DEFAULTS["batch_negative"])),
            layer_size=int(training_cfg.get("layer_size", RUN_EXPERIMENT_DEFAULTS["layer_size"])),
            model_type=str(training_cfg.get("model_type", RUN_EXPERIMENT_DEFAULTS["model_type"])),
            max_negative_weight=float(training_cfg.get("max_negative_weight", RUN_EXPERIMENT_DEFAULTS["max_negative_weight"])),
            target_false_positives_per_hour=float(
                training_cfg.get("target_false_positives_per_hour", RUN_EXPERIMENT_DEFAULTS["target_false_positives_per_hour"])
            ),
            batch_counts=training_batch_counts,
            train_feature_groups=train_feature_groups,
            seed=seed,
            require_cuda=bool(training_cfg.get("require_cuda", False)),
            overwrite=overwrite,
        ),
        status,
    )

    eval_output = evaluation_dir / "eval_summary.json"
    eval_config_output = evaluation_dir / "evaluation_config.json"
    run_step(
        "evaluate",
        command_evaluate,
        argparse.Namespace(
            evaluation_config=None,
            model=str(model_out / f"{model_name}.onnx"),
            model_dir=str(model_assets_dir),
            positive_manifest=str(materialized_manifests["positive_test"]),
            negative_manifest=str(materialized_manifests["negative_test"]),
            negative_speech_manifest=None,
            background_manifest=str(materialized_manifests["background_test"]),
            output_json=str(eval_output),
            details_jsonl=str(evaluation_dir / "eval_details.jsonl"),
            abnormal_jsonl=str(evaluation_dir / "eval_abnormal.jsonl"),
            output_config_json=str(eval_config_output),
            threshold=evaluation_cfg.get("threshold", EVALUATION_DEFAULTS["threshold"]),
            debounce_seconds=evaluation_cfg.get("debounce_seconds", EVALUATION_DEFAULTS["debounce_seconds"]),
            positive_padding=evaluation_cfg.get("positive_padding", EVALUATION_DEFAULTS["positive_padding"]),
            negative_padding=evaluation_cfg.get("negative_padding", EVALUATION_DEFAULTS["negative_padding"]),
            chunk_size=evaluation_cfg.get("chunk_size", EVALUATION_DEFAULTS["chunk_size"]),
            model_window_seconds=evaluation_cfg.get("model_window_seconds", EVALUATION_DEFAULTS["model_window_seconds"]),
            record_window_scores=evaluation_cfg.get("record_window_scores", EVALUATION_DEFAULTS["record_window_scores"]),
            limit_positive=None,
            limit_negative_seconds=None,
            sample_rate=sample_rate,
            overwrite=overwrite,
        ),
        status,
    )

    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["outputs"] = {
        "experiment_dir": str(exp),
        "model": str(model_out / f"{model_name}.onnx"),
        "training_summary": str(model_out / "training_summary.json"),
        "evaluation": str(eval_output),
        "evaluation_config": str(eval_config_output),
        "evaluation_details": str(evaluation_dir / "eval_details.jsonl"),
        "evaluation_abnormal": str(evaluation_dir / "eval_abnormal.jsonl"),
        "manifest_summary": str(manifests_dir / "manifest_summary.json"),
    }
    write_json(exp / "experiment_status.json", status)
    write_experiment_report(exp, eval_output, model_out / "training_summary.json", status)
    print(f"Split experiment complete: {exp}")


def command_run_experiment(args: argparse.Namespace) -> None:
    args = merge_config(args, RUN_EXPERIMENT_DEFAULTS)
    if args.quick:
        args.negative_train_count = min(args.negative_train_count, 400)
        args.background_train_count = min(args.background_train_count, 300)
        args.negative_dev_count = min(args.negative_dev_count, 80)
        args.background_dev_count = min(args.background_dev_count, 80)
        args.positive_dev_count = min(args.positive_dev_count, 100)
        args.positive_test_count = min(args.positive_test_count, 100)
        args.negative_test_seconds = min(args.negative_test_seconds, 120)
        args.background_test_seconds = min(args.background_test_seconds, 120)
        args.steps = min(args.steps, 100)
        args.batch_size = min(args.batch_size, 32)

    exp = Path(args.experiment_dir).resolve()
    exp.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": args.quick, "torch_cuda_available": torch.cuda.is_available()}
    write_json(exp / "experiment_config.json", namespace_config(args))

    model_dir = exp / "models"
    if not args.skip_download:
        run_step(
            "download_models",
            command_download_models,
            argparse.Namespace(output_dir=str(model_dir), models=["all"]),
            status,
        )

    run_step(
        "prepare_manifests",
        command_prepare_manifests,
            argparse.Namespace(
                config=None,
                output_dir=str(exp),
                positive_jsonl=args.positive_jsonl,
                positive_audio_base_path=args.positive_audio_base_path,
                negative_jsonl=args.negative_jsonl,
                negative_audio_base_path=args.negative_audio_base_path,
                background_jsonl=args.background_jsonl,
                background_audio_base_path=args.background_audio_base_path,
                positive_train_count=args.positive_train_count,
                positive_dev_count=args.positive_dev_count,
                positive_test_count=args.positive_test_count,
                negative_train_count=args.negative_train_count,
                negative_dev_count=args.negative_dev_count,
            negative_test_seconds=args.negative_test_seconds,
            background_train_count=args.background_train_count,
            background_dev_count=args.background_dev_count,
            background_test_seconds=args.background_test_seconds,
            clip_seconds=args.clip_seconds,
            seed=args.seed,
        ),
        status,
    )

    manifests = exp / "manifests"
    converted = exp / "audio" / "converted"
    augmented = exp / "audio" / "augmented"
    features = exp / "features"
    model_out = exp / "trained_model"

    conversion_jobs = [
        ("positive_train", "end"),
        ("positive_dev", "end"),
        ("positive_test", "end"),
        ("negative_train", "random"),
        ("negative_dev", "random"),
        ("negative_test", "random"),
        ("background_train", "random"),
        ("background_dev", "random"),
        ("background_test", "random"),
    ]
    converted_manifests: dict[str, Path] = {}
    for name, placement in conversion_jobs:
        out_manifest = converted / f"{name}.jsonl"
        converted_manifests[name] = out_manifest
        run_step(
            f"convert_{name}",
            command_convert_manifest,
            argparse.Namespace(
                manifest=str(manifests / f"{name}.jsonl"),
                output_dir=str(converted / name),
                output_manifest=str(out_manifest),
                limit=None,
                clip_seconds=args.clip_seconds,
                sample_rate=args.sample_rate,
                placement=placement,
                seed=args.seed,
                overwrite=args.overwrite,
                workers=args.convert_workers,
            ),
            status,
        )

    noise_dirs = [str(converted / "background_train")]
    augmented_manifests: dict[str, Path] = {}
    for name, placement in [("positive_train", "end"), ("negative_train", "random")]:
        out_manifest = augmented / f"{name}.jsonl"
        augmented_manifests[name] = out_manifest
        run_step(
            f"augment_{name}",
            command_augment_audio,
            argparse.Namespace(
                input_manifest=str(converted_manifests[name]),
                input_dir=None,
                noise_dir=noise_dirs,
                output_dir=str(augmented / name),
                output_manifest=str(out_manifest),
                rounds=args.augmentation_rounds,
                snr_low=args.snr_low,
                snr_high=args.snr_high,
                artificial_prob=args.artificial_prob,
                random_gain_db=3.0,
                clip_seconds=args.clip_seconds,
                sample_rate=args.sample_rate,
                placement=placement,
                seed=args.seed,
                overwrite=args.overwrite,
                workers=args.augment_workers,
            ),
            status,
        )

    feature_jobs = [
        ("positive_train", [converted_manifests["positive_train"], augmented_manifests["positive_train"]], "end"),
        ("positive_dev", [converted_manifests["positive_dev"]], "end"),
        ("positive_test", [converted_manifests["positive_test"]], "end"),
        ("negative_train", [converted_manifests["negative_train"], augmented_manifests["negative_train"]], "random"),
        ("negative_dev", [converted_manifests["negative_dev"]], "random"),
        ("negative_test", [converted_manifests["negative_test"]], "random"),
        ("background_train", [converted_manifests["background_train"]], "random"),
        ("background_dev", [converted_manifests["background_dev"]], "random"),
        ("background_test", [converted_manifests["background_test"]], "random"),
    ]
    feature_files: dict[str, Path] = {}
    for name, input_manifests, placement in feature_jobs:
        out_file = features / f"{name}.npy"
        feature_files[name] = out_file
        run_step(
            f"features_{name}",
            command_generate_features,
            argparse.Namespace(
                audio_manifest=[str(path) for path in input_manifests],
                audio_dir=[],
                output_file=str(out_file),
                model_dir=str(model_dir),
                batch_size=args.batch_size,
                audio_loader_workers=args.audio_loader_workers,
                prefetch_batches=args.prefetch_batches,
                ncpu=args.ncpu,
                device=args.device,
                limit=None,
                clip_seconds=args.clip_seconds,
                sample_rate=args.sample_rate,
                placement=placement,
                seed=args.seed,
                overwrite=args.overwrite,
            ),
            status,
        )

    run_step(
        "train_model",
        command_train_model,
        argparse.Namespace(
            positive_train_features=str(feature_files["positive_train"]),
            negative_train_features=[str(feature_files["negative_train"]), str(feature_files["background_train"])],
            positive_dev_features=str(feature_files["positive_dev"]),
            negative_dev_features=str(feature_files["negative_dev"]),
            false_positive_features=[str(feature_files["background_dev"])],
            output_dir=str(model_out),
            model_name=args.model_name,
            steps=args.steps,
            batch_positive=args.batch_positive,
            batch_negative=args.batch_negative,
            layer_size=args.layer_size,
            model_type=args.model_type,
            max_negative_weight=args.max_negative_weight,
            target_false_positives_per_hour=args.target_false_positives_per_hour,
            seed=args.seed,
            require_cuda=args.require_cuda,
            overwrite=args.overwrite,
        ),
        status,
    )

    eval_output = exp / "evaluation" / "eval_summary.json"
    eval_config_output = exp / "evaluation" / "evaluation_config.json"
    run_step(
        "evaluate",
        command_evaluate,
        argparse.Namespace(
            evaluation_config=args.evaluation_config,
            model=str(model_out / f"{args.model_name}.onnx"),
            model_dir=str(model_dir),
            positive_manifest=str(converted_manifests["positive_test"]),
            negative_manifest=str(converted_manifests["negative_test"]),
            negative_speech_manifest=None,
            background_manifest=str(converted_manifests["background_test"]),
            output_json=str(eval_output),
            details_jsonl=str(exp / "evaluation" / "eval_details.jsonl"),
            abnormal_jsonl=str(exp / "evaluation" / "eval_abnormal.jsonl"),
            output_config_json=str(eval_config_output),
            threshold=args.threshold,
            debounce_seconds=args.debounce_seconds,
            positive_padding=1,
            negative_padding=0,
            chunk_size=args.eval_chunk_size,
            model_window_seconds=(
                args.model_window_seconds
                if args.model_window_seconds is not None
                else None if args.evaluation_config else args.clip_seconds
            ),
            record_window_scores=args.record_window_scores,
            limit_positive=None,
            limit_negative_seconds=None,
            sample_rate=args.sample_rate,
            overwrite=args.overwrite,
        ),
        status,
    )

    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["outputs"] = {
        "experiment_dir": str(exp),
        "model": str(model_out / f"{args.model_name}.onnx"),
        "training_summary": str(model_out / "training_summary.json"),
        "evaluation": str(eval_output),
        "evaluation_config": str(eval_config_output),
        "evaluation_details": str(exp / "evaluation" / "eval_details.jsonl"),
        "evaluation_abnormal": str(exp / "evaluation" / "eval_abnormal.jsonl"),
    }
    write_json(exp / "experiment_status.json", status)
    write_experiment_report(exp, eval_output, model_out / "training_summary.json", status)
    print(f"Experiment complete: {exp}")


def write_experiment_report(exp: Path, eval_json: Path, training_json: Path, status: dict[str, Any]) -> None:
    evaluation = json.loads(eval_json.read_text(encoding="utf-8")) if eval_json.exists() else {}
    training = json.loads(training_json.read_text(encoding="utf-8")) if training_json.exists() else {}
    lines = [
        "# openWakeWord Training Experiment",
        "",
        f"- Experiment directory: `{exp}`",
        f"- CUDA visible to PyTorch: `{training.get('torch_cuda_available', status.get('torch_cuda_available'))}`",
        f"- Training device: `{training.get('torch_device', 'unknown')}`",
        f"- Trained model: `{training.get('onnx_model', '')}`",
        f"- Training steps: `{training.get('steps', '')}`",
        "",
        "## Evaluation",
        "",
        f"- Evaluation config: `{evaluation.get('evaluation_config', '')}`",
        f"- Per-case details: `{evaluation.get('detail_jsonl', '')}`",
        f"- Abnormal cases: `{evaluation.get('abnormal_jsonl', '')}`",
        "",
    ]
    for name, metrics in evaluation.get("sets", {}).items():
        lines.append(f"### {name}")
        for key, value in metrics.items():
            if key == "subsets":
                continue
            if key == "errors" and not value:
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    if status.get("source") == "split_manifests":
        data_notes = [
            "- Training started from pre-split JSONL manifests; no prepare-manifests stage was run.",
            "- Feature extraction loaded JSONL audio paths directly and created fixed-length embeddings in memory.",
            "- Positive and negative train manifests were augmented with noise sampled only from the background train manifest.",
            "- Background clips were also included as negative training and false-positive validation data.",
        ]
    else:
        data_notes = [
            "- Positive train clips were converted to fixed 16 kHz WAV and augmented once with sampled background SNR.",
            "- Negative train clips were sampled from the configured JSONL source(s), converted to WAV, and augmented once.",
            "- Background clips were also included as negative training and false-positive validation data.",
        ]

    lines.extend(
        [
            "## Conclusion",
            "",
            experiment_conclusion(evaluation),
            "",
            "## Notes",
            "",
            *data_notes,
            "- Evaluation is a sliding-window test. Each per-case detail row records score windows, threshold crossings, events, and abnormal FA/FR cases.",
            "- False accept metrics are event counts over the evaluated negative duration with debounce applied.",
        ]
    )
    (exp / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def experiment_conclusion(evaluation: dict[str, Any]) -> str:
    positive = evaluation.get("sets", {}).get("positive", {})
    negative = evaluation.get("sets", {}).get("negative", evaluation.get("sets", {}).get("negative_speech", {}))
    background = evaluation.get("sets", {}).get("background", {})
    false_reject_rate = positive.get("false_reject_rate")
    negative_fa = negative.get("false_accepts_per_hour")
    background_fa = background.get("false_accepts_per_hour")
    if false_reject_rate is None:
        return "Evaluation did not include a positive test set, so wake-word recall was not measured."
    if false_reject_rate > 0.2:
        return (
            "This run verifies that the pipeline executes end to end, but the trained model is not ready to use. "
            f"The positive false-reject rate was {false_reject_rate:.3f}; false accepts/hour were "
            f"{negative_fa} on negative audio and {background_fa} on background audio."
        )
    return (
        "This run produced a usable candidate for further threshold tuning. "
        f"The positive false-reject rate was {false_reject_rate:.3f}; false accepts/hour were "
        f"{negative_fa} on negative audio and {background_fa} on background audio."
    )


def namespace_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key != "func" and not callable(value)}


def add_common_args(parser: argparse.ArgumentParser, default_none: bool = False) -> None:
    parser.add_argument("--sample-rate", type=int, default=None if default_none else DEFAULT_SR)
    parser.add_argument("--clip-seconds", type=float, default=None if default_none else 2.0)
    parser.add_argument("--seed", type=int, default=None if default_none else 1337)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="openWakeWord local training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # These are dispatched by ``main`` to wuw_training.cli before this legacy
    # parser is evaluated. Keeping their signatures here makes them visible in
    # the top-level --help output alongside the compatible low-level commands.
    p = sub.add_parser("run", help="Run the modular INI pipeline in [steps] order")
    p.add_argument("--config", required=True, help="INI configuration file")
    p.add_argument("--from", dest="from_step")
    p.add_argument("--to", dest="to_step")
    p.add_argument("--force", action="append")

    p = sub.add_parser("run-step", help="Run one named modular INI pipeline stage")
    p.add_argument("--config", required=True, help="INI configuration file")
    p.add_argument("step")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("validate", help="Validate a modular INI pipeline without running it")
    p.add_argument("--config", required=True, help="INI configuration file")

    p = sub.add_parser("status", help="Show modular INI pipeline checkpoint status")
    p.add_argument("--config", required=True, help="INI configuration file")

    p = sub.add_parser("download-models", help="Download/verify openWakeWord model resources")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models", nargs="+", default=["all"], help="Use 'all' or model name substrings")
    p.set_defaults(func=command_download_models)

    p = sub.add_parser("index-audio", help="Create a simple JSONL manifest by scanning an audio directory")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--source", default="audio")
    p.add_argument("--label", type=int, choices=[0, 1], default=0)
    p.add_argument("--limit", type=int)
    p.set_defaults(relative_paths=True)
    p.add_argument("--absolute-paths", dest="relative_paths", action="store_false")
    p.set_defaults(func=command_index_audio)

    p = sub.add_parser("prepare-manifests", help="Create train/dev/test manifests for this experiment")
    p.add_argument("--output-dir")
    p.add_argument("--config", help="JSON config file; JSONL dataset specs can be strings, objects, or lists")
    p.add_argument("--positive-jsonl")
    p.add_argument("--positive-audio-base-path")
    p.add_argument("--negative-jsonl")
    p.add_argument("--negative-audio-base-path")
    p.add_argument("--background-jsonl")
    p.add_argument("--background-audio-base-path")
    p.add_argument("--positive-train-count", type=int)
    p.add_argument("--positive-dev-count", type=int)
    p.add_argument("--positive-test-count", type=int)
    p.add_argument("--negative-train-count", type=int)
    p.add_argument("--negative-dev-count", type=int)
    p.add_argument("--negative-test-seconds", type=float)
    p.add_argument("--background-train-count", type=int)
    p.add_argument("--background-dev-count", type=int)
    p.add_argument("--background-test-seconds", type=float)
    add_common_args(p, default_none=True)
    p.set_defaults(func=command_prepare_manifests)

    p = sub.add_parser("convert-manifest", help="Convert a manifest to fixed-length 16 kHz WAV files")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-manifest")
    p.add_argument("--limit", type=int)
    p.add_argument("--placement", choices=["start", "end", "center", "random"], default="random")
    p.add_argument("--workers", type=int, default=DEFAULT_IO_WORKERS)
    p.add_argument("--overwrite", action="store_true")
    add_common_args(p)
    p.set_defaults(func=command_convert_manifest)

    p = sub.add_parser("augment-audio", help="Mix input clips with background and optional artificial noise")
    p.add_argument("--input-manifest")
    p.add_argument("--input-dir")
    p.add_argument("--noise-manifest", nargs="*", default=[])
    p.add_argument("--noise-dir", nargs="*", default=[])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-manifest")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--snr-low", type=float, default=-5.0)
    p.add_argument("--snr-high", type=float, default=15.0)
    p.add_argument("--artificial-prob", type=float, default=0.15)
    p.add_argument("--random-gain-db", type=float, default=3.0)
    p.add_argument("--placement", choices=["start", "end", "center", "random"], default="random")
    p.add_argument("--workers", type=int, default=DEFAULT_IO_WORKERS)
    p.add_argument("--overwrite", action="store_true")
    add_common_args(p)
    p.set_defaults(func=command_augment_audio)

    p = sub.add_parser("generate-features", help="Generate openWakeWord .npy features from WAV clips")
    p.add_argument("--audio-manifest", nargs="*", default=[])
    p.add_argument("--audio-dir", nargs="*", default=[])
    p.add_argument("--output-file", required=True)
    p.add_argument("--model-dir")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--audio-loader-workers", type=int, default=1)
    p.add_argument("--prefetch-batches", type=int, default=1)
    p.add_argument("--ncpu", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--limit", type=int)
    p.add_argument("--placement", choices=["start", "end", "center", "random"], default="random")
    p.add_argument("--overwrite", action="store_true")
    add_common_args(p)
    p.set_defaults(func=command_generate_features)

    p = sub.add_parser("train-model", help="Train a binary openWakeWord model from feature files")
    p.add_argument("--positive-train-features", required=True)
    p.add_argument("--negative-train-features", nargs="+", required=True)
    p.add_argument("--positive-dev-features", required=True)
    p.add_argument("--negative-dev-features", nargs="+", required=True)
    p.add_argument("--false-positive-features", nargs="*")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="turn_on_the_office_lights")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-positive", type=int, default=64)
    p.add_argument("--batch-negative", type=int, default=256)
    p.add_argument(
        "--batch-counts-json",
        help="JSON object, JSON file path, or key=value list overriding samples per mini-batch group.",
    )
    p.add_argument("--layer-size", type=int, default=64)
    p.add_argument("--model-type", choices=["dnn", "rnn", "cnn", "attention"], default="dnn")
    p.add_argument("--max-negative-weight", type=float, default=500.0)
    p.add_argument("--target-false-positives-per-hour", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=command_train_model)

    p = sub.add_parser("evaluate", help="Evaluate false rejects and false accepts for a trained ONNX model")
    p.add_argument("--evaluation-config", help="JSON config with evaluation inputs, thresholds, and output paths")
    p.add_argument("--model")
    p.add_argument("--model-dir")
    p.add_argument("--positive-manifest")
    p.add_argument("--negative-manifest")
    p.add_argument("--negative-speech-manifest")
    p.add_argument("--background-manifest")
    p.add_argument("--output-json")
    p.add_argument("--details-jsonl")
    p.add_argument("--abnormal-jsonl")
    p.add_argument("--output-config-json")
    p.add_argument("--threshold", type=float)
    p.add_argument("--debounce-seconds", type=float)
    p.add_argument("--positive-padding", type=int)
    p.add_argument("--negative-padding", type=int)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--model-window-seconds", type=float)
    p.add_argument("--limit-positive", type=int)
    p.add_argument("--limit-negative-seconds", type=float)
    p.add_argument("--sample-rate", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(record_window_scores=None)
    p.add_argument("--record-window-scores", dest="record_window_scores", action="store_true")
    p.add_argument("--no-record-window-scores", dest="record_window_scores", action="store_false")
    p.set_defaults(func=command_evaluate)

    p = sub.add_parser("run-experiment", help="Run the end-to-end experiment")
    p.add_argument("--config", help="JSON config file for data, training, and evaluation settings")
    p.add_argument("--experiment-dir")
    p.add_argument("--positive-jsonl")
    p.add_argument("--positive-audio-base-path")
    p.add_argument("--negative-jsonl")
    p.add_argument("--negative-audio-base-path")
    p.add_argument("--background-jsonl")
    p.add_argument("--background-audio-base-path")
    p.add_argument("--model-name")
    p.add_argument("--positive-train-count", type=int)
    p.add_argument("--positive-dev-count", type=int)
    p.add_argument("--positive-test-count", type=int)
    p.add_argument("--negative-train-count", type=int)
    p.add_argument("--negative-dev-count", type=int)
    p.add_argument("--negative-test-seconds", type=float)
    p.add_argument("--background-train-count", type=int)
    p.add_argument("--background-dev-count", type=int)
    p.add_argument("--background-test-seconds", type=float)
    p.add_argument("--snr-low", type=float)
    p.add_argument("--snr-high", type=float)
    p.add_argument("--artificial-prob", type=float)
    p.add_argument("--augmentation-rounds", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--audio-loader-workers", type=int)
    p.add_argument("--prefetch-batches", type=int)
    p.add_argument("--batch-positive", type=int)
    p.add_argument("--batch-negative", type=int)
    p.add_argument("--convert-workers", type=int)
    p.add_argument("--augment-workers", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--layer-size", type=int)
    p.add_argument("--model-type", choices=["dnn", "rnn", "cnn", "attention"])
    p.add_argument("--max-negative-weight", type=float)
    p.add_argument("--target-false-positives-per-hour", type=float)
    p.add_argument("--evaluation-config", help="JSON config for the evaluation stage")
    p.add_argument("--threshold", type=float)
    p.add_argument("--debounce-seconds", type=float)
    p.add_argument("--eval-chunk-size", type=int)
    p.add_argument("--model-window-seconds", type=float)
    p.set_defaults(record_window_scores=None)
    p.add_argument("--record-window-scores", dest="record_window_scores", action="store_true")
    p.add_argument("--no-record-window-scores", dest="record_window_scores", action="store_false")
    p.add_argument("--ncpu", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "gpu"])
    p.add_argument("--require-cuda", action="store_true", default=None)
    p.add_argument("--skip-download", action="store_true", default=None)
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument("--quick", action="store_true", default=None, help="Small smoke-test experiment")
    add_common_args(p, default_none=True)
    p.set_defaults(func=command_run_experiment)

    p = sub.add_parser("run-from-splits", help="Run augmentation, feature extraction, training, and evaluation from split JSONL manifests")
    p.add_argument("--config", required=True, help="JSON config containing split_manifests")
    p.add_argument("--experiment-dir")
    p.add_argument("--model-name")
    p.add_argument("--steps", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "gpu"])
    p.add_argument("--require-cuda", action="store_true", default=None)
    p.add_argument("--skip-download", action="store_true", default=None)
    p.add_argument("--overwrite", action="store_true", default=None)
    p.set_defaults(func=command_run_from_splits)

    return parser


def main() -> None:
    # The INI runner is intentionally dispatched before constructing the legacy
    # argparse tree.  This keeps all existing low-level commands compatible
    # while making `run`, `run-step`, `validate`, and `status` the new public
    # orchestration interface.
    if len(sys.argv) > 1 and sys.argv[1] in {"run", "run-step", "validate", "status"}:
        from wuw_training.cli import main as ini_main

        raise SystemExit(ini_main(sys.argv[1:]))
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"run-experiment", "run-from-splits"}:
        warnings.warn(
            f"{args.command} uses the legacy JSON end-to-end configuration. "
            "Use `run --config config.ini` for new experiments.",
            DeprecationWarning,
            stacklevel=2,
        )
    args.func(args)


if __name__ == "__main__":
    main()
