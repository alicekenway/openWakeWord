"""Building blocks for the optional two-stage WeNet CTC + WAC pipeline.

The normal openWakeWord pipeline remains unchanged.  This module is used only
when a feature block declares ``extractor = wenet_ctc_wac`` and a training or
testing block declares ``structure = ctc_wac``.

The important boundary is deliberate: stage 1 is a frozen external CTC ONNX
model.  It creates encoder features and CTC keyword scores.  Stage 2 is the
small WAC-like classifier trained by this repository.  There is no gradient
path from stage 2 back into stage 1.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap

from .artifacts import hash_payload, read_json, read_jsonl, write_json, write_jsonl
from .config import ConfigurationError


BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Keyword:
    """One wake-word hypothesis accepted by the stage-1 CTC scorer."""

    id: str
    display_text: str
    token_ids: tuple[int, ...]
    threshold: float


@dataclass(frozen=True)
class CacheSpec:
    """One recurrent/cache tensor needed by a streaming ONNX graph."""

    input_name: str
    output_name: str
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class AttentionMaskSpec:
    """WeNet's mask for the empty fixed cache used by the first chunk."""

    input_name: str
    cache_frames: int
    chunk_frames: int


@dataclass(frozen=True)
class Stage1Contract:
    """The small explicit interface between an exported WeNet model and WAC.

    Different WeNet exports use different ONNX tensor names and cache layouts.
    Putting those names in JSON keeps this pipeline token-model agnostic: a
    BPE CTC model and a phoneme CTC model use the same Python code.
    """

    sample_rate: int
    num_mel_bins: int
    frame_length_ms: float
    frame_shift_ms: float
    dither: float
    chunk_frames: int
    chunk_stride_frames: int
    minimum_input_frames: int
    pad_final_chunk: bool
    feature_input: str
    feature_length_input: str | None
    offset_input: str | None
    encoder_output: str
    ctc_output: str
    ctc_output_is_log_probs: bool
    blank_id: int
    input_layout: str
    initial_offset: int
    cache_specs: tuple[CacheSpec, ...]
    attention_mask: AttentionMaskSpec | None
    constant_inputs: dict[str, Any]
    feature_mean: tuple[float, ...] | None
    feature_istd: tuple[float, ...] | None

    @classmethod
    def from_json(cls, path: Path) -> "Stage1Contract":
        try:
            raw = read_json(path)
        except Exception as exc:
            raise ConfigurationError(f"Could not read stage-1 contract {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Stage-1 contract must be a JSON object: {path}")

        def string(mapping: dict[str, Any], key: str, *, required: bool = True) -> str | None:
            value = mapping.get(key)
            if value is None and not required:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"Stage-1 contract {path}: {key} must be a non-empty string")
            return value

        fbank = raw.get("fbank", {})
        if not isinstance(fbank, dict):
            raise ConfigurationError(f"Stage-1 contract {path}: fbank must be an object")
        inputs = raw.get("inputs", {})
        outputs = raw.get("outputs", {})
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ConfigurationError(f"Stage-1 contract {path}: inputs and outputs must be objects")
        try:
            sample_rate = int(raw.get("sample_rate", 16000))
            num_mel_bins = int(fbank.get("num_mel_bins", 80))
            frame_length_ms = float(fbank.get("frame_length_ms", 25.0))
            frame_shift_ms = float(fbank.get("frame_shift_ms", 10.0))
            dither = float(fbank.get("dither", 0.0))
            chunk_frames = int(raw["chunk_frames"])
            chunk_stride_frames = int(raw.get("chunk_stride_frames", chunk_frames))
            minimum_input_frames = int(raw.get("minimum_input_frames", 1))
            blank_id = int(raw.get("blank_id", 0))
            initial_offset = int(raw.get("initial_offset", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Stage-1 contract {path} needs numeric sample_rate/fbank/chunk_frames values"
            ) from exc
        if (
            sample_rate < 1
            or num_mel_bins < 1
            or frame_length_ms <= 0
            or frame_shift_ms <= 0
            or chunk_frames < 1
            or chunk_stride_frames < 1
            or minimum_input_frames < 1
            or minimum_input_frames > chunk_frames
            or initial_offset < 0
        ):
            raise ConfigurationError(f"Stage-1 contract {path} contains invalid frontend or chunk values")

        feature_input = string(inputs, "features")
        assert feature_input is not None
        feature_length_input = string(inputs, "features_length", required=False)
        offset_input = string(inputs, "offset", required=False)
        encoder_output = string(outputs, "encoder")
        ctc_output = outputs.get("ctc_log_probs", outputs.get("ctc"))
        if not isinstance(ctc_output, str) or not ctc_output.strip():
            raise ConfigurationError(
                f"Stage-1 contract {path}: outputs must include ctc_log_probs (or the legacy name ctc)"
            )
        assert encoder_output is not None
        layout = str(raw.get("input_layout", "BTF")).upper()
        if layout != "BTF":
            raise ConfigurationError(
                f"Stage-1 contract {path}: only input_layout = BTF is supported; got {layout!r}"
            )
        log_probs = raw.get("ctc_output_is_log_probs", False)
        if not isinstance(log_probs, bool):
            raise ConfigurationError(f"Stage-1 contract {path}: ctc_output_is_log_probs must be true or false")
        pad_final_chunk = raw.get("pad_final_chunk", True)
        if not isinstance(pad_final_chunk, bool):
            raise ConfigurationError(f"Stage-1 contract {path}: pad_final_chunk must be true or false")

        raw_caches = raw.get("cache_inputs", raw.get("caches", []))
        if not isinstance(raw_caches, list):
            raise ConfigurationError(f"Stage-1 contract {path}: cache_inputs must be a JSON list")
        caches: list[CacheSpec] = []
        for index, value in enumerate(raw_caches):
            if not isinstance(value, dict):
                raise ConfigurationError(f"Stage-1 contract {path}: cache_inputs[{index}] must be an object")
            input_name = string(value, "input")
            output_name = string(value, "output")
            shape = value.get("shape")
            if not isinstance(shape, list) or not shape:
                raise ConfigurationError(f"Stage-1 contract {path}: cache_inputs[{index}].shape must be a non-empty list")
            try:
                parsed_shape = tuple(int(item) for item in shape)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Stage-1 contract {path}: cache_inputs[{index}].shape must contain integers"
                ) from exc
            if any(item < 0 for item in parsed_shape):
                raise ConfigurationError(f"Stage-1 contract {path}: cache tensor dimensions cannot be negative")
            dtype = str(value.get("dtype", "float32"))
            try:
                np.dtype(dtype)
            except TypeError as exc:
                raise ConfigurationError(
                    f"Stage-1 contract {path}: unsupported cache dtype {dtype!r}"
                ) from exc
            assert input_name is not None and output_name is not None
            caches.append(CacheSpec(input_name, output_name, parsed_shape, dtype))

        raw_attention_mask = raw.get("attention_mask")
        attention_mask: AttentionMaskSpec | None = None
        if raw_attention_mask is not None:
            if not isinstance(raw_attention_mask, dict):
                raise ConfigurationError(f"Stage-1 contract {path}: attention_mask must be an object")
            mask_input = string(raw_attention_mask, "input")
            try:
                mask_cache_frames = int(raw_attention_mask["cache_frames"])
                mask_chunk_frames = int(raw_attention_mask["chunk_frames"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Stage-1 contract {path}: attention_mask needs integer cache_frames and chunk_frames"
                ) from exc
            if mask_cache_frames < 0 or mask_chunk_frames < 1:
                raise ConfigurationError(f"Stage-1 contract {path}: invalid attention_mask dimensions")
            assert mask_input is not None
            attention_mask = AttentionMaskSpec(mask_input, mask_cache_frames, mask_chunk_frames)

        constants = raw.get("constant_inputs", {})
        if not isinstance(constants, dict):
            raise ConfigurationError(f"Stage-1 contract {path}: constant_inputs must be an object")
        raw_mean = fbank.get("mean")
        raw_istd = fbank.get("istd")
        if (raw_mean is None) != (raw_istd is None):
            raise ConfigurationError(f"Stage-1 contract {path}: fbank.mean and fbank.istd must be supplied together")
        mean: tuple[float, ...] | None = None
        istd: tuple[float, ...] | None = None
        if raw_mean is not None:
            if not isinstance(raw_mean, list) or not isinstance(raw_istd, list):
                raise ConfigurationError(f"Stage-1 contract {path}: fbank.mean and fbank.istd must be lists")
            try:
                mean = tuple(float(item) for item in raw_mean)
                istd = tuple(float(item) for item in raw_istd)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Stage-1 contract {path}: invalid fbank CMVN values") from exc
            if len(mean) != num_mel_bins or len(istd) != num_mel_bins:
                raise ConfigurationError(
                    f"Stage-1 contract {path}: fbank.mean/istd must each have {num_mel_bins} values"
                )
        return cls(
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms,
            dither=dither,
            chunk_frames=chunk_frames,
            chunk_stride_frames=chunk_stride_frames,
            minimum_input_frames=minimum_input_frames,
            pad_final_chunk=pad_final_chunk,
            feature_input=feature_input,
            feature_length_input=feature_length_input,
            offset_input=offset_input,
            encoder_output=encoder_output,
            ctc_output=ctc_output,
            ctc_output_is_log_probs=log_probs,
            blank_id=blank_id,
            input_layout=layout,
            initial_offset=initial_offset,
            cache_specs=tuple(caches),
            attention_mask=attention_mask,
            constant_inputs=dict(constants),
            feature_mean=mean,
            feature_istd=istd,
        )

    def fingerprint(self) -> str:
        return hash_payload(asdict(self))


def load_keywords(path: Path, *, require_threshold: bool = True) -> list[Keyword]:
    """Load wake-word token IDs and, when needed, manual stage-1 gates.

    Feature extraction needs only ids and token IDs.  Training/evaluation also
    need thresholds.  Supporting a token-only file lets a threshold change
    retrain WAC without needlessly regenerating frozen CTC features.
    """

    try:
        raw = read_json(path)
    except Exception as exc:
        raise ConfigurationError(f"Could not read keyword config {path}: {exc}") from exc
    values = raw.get("keywords") if isinstance(raw, dict) else None
    if not isinstance(values, list) or not values:
        raise ConfigurationError(f"Keyword config {path} must contain a non-empty 'keywords' list")
    parsed: list[Keyword] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ConfigurationError(f"Keyword config {path}: keywords[{index}] must be an object")
        key_id = value.get("id")
        if not isinstance(key_id, str) or not key_id.strip():
            raise ConfigurationError(f"Keyword config {path}: keywords[{index}].id must be a non-empty string")
        if key_id in seen:
            raise ConfigurationError(f"Keyword config {path}: duplicate keyword id {key_id!r}")
        seen.add(key_id)
        display_text = value.get("display_text", value.get("text", key_id))
        if not isinstance(display_text, str):
            raise ConfigurationError(f"Keyword config {path}: keywords[{index}].display_text must be a string")
        token_values = value.get("token_ids")
        if not isinstance(token_values, list) or not token_values:
            raise ConfigurationError(f"Keyword config {path}: keywords[{index}].token_ids must be a non-empty list")
        try:
            token_ids = tuple(int(item) for item in token_values)
            threshold = float(value["threshold"]) if "threshold" in value else 0.0
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Keyword config {path}: keywords[{index}] needs integer token_ids and numeric threshold"
            ) from exc
        if require_threshold and "threshold" not in value:
            raise ConfigurationError(f"Keyword config {path}: keywords[{index}].threshold is required")
        if any(item < 0 for item in token_ids) or not math.isfinite(threshold):
            raise ConfigurationError(f"Keyword config {path}: invalid tokens or threshold at keywords[{index}]")
        parsed.append(Keyword(key_id, display_text, token_ids, threshold))
    return parsed


def keyword_fingerprint(keywords: Sequence[Keyword]) -> str:
    return hash_payload([asdict(item) for item in keywords])


def keyword_token_fingerprint(keywords: Sequence[Keyword]) -> str:
    """Fingerprint the model-facing parts only; deliberately omit thresholds."""

    return hash_payload(
        [
            {
                "id": item.id,
                "display_text": item.display_text,
                "token_ids": item.token_ids,
            }
            for item in keywords
        ]
    )


@dataclass(frozen=True)
class FeatureBundlePaths:
    features: Path
    all_scores: Path
    top_score: Path
    margin: Path
    winner_onehot: Path
    rows: Path
    summary: Path

    def all(self) -> list[Path]:
        return [
            self.features,
            self.all_scores,
            self.top_score,
            self.margin,
            self.winner_onehot,
            self.rows,
            self.summary,
        ]


def feature_bundle_paths(features: Path) -> FeatureBundlePaths:
    """Return stable paths for a stage-1 feature bundle.

    The principal NPY file retains the normal feature-stage ``output_file``
    name so train blocks can still reference it in the same way as the old
    pipeline.  The extra stage-1 values live next to it.
    """

    stem = features.stem
    parent = features.parent
    return FeatureBundlePaths(
        features=features,
        all_scores=parent / f"{stem}.all_scores.npy",
        top_score=parent / f"{stem}.top_score.npy",
        margin=parent / f"{stem}.margin.npy",
        winner_onehot=parent / f"{stem}.winner_onehot.npy",
        rows=parent / f"{stem}.rows.jsonl",
        summary=parent / f"{stem}.summary.json",
    )


def feature_bundle_valid(features: Path, *, require_complete: bool = True) -> bool:
    """Check the shapes and summary needed by CTC-WAC training.

    It is intentionally inexpensive: it reads NPY headers through memory maps
    instead of pulling the feature tensors into RAM.
    """

    paths = feature_bundle_paths(features)
    if not all(path.is_file() for path in paths.all()):
        return False
    try:
        summary = read_json(paths.summary)
        x = np.load(paths.features, mmap_mode="r")
        scores = np.load(paths.all_scores, mmap_mode="r")
        top = np.load(paths.top_score, mmap_mode="r")
        margin = np.load(paths.margin, mmap_mode="r")
        winner = np.load(paths.winner_onehot, mmap_mode="r")
        n = int(x.shape[0])
        valid = (
            int(summary.get("bundle_schema", -1)) == BUNDLE_SCHEMA_VERSION
            and (not require_complete or int(summary.get("error_count", -1)) == 0)
            and int(summary.get("feature_count", -1)) == n
            and x.ndim == 3
            and scores.ndim == 2
            and top.shape == (n, 1)
            and margin.shape == (n, 1)
            and winner.shape == scores.shape
            and scores.shape[0] == n
            and scores.shape[1] >= 1
        )
        return bool(valid)
    except Exception:
        return False


def _log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(values, axis=-1, keepdims=True)
    return values - maximum - np.log(np.sum(np.exp(values - maximum), axis=-1, keepdims=True))


def ctc_keyword_score_trace(log_probs: np.ndarray, token_ids: Sequence[int], *, blank_id: int = 0) -> np.ndarray:
    """Return the best length-normalized CTC alignment score at each frame.

    This is a Viterbi-style CTC keyword scorer.  A fresh alignment may start at
    any frame, which is what makes it suitable for a wake-word trigger rather
    than for scoring an entire utterance.  ``token_ids`` should not include the
    CTC blank token.
    """

    values = np.asarray(log_probs, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"CTC scores must have shape [frames, vocabulary], got {values.shape}")
    tokens = tuple(int(item) for item in token_ids)
    if not tokens:
        raise ValueError("A CTC keyword needs at least one token")
    if blank_id < 0 or blank_id >= values.shape[1] or any(item < 0 or item >= values.shape[1] for item in tokens):
        raise ValueError("CTC keyword token ID is outside the output vocabulary")
    if blank_id in tokens:
        raise ValueError("CTC keyword token IDs must not include the blank ID")

    # [blank, token_0, blank, token_1, ..., blank]
    extended: list[int] = [blank_id]
    for token in tokens:
        extended.extend([token, blank_id])
    state_count = len(extended)
    negative_infinity = np.float32(-1.0e30)
    previous = np.full(state_count, negative_infinity, dtype=np.float32)
    trace = np.full(values.shape[0], negative_infinity, dtype=np.float32)

    for frame_index, frame in enumerate(values):
        current = np.full(state_count, negative_infinity, dtype=np.float32)
        # Restarting here removes the score of arbitrary speech before a
        # possible keyword.  Keeping previous[0] handles blank stretches.
        current[0] = np.float32(frame[blank_id])
        for state in range(1, state_count):
            symbol = extended[state]
            candidates = [previous[state], previous[state - 1]]
            if state == 1:
                candidates.append(np.float32(0.0))  # begin the keyword now
            elif state % 2 == 1 and extended[state] != extended[state - 2]:
                candidates.append(previous[state - 2])
            current[state] = np.max(np.asarray(candidates, dtype=np.float32)) + frame[symbol]
        previous = current
        trace[frame_index] = max(current[-1], current[-2]) / np.float32(len(tokens))
    return trace


def ctc_keyword_score_traces(
    log_probs: np.ndarray,
    keywords: Sequence[Keyword],
    *,
    blank_id: int = 0,
) -> np.ndarray:
    """Return ``[frames, keyword_count]`` normalized CTC scores."""

    if not keywords:
        raise ValueError("At least one keyword is required")
    return np.stack(
        [ctc_keyword_score_trace(log_probs, item.token_ids, blank_id=blank_id) for item in keywords],
        axis=1,
    ).astype(np.float32, copy=False)


def rank_keyword_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top score, top-minus-second margin, and winning keyword index."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"Keyword scores must have shape [rows, keywords], got {values.shape}")
    winner = np.argmax(values, axis=1).astype(np.int64)
    top = values[np.arange(values.shape[0]), winner]
    if values.shape[1] == 1:
        margin = np.zeros_like(top)
    else:
        second = np.partition(values, -2, axis=1)[:, -2]
        margin = top - second
    return top.astype(np.float32), margin.astype(np.float32), winner


def winner_onehot(winner: np.ndarray, keyword_count: int) -> np.ndarray:
    result = np.zeros((int(winner.shape[0]), int(keyword_count)), dtype=np.float32)
    result[np.arange(result.shape[0]), winner.astype(np.int64)] = 1.0
    return result


def stage1_gate(scores: np.ndarray, keywords: Sequence[Keyword]) -> tuple[np.ndarray, np.ndarray]:
    """Apply the per-keyword manual gate only when a consumer asks for it."""

    top, _margin, winner = rank_keyword_scores(scores)
    thresholds = np.asarray([item.threshold for item in keywords], dtype=np.float32)
    if scores.shape[1] != thresholds.shape[0]:
        raise ValueError("Score columns do not match keyword thresholds")
    return top >= thresholds[winner], winner


def retained_stage1_indices(
    scores: np.ndarray,
    keywords: Sequence[Keyword],
    *,
    chunk_rows: int = 65536,
) -> np.ndarray:
    """Return gated row IDs without materializing a large score memory-map.

    A background corpus can contain millions of fixed windows.  The artifact
    stays on disk, so filtering must also stay bounded in memory.
    """

    if scores.ndim != 2 or scores.shape[1] != len(keywords):
        raise ValueError("Score columns do not match the keyword configuration")
    thresholds = np.asarray([item.threshold for item in keywords], dtype=np.float32)
    kept: list[np.ndarray] = []
    for start in range(0, int(scores.shape[0]), max(1, int(chunk_rows))):
        values = np.asarray(scores[start:start + chunk_rows], dtype=np.float32)
        winner = np.argmax(values, axis=1)
        top = values[np.arange(values.shape[0]), winner]
        local = np.flatnonzero(top >= thresholds[winner]).astype(np.int64)
        if local.size:
            kept.append(local + start)
    return np.concatenate(kept) if kept else np.empty((0,), dtype=np.int64)


def _onnx_numpy_dtype(onnx_type: str) -> np.dtype[Any]:
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int16)": np.dtype(np.int16),
        "tensor(int8)": np.dtype(np.int8),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(bool)": np.dtype(np.bool_),
    }
    if onnx_type not in mapping:
        raise ConfigurationError(f"Unsupported ONNX input type {onnx_type!r}")
    return mapping[onnx_type]


class StreamingCtcStage1:
    """Run a contracted ONNX CTC model chunk by chunk, including its caches."""

    def __init__(self, model_path: Path, contract: Stage1Contract, *, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for the wenet_ctc_wac feature extractor") from exc
        if not model_path.is_file():
            raise FileNotFoundError(f"Stage-1 ONNX model does not exist: {model_path}")
        requested = device.lower()
        if requested not in {"cpu", "gpu", "auto"}:
            raise ConfigurationError("Stage-1 device must be cpu, gpu, or auto")
        available = ort.get_available_providers()
        if requested == "gpu" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("Stage-1 device=gpu was requested but ONNX Runtime has no CUDAExecutionProvider")
        providers = ["CPUExecutionProvider"]
        if requested == "gpu" or (requested == "auto" and "CUDAExecutionProvider" in available):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.model_path = model_path
        self.contract = contract
        self._inputs = {item.name: item for item in self.session.get_inputs()}
        self._outputs = {item.name: item for item in self.session.get_outputs()}
        required_inputs = {
            contract.feature_input,
            *(item.input_name for item in contract.cache_specs),
            *([contract.feature_length_input] if contract.feature_length_input else []),
            *([contract.offset_input] if contract.offset_input else []),
            *([contract.attention_mask.input_name] if contract.attention_mask else []),
            *contract.constant_inputs.keys(),
        }
        missing_inputs = sorted(item for item in required_inputs if item not in self._inputs)
        if missing_inputs:
            raise ConfigurationError(
                f"Stage-1 ONNX model is missing contract input(s): {', '.join(missing_inputs)}"
            )
        required_outputs = {
            contract.encoder_output,
            contract.ctc_output,
            *(item.output_name for item in contract.cache_specs),
        }
        missing_outputs = sorted(item for item in required_outputs if item not in self._outputs)
        if missing_outputs:
            raise ConfigurationError(
                f"Stage-1 ONNX model is missing contract output(s): {', '.join(missing_outputs)}"
            )
        feature_type = self._inputs[contract.feature_input].type
        if feature_type not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
            raise ConfigurationError(
                f"Stage-1 feature input {contract.feature_input!r} is {feature_type}; "
                "use a QDQ/int8 model that still exposes floating fbank input/output tensors"
            )
        for name in (contract.encoder_output, contract.ctc_output):
            tensor_type = self._outputs[name].type
            if tensor_type not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
                raise ConfigurationError(
                    f"Stage-1 output {name!r} is {tensor_type}; expose a dequantized floating output before using int8 ONNX"
                )
        self.reset()

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def _array_for_input(self, name: str, value: Any) -> np.ndarray:
        dtype = _onnx_numpy_dtype(self._inputs[name].type)
        return np.asarray(value, dtype=dtype)

    def _scalar_for_input(self, name: str, value: int) -> np.ndarray:
        shape = self._inputs[name].shape
        raw: Any = value if len(shape) == 0 else [value]
        return self._array_for_input(name, raw)

    def reset(self) -> None:
        self._offset = self.contract.initial_offset
        self._chunks_run = 0
        self._caches = {
            spec.input_name: np.zeros(spec.shape, dtype=np.dtype(spec.dtype)) for spec in self.contract.cache_specs
        }

    def _streaming_attention_mask(self) -> np.ndarray:
        spec = self.contract.attention_mask
        if spec is None:
            raise RuntimeError("The stage-1 contract does not define an attention mask")
        mask = np.ones((1, 1, spec.cache_frames + spec.chunk_frames), dtype=np.bool_)
        # We allocate the full fixed-size cache before inference.  Its zeros
        # are not real history on the first chunk, so WeNet masks them.  After
        # the first call, forward_chunk has populated the cache and WeNet's
        # own ONNX simulation uses an all-true mask.
        if self._chunks_run == 0:
            mask[:, :, :spec.cache_frames] = False
        return mask

    @staticmethod
    def _remove_batch_dimension(values: np.ndarray, *, name: str, dimensions: int) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == dimensions + 1:
            if array.shape[0] != 1:
                raise RuntimeError(f"Stage-1 output {name} must have batch size 1, got {array.shape}")
            array = array[0]
        if array.ndim != dimensions:
            raise RuntimeError(f"Stage-1 output {name} has unexpected shape {array.shape}")
        return array.astype(np.float32, copy=False)

    def run_chunk(self, values: np.ndarray, *, valid_frames: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run one BTF fbank chunk and return ``[frames, D]``, ``[frames, V]``."""

        frames = np.asarray(values, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self.contract.num_mel_bins:
            raise ValueError(
                f"Expected fbank [frames, {self.contract.num_mel_bins}], got {frames.shape}"
            )
        if frames.shape[0] > self.contract.chunk_frames:
            raise ValueError("run_chunk received more frames than contract.chunk_frames")
        original_frames = int(frames.shape[0] if valid_frames is None else valid_frames)
        if original_frames < 1 or original_frames > frames.shape[0]:
            raise ValueError("valid_frames must be in [1, chunk frame count]")
        if frames.shape[0] < self.contract.chunk_frames and self.contract.pad_final_chunk:
            padded = np.zeros((self.contract.chunk_frames, self.contract.num_mel_bins), dtype=np.float32)
            padded[: frames.shape[0]] = frames
            frames = padded
        feed: dict[str, np.ndarray] = {
            self.contract.feature_input: self._array_for_input(self.contract.feature_input, frames[np.newaxis, ...])
        }
        if self.contract.feature_length_input:
            feed[self.contract.feature_length_input] = self._scalar_for_input(
                self.contract.feature_length_input, original_frames
            )
        if self.contract.offset_input:
            feed[self.contract.offset_input] = self._scalar_for_input(self.contract.offset_input, self._offset)
        if self.contract.attention_mask:
            mask_name = self.contract.attention_mask.input_name
            feed[mask_name] = self._array_for_input(mask_name, self._streaming_attention_mask())
        for name, value in self.contract.constant_inputs.items():
            feed[name] = self._array_for_input(name, value)
        for spec in self.contract.cache_specs:
            feed[spec.input_name] = self._array_for_input(spec.input_name, self._caches[spec.input_name])

        output_names = [
            self.contract.encoder_output,
            self.contract.ctc_output,
            *(spec.output_name for spec in self.contract.cache_specs),
        ]
        outputs = self.session.run(output_names, feed)
        encoder = self._remove_batch_dimension(outputs[0], name=self.contract.encoder_output, dimensions=2)
        ctc = self._remove_batch_dimension(outputs[1], name=self.contract.ctc_output, dimensions=2)
        if encoder.shape[0] != ctc.shape[0] or encoder.shape[0] < 1:
            raise RuntimeError(
                f"Stage-1 encoder/CTC time dimensions differ: {encoder.shape} vs {ctc.shape}"
            )
        for spec, output in zip(self.contract.cache_specs, outputs[2:]):
            self._caches[spec.input_name] = self._array_for_input(spec.input_name, output)
        self._offset += int(encoder.shape[0])
        self._chunks_run += 1
        if not self.contract.ctc_output_is_log_probs:
            ctc = _log_softmax(ctc)
        if not np.isfinite(encoder).all() or not np.isfinite(ctc).all():
            raise RuntimeError("Stage-1 ONNX inference produced NaN or infinity")
        return encoder, ctc

    def infer_fbank(self, fbank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(fbank, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"Expected a non-empty fbank matrix, got {values.shape}")
        self.reset()
        encoders: list[np.ndarray] = []
        ctc_values: list[np.ndarray] = []
        if values.shape[0] < self.contract.minimum_input_frames:
            starts = [0]
        else:
            starts = range(
                0,
                values.shape[0] - self.contract.minimum_input_frames + 1,
                self.contract.chunk_stride_frames,
            )
        for start in starts:
            chunk = values[start:start + self.contract.chunk_frames]
            encoder, ctc = self.run_chunk(chunk, valid_frames=int(chunk.shape[0]))
            encoders.append(encoder)
            ctc_values.append(ctc)
        return np.concatenate(encoders, axis=0), np.concatenate(ctc_values, axis=0)


def _torchaudio() -> tuple[Any, Any]:
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required to read audio and compute fbank features") from exc
    return torch, torchaudio


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Read arbitrary audio, make it mono, and resample it for stage 1."""

    _torch, torchaudio = _torchaudio()
    waveform, source_rate = torchaudio.load(str(path))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.squeeze(0).detach().cpu().to(torch.float32).numpy()


def fixed_audio_window(
    audio: np.ndarray,
    *,
    target_samples: int,
    placement: str,
    seed: int,
) -> np.ndarray:
    """Use the same simple placement idea as the existing feature stage."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == target_samples:
        return values
    rng = random.Random(seed)
    if values.size > target_samples:
        if placement == "start":
            start = 0
        elif placement == "end":
            start = values.size - target_samples
        elif placement == "center":
            start = (values.size - target_samples) // 2
        elif placement == "random":
            start = rng.randint(0, values.size - target_samples)
        else:
            raise ValueError(f"Unknown placement {placement!r}")
        return values[start:start + target_samples]
    output = np.zeros(target_samples, dtype=np.float32)
    if placement == "start":
        start = 0
    elif placement == "end":
        start = target_samples - values.size
    elif placement == "center":
        start = (target_samples - values.size) // 2
    elif placement == "random":
        start = rng.randint(0, target_samples - values.size)
    else:
        raise ValueError(f"Unknown placement {placement!r}")
    output[start:start + values.size] = values
    return output


def audio_to_fbank(audio: np.ndarray, contract: Stage1Contract) -> np.ndarray:
    """Compute the fbank matrix described by the stage-1 contract.

    The optional mean/istd values are deliberately contract data instead of
    hidden code.  Put them in the contract only when the exported ONNX graph
    expects CMVN to have already happened outside the graph.
    """

    _torch, torchaudio = _torchaudio()
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
    values = torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=float(contract.sample_rate),
        frame_length=float(contract.frame_length_ms),
        frame_shift=float(contract.frame_shift_ms),
        num_mel_bins=int(contract.num_mel_bins),
        dither=float(contract.dither),
        energy_floor=0.0,
        use_energy=False,
    ).detach().cpu().numpy().astype(np.float32)
    if values.shape[0] == 0:
        values = np.zeros((1, contract.num_mel_bins), dtype=np.float32)
    if contract.feature_mean is not None and contract.feature_istd is not None:
        values = (values - np.asarray(contract.feature_mean, dtype=np.float32)) * np.asarray(
            contract.feature_istd, dtype=np.float32
        )
    return values


def _temporary_path(path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    temporary.unlink(missing_ok=True)
    return temporary


def _atomic_write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        write_jsonl(temporary, rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_ctc_wac_feature_bundle(
    *,
    records: Sequence[dict[str, Any]],
    output_file: Path,
    model_path: Path,
    contract_path: Path,
    keywords_path: Path,
    clip_seconds: float,
    placement: str,
    seed: int,
    device: str = "cpu",
    overwrite: bool = False,
    index_offset: int = 0,
) -> dict[str, Any]:
    """Generate features and all stage-1 values, without applying the gate.

    The feature bundle is intentionally all-row.  A later train step applies
    the keyword-specific stage-1 threshold, so a changed threshold never
    requires expensive stage-1 inference again.
    """

    if not records:
        raise ValueError("Cannot create a CTC-WAC feature bundle from an empty manifest")
    output_file = output_file.resolve()
    paths = feature_bundle_paths(output_file)
    if not overwrite and feature_bundle_valid(output_file):
        return read_json(paths.summary)
    contract = Stage1Contract.from_json(contract_path)
    keywords = load_keywords(keywords_path, require_threshold=False)
    target_samples = int(round(float(clip_seconds) * contract.sample_rate))
    if target_samples < 1:
        raise ValueError("clip_seconds must produce at least one audio sample")
    stage1 = StreamingCtcStage1(model_path, contract, device=device)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = {name: _temporary_path(path) for name, path in asdict(paths).items()}
    # asdict converts Path values unchanged; make type checkers and readers happy.
    temporary = {name: Path(value) for name, value in temporary_paths.items()}
    feature_mmap: np.memmap | None = None
    score_mmap: np.memmap | None = None
    top_mmap: np.memmap | None = None
    margin_mmap: np.memmap | None = None
    winner_mmap: np.memmap | None = None
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    feature_shape: tuple[int, int] | None = None
    row = 0
    try:
        for local_index, record in enumerate(records):
            index = int(index_offset) + local_index
            path_value = record.get("path")
            try:
                if not path_value:
                    raise ValueError("normalized manifest row has no path")
                audio = load_audio(Path(str(path_value)), contract.sample_rate)
                window = fixed_audio_window(
                    audio,
                    target_samples=target_samples,
                    placement=placement,
                    seed=int(seed) + index,
                )
                fbank = audio_to_fbank(window, contract)
                encoder, ctc = stage1.infer_fbank(fbank)
                if encoder.ndim != 2 or ctc.ndim != 2:
                    raise RuntimeError("stage-1 inference did not return two time-major matrices")
                if feature_shape is None:
                    feature_shape = (int(encoder.shape[0]), int(encoder.shape[1]))
                    feature_mmap = open_memmap(
                        temporary["features"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(records), *feature_shape),
                    )
                    score_mmap = open_memmap(
                        temporary["all_scores"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(records), len(keywords)),
                    )
                    top_mmap = open_memmap(temporary["top_score"], mode="w+", dtype=np.float32, shape=(len(records), 1))
                    margin_mmap = open_memmap(temporary["margin"], mode="w+", dtype=np.float32, shape=(len(records), 1))
                    winner_mmap = open_memmap(
                        temporary["winner_onehot"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(records), len(keywords)),
                    )
                if encoder.shape != feature_shape:
                    raise RuntimeError(
                        f"stage-1 encoder shape changed from {feature_shape} to {encoder.shape}; fixed windows must be consistent"
                    )
                score_traces = ctc_keyword_score_traces(ctc, keywords, blank_id=contract.blank_id)
                # Keep all keyword values from one actual candidate frame.
                # Taking a separate maximum for every keyword would mix scores
                # from different moments and give stage 2 a margin it will
                # never see during streaming evaluation.
                candidate_frame = int(np.argmax(np.max(score_traces, axis=1)))
                keyword_scores = score_traces[candidate_frame:candidate_frame + 1]
                top, margin, winner = rank_keyword_scores(keyword_scores)
                assert feature_mmap is not None and score_mmap is not None
                assert top_mmap is not None and margin_mmap is not None and winner_mmap is not None
                feature_mmap[row] = encoder
                score_mmap[row] = keyword_scores[0]
                top_mmap[row, 0] = top[0]
                margin_mmap[row, 0] = margin[0]
                winner_mmap[row] = winner_onehot(winner, len(keywords))[0]
                winning = int(winner[0])
                rows.append(
                    {
                        "row": row,
                        "source_index": index,
                        "id": record.get("id"),
                        "path": str(path_value),
                        "label": record.get("label"),
                        "keyword_id": keywords[winning].id,
                        "candidate_frame": candidate_frame,
                        "top_score": float(top[0]),
                        "margin": float(margin[0]),
                    }
                )
                row += 1
            except Exception as exc:
                errors.append(
                    {
                        "index": index,
                        "id": record.get("id"),
                        "path": str(path_value) if path_value else None,
                        "error": repr(exc),
                    }
                )
        if errors:
            sample = "; ".join(f"#{item['index']}: {item['error']}" for item in errors[:3])
            raise RuntimeError(
                f"CTC-WAC feature generation had {len(errors)} error(s); no partial bundle was published. {sample}"
            )
        if row != len(records) or feature_shape is None:
            raise RuntimeError("CTC-WAC feature generation did not produce all requested rows")
        assert feature_mmap is not None and score_mmap is not None
        assert top_mmap is not None and margin_mmap is not None and winner_mmap is not None
        for mmap in (feature_mmap, score_mmap, top_mmap, margin_mmap, winner_mmap):
            mmap.flush()
        # Drop references before atomic renames, which matters on Windows and
        # makes the intent clear on Linux too.
        del feature_mmap, score_mmap, top_mmap, margin_mmap, winner_mmap
        feature_mmap = score_mmap = top_mmap = margin_mmap = winner_mmap = None
        summary = {
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "output_file": str(output_file),
            "feature_count": row,
            "feature_shape": list(feature_shape),
            "keyword_count": len(keywords),
            "keyword_ids": [item.id for item in keywords],
            "keyword_token_fingerprint": keyword_token_fingerprint(keywords),
            "stage1_contract_fingerprint": contract.fingerprint(),
            "stage1_model": str(model_path),
            "stage1_contract": str(contract_path),
            "keyword_tokens": str(keywords_path),
            "stage1_providers": stage1.providers,
            "sample_rate": contract.sample_rate,
            "clip_seconds": float(clip_seconds),
            "placement": placement,
            "index_offset": int(index_offset),
            "error_count": 0,
            "errors": [],
            "stage1_gate_applied": False,
        }
        _atomic_write_rows(temporary["rows"], rows)
        _atomic_write_json(temporary["summary"], summary)
        for name, destination in asdict(paths).items():
            os.replace(temporary[name], Path(destination))
        return summary
    finally:
        # A failed extraction never overwrites a last known-good bundle.
        for path in temporary.values():
            path.unlink(missing_ok=True)


@dataclass
class CtcWacFeatureBlock:
    """Memory-mapped feature bundle plus rows retained by the stage-1 gate."""

    name: str
    label: int
    split: str
    paths: FeatureBundlePaths
    features: np.ndarray
    all_scores: np.ndarray
    top_score: np.ndarray
    margin: np.ndarray
    winner_onehot_values: np.ndarray
    retained_indices: np.ndarray

    @classmethod
    def from_feature_block(cls, block: Any, keywords: Sequence[Keyword]) -> "CtcWacFeatureBlock":
        if not feature_bundle_valid(block.path):
            raise ConfigurationError(
                f"[{block.name}] is not a valid CTC-WAC feature bundle; run its wenet_ctc_wac feature stage first"
            )
        paths = feature_bundle_paths(block.path)
        summary = read_json(paths.summary)
        expected = keyword_token_fingerprint(keywords)
        if summary.get("keyword_token_fingerprint") != expected:
            raise ConfigurationError(
                f"[{block.name}] was generated with different keyword token IDs; regenerate its feature bundle"
            )
        features = np.load(paths.features, mmap_mode="r")
        scores = np.load(paths.all_scores, mmap_mode="r")
        top = np.load(paths.top_score, mmap_mode="r")
        margin = np.load(paths.margin, mmap_mode="r")
        onehot = np.load(paths.winner_onehot, mmap_mode="r")
        return cls(
            name=block.name,
            label=int(block.label),
            split=str(block.split),
            paths=paths,
            features=features,
            all_scores=scores,
            top_score=top,
            margin=margin,
            winner_onehot_values=onehot,
            retained_indices=retained_stage1_indices(scores, keywords),
        )

    @property
    def input_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def retained_count(self) -> int:
        return int(self.retained_indices.shape[0])

    @property
    def feature_shape(self) -> tuple[int, int]:
        return (int(self.features.shape[1]), int(self.features.shape[2]))

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        return (
            np.asarray(self.features[selected], dtype=np.float32),
            np.asarray(self.top_score[selected], dtype=np.float32),
            np.asarray(self.margin[selected], dtype=np.float32),
            np.asarray(self.winner_onehot_values[selected], dtype=np.float32),
            np.full(selected.shape[0], self.label, dtype=np.float32),
        )

    def filtering_summary(self) -> dict[str, int]:
        return {
            "input_rows": self.input_count,
            "retained_rows": self.retained_count,
            "dropped_rows": self.input_count - self.retained_count,
        }


class CtcWacClassifier(torch.nn.Module):
    """A compact WAC-style model over frozen encoder frames and CTC context."""

    def __init__(
        self,
        *,
        feature_dim: int,
        keyword_count: int,
        frame_hidden: int = 128,
        frame_layers: int = 3,
        head_hidden: int = 128,
        dropout: float = 0.1,
        score_mean: float = 0.0,
        score_std: float = 1.0,
        margin_mean: float = 0.0,
        margin_std: float = 1.0,
    ):
        super().__init__()
        if min(feature_dim, keyword_count, frame_hidden, frame_layers, head_hidden) < 1:
            raise ValueError("CtcWacClassifier dimensions must all be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("CtcWacClassifier dropout must be in [0, 1)")
        self.feature_dim = int(feature_dim)
        self.keyword_count = int(keyword_count)
        self.frame_norm = torch.nn.LayerNorm(feature_dim)
        frame_layers_list: list[torch.nn.Module] = []
        input_dim = feature_dim
        for _ in range(frame_layers):
            frame_layers_list.extend(
                [torch.nn.Linear(input_dim, frame_hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            )
            input_dim = frame_hidden
        self.frame_net = torch.nn.Sequential(*frame_layers_list)
        context_dim = frame_hidden + 2 + keyword_count
        self.context_norm = torch.nn.LayerNorm(context_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(context_dim, head_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(head_hidden, 1),
        )
        self.register_buffer("score_mean", torch.tensor(float(score_mean), dtype=torch.float32))
        self.register_buffer("score_std", torch.tensor(max(float(score_std), 1.0e-6), dtype=torch.float32))
        self.register_buffer("margin_mean", torch.tensor(float(margin_mean), dtype=torch.float32))
        self.register_buffer("margin_std", torch.tensor(max(float(margin_std), 1.0e-6), dtype=torch.float32))

    def forward_logits(
        self,
        encoder_features: torch.Tensor,
        top_score: torch.Tensor,
        margin: torch.Tensor,
        winner_onehot_values: torch.Tensor,
    ) -> torch.Tensor:
        is_exporting = bool(getattr(torch.onnx, "is_in_onnx_export", lambda: False)())
        if not is_exporting:
            if encoder_features.ndim != 3:
                raise ValueError("encoder_features must have shape [batch, frames, feature_dim]")
            if encoder_features.shape[-1] != self.feature_dim:
                raise ValueError("encoder_features has the wrong feature dimension")
            if winner_onehot_values.ndim != 2 or winner_onehot_values.shape[-1] != self.keyword_count:
                raise ValueError("winner_onehot has the wrong keyword dimension")
        pooled = self.frame_net(self.frame_norm(encoder_features)).mean(dim=1)
        normalized_score = (top_score.reshape(-1, 1) - self.score_mean) / self.score_std
        normalized_margin = (margin.reshape(-1, 1) - self.margin_mean) / self.margin_std
        context = torch.cat([pooled, normalized_score, normalized_margin, winner_onehot_values], dim=1)
        return self.decoder(self.context_norm(context))

    def forward(
        self,
        encoder_features: torch.Tensor,
        top_score: torch.Tensor,
        margin: torch.Tensor,
        winner_onehot_values: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(encoder_features, top_score, margin, winner_onehot_values))


def ctc_wac_model_config(section: dict[str, str], *, section_name: str) -> dict[str, Any]:
    """Parse the small, readable stage-2 model configuration."""

    defaults = {
        "frame_hidden": 128,
        "frame_layers": 3,
        "head_hidden": 128,
        "dropout": 0.1,
    }
    result: dict[str, Any] = {}
    for key, default in defaults.items():
        raw = section.get(f"wac.{key}", section.get(key if key == "dropout" else f"wac_{key}"))
        value: Any = default if raw is None else raw
        try:
            result[key] = float(value) if key == "dropout" else int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"[{section_name}] wac.{key} is invalid") from exc
    if min(result["frame_hidden"], result["frame_layers"], result["head_hidden"]) < 1:
        raise ConfigurationError(f"[{section_name}] WAC hidden sizes and layer count must be >= 1")
    if not 0.0 <= result["dropout"] < 1.0:
        raise ConfigurationError(f"[{section_name}] wac.dropout must be in [0, 1)")
    return result


def make_ctc_wac_model(
    *,
    feature_dim: int,
    keyword_count: int,
    model_config: dict[str, Any],
    score_mean: float = 0.0,
    score_std: float = 1.0,
    margin_mean: float = 0.0,
    margin_std: float = 1.0,
) -> CtcWacClassifier:
    return CtcWacClassifier(
        feature_dim=feature_dim,
        keyword_count=keyword_count,
        frame_hidden=int(model_config["frame_hidden"]),
        frame_layers=int(model_config["frame_layers"]),
        head_hidden=int(model_config["head_hidden"]),
        dropout=float(model_config["dropout"]),
        score_mean=score_mean,
        score_std=score_std,
        margin_mean=margin_mean,
        margin_std=margin_std,
    )


def load_stage2_onnx(path: Path) -> tuple[Any, dict[str, tuple[int | None, ...]]]:
    """Open a multi-input WAC ONNX model and expose its concrete input shapes."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for CTC-WAC evaluation") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = {"encoder_features", "top_score", "margin", "winner_onehot"}
    available = {item.name: item for item in session.get_inputs()}
    missing = sorted(expected - set(available))
    if missing:
        raise ConfigurationError(
            f"Stage-2 ONNX model {path} is not a CTC-WAC export; missing input(s): {', '.join(missing)}"
        )
    shapes: dict[str, tuple[int | None, ...]] = {}
    for name in expected:
        shape: list[int | None] = []
        for value in available[name].shape:
            shape.append(int(value) if isinstance(value, int) else None)
        shapes[name] = tuple(shape)
    return session, shapes
