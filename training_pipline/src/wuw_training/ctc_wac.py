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
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap

from .artifacts import hash_payload, read_json, write_json, write_jsonl
from .config import ConfigurationError


# Schema 4 stores variable-length candidate features as one flat matrix plus
# row offsets, records the expected wake word for positive examples, and keeps
# the selected keyword-versus-filler CTC comparison values. It deliberately
# cannot be confused with the old fixed-window ``[N, T, D]`` artifact format.
BUNDLE_SCHEMA_VERSION = 4

# torchaudio.load returns floating-point PCM in roughly [-1, 1], while
# WeNet's dataset frontend converts it back to the signed-16-bit amplitude
# range before calling torchaudio.compliance.kaldi.fbank. The exported
# encoder contains the global CMVN learned from those features, so omitting
# this scale makes an otherwise healthy CTC model emit almost only blanks.
WENET_WAVEFORM_SCALE = float(1 << 15)


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
    waveform_scale: float
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
    schema_version: int = 1
    encoder_frame_shift_ms: float = 40.0
    encoder_output_size: int | None = None
    vocab_size: int | None = None
    subsampling_factor: int | None = None
    encoder_chunk_frames: int | None = None
    token_table_fingerprint: str | None = None

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
            schema_version = int(raw.get("schema_version", 1))
            sample_rate = int(raw.get("sample_rate", 16000))
            num_mel_bins = int(fbank.get("num_mel_bins", 80))
            frame_length_ms = float(fbank.get("frame_length_ms", 25.0))
            frame_shift_ms = float(fbank.get("frame_shift_ms", 10.0))
            dither = float(fbank.get("dither", 0.0))
            waveform_scale = float(fbank.get("waveform_scale", WENET_WAVEFORM_SCALE))
            chunk_frames = int(raw["chunk_frames"])
            chunk_stride_frames = int(raw.get("chunk_stride_frames", chunk_frames))
            minimum_input_frames = int(raw.get("minimum_input_frames", 1))
            blank_id = int(raw.get("blank_id", 0))
            initial_offset = int(raw.get("initial_offset", 0))
            subsampling_factor = int(raw.get("subsampling_factor", 4))
            encoder_frame_shift_ms = float(
                raw.get("encoder_frame_shift_ms", frame_shift_ms * subsampling_factor)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Stage-1 contract {path} needs numeric sample_rate/fbank/chunk_frames values"
            ) from exc
        if (
            sample_rate < 1
            or num_mel_bins < 1
            or frame_length_ms <= 0
            or frame_shift_ms <= 0
            or not math.isfinite(waveform_scale)
            or waveform_scale <= 0
            or chunk_frames < 1
            or chunk_stride_frames < 1
            or minimum_input_frames < 1
            or minimum_input_frames > chunk_frames
            or initial_offset < 0
            or schema_version < 1
            or subsampling_factor < 1
            or encoder_frame_shift_ms <= 0
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
        def optional_positive_int(key: str) -> int | None:
            value = raw.get(key)
            if value is None:
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Stage-1 contract {path}: {key} must be an integer") from exc
            if parsed < 1:
                raise ConfigurationError(f"Stage-1 contract {path}: {key} must be >= 1")
            return parsed

        token_table_fingerprint = raw.get("token_table_fingerprint")
        if token_table_fingerprint is not None and not isinstance(token_table_fingerprint, str):
            raise ConfigurationError(f"Stage-1 contract {path}: token_table_fingerprint must be a string")
        return cls(
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms,
            dither=dither,
            waveform_scale=waveform_scale,
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
            schema_version=schema_version,
            encoder_frame_shift_ms=encoder_frame_shift_ms,
            encoder_output_size=optional_positive_int("encoder_output_size"),
            vocab_size=optional_positive_int("vocab_size"),
            subsampling_factor=subsampling_factor,
            encoder_chunk_frames=optional_positive_int("encoder_chunk_frames"),
            token_table_fingerprint=token_table_fingerprint,
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


def _expected_keyword_id(
    record: dict[str, Any],
    *,
    keyword_ids: set[str],
    keyword_by_text: dict[str, str],
) -> str | None:
    """Resolve a positive record's expected keyword without using its CTC winner."""

    for field in ("expected_keyword_id", "keyword_id", "wakeword_id"):
        value = record.get(field)
        if isinstance(value, str) and value in keyword_ids:
            return value
    text = record.get("text")
    if isinstance(text, str):
        return keyword_by_text.get(text.strip().casefold())
    return None


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
    offsets: Path
    lengths: Path
    all_scores: Path
    top_score: Path
    keyword_score: Path
    filler_score: Path
    raw_score: Path
    normalized_raw_score: Path
    confidence: Path
    normalized_confidence: Path
    segment_length: Path
    margin: Path
    winner_onehot: Path
    rows: Path
    summary: Path

    @property
    def debug_alignments(self) -> Path:
        """Optional per-input CTC alignment log, written only in debug mode."""

        return self.rows.with_name(f"{self.features.stem}.debug.jsonl")

    def all(self) -> list[Path]:
        return [
            self.features,
            self.offsets,
            self.lengths,
            self.all_scores,
            self.top_score,
            self.keyword_score,
            self.filler_score,
            self.raw_score,
            self.normalized_raw_score,
            self.confidence,
            self.normalized_confidence,
            self.segment_length,
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
        offsets=parent / f"{stem}.offsets.npy",
        lengths=parent / f"{stem}.lengths.npy",
        all_scores=parent / f"{stem}.all_scores.npy",
        top_score=parent / f"{stem}.top_score.npy",
        keyword_score=parent / f"{stem}.keyword_score.npy",
        filler_score=parent / f"{stem}.filler_score.npy",
        raw_score=parent / f"{stem}.raw_score.npy",
        normalized_raw_score=parent / f"{stem}.normalized_raw_score.npy",
        confidence=parent / f"{stem}.confidence.npy",
        normalized_confidence=parent / f"{stem}.normalized_confidence.npy",
        segment_length=parent / f"{stem}.segment_length.npy",
        margin=parent / f"{stem}.margin.npy",
        winner_onehot=parent / f"{stem}.winner_onehot.npy",
        rows=parent / f"{stem}.rows.jsonl",
        summary=parent / f"{stem}.summary.json",
    )


def feature_bundle_valid(
    features: Path,
    *,
    require_complete: bool = True,
    expected_stage1_contract_fingerprint: str | None = None,
    require_debug_alignments: bool = False,
) -> bool:
    """Check the shapes and summary needed by CTC-WAC training.

    It is intentionally inexpensive: it reads NPY headers through memory maps
    instead of pulling the feature tensors into RAM.
    """

    paths = feature_bundle_paths(features)
    if not all(path.is_file() for path in paths.all()):
        return False
    if require_debug_alignments and not paths.debug_alignments.is_file():
        return False
    try:
        summary = read_json(paths.summary)
        x = np.load(paths.features, mmap_mode="r")
        offsets = np.load(paths.offsets, mmap_mode="r")
        lengths = np.load(paths.lengths, mmap_mode="r")
        scores = np.load(paths.all_scores, mmap_mode="r")
        top = np.load(paths.top_score, mmap_mode="r")
        keyword_score = np.load(paths.keyword_score, mmap_mode="r")
        filler_score = np.load(paths.filler_score, mmap_mode="r")
        raw_score = np.load(paths.raw_score, mmap_mode="r")
        normalized_raw_score = np.load(paths.normalized_raw_score, mmap_mode="r")
        confidence = np.load(paths.confidence, mmap_mode="r")
        normalized_confidence = np.load(paths.normalized_confidence, mmap_mode="r")
        segment_length = np.load(paths.segment_length, mmap_mode="r")
        margin = np.load(paths.margin, mmap_mode="r")
        winner = np.load(paths.winner_onehot, mmap_mode="r")
        n = int(lengths.shape[0])
        offset_values = np.asarray(offsets, dtype=np.int64)
        length_values = np.asarray(lengths, dtype=np.int64)
        valid = (
            int(summary.get("bundle_schema", -1)) == BUNDLE_SCHEMA_VERSION
            and (not require_complete or int(summary.get("error_count", -1)) == 0)
            and (not require_debug_alignments or summary.get("debug_alignment_enabled") is True)
            and (
                not require_debug_alignments
                or int(summary.get("debug_alignment_rows", -1)) == int(summary.get("input_count", -2))
            )
            and (
                expected_stage1_contract_fingerprint is None
                or summary.get("stage1_contract_fingerprint")
                == expected_stage1_contract_fingerprint
            )
            and int(summary.get("feature_count", -1)) == n
            and x.ndim == 2
            and x.shape[1] >= 1
            and offsets.ndim == 1
            and lengths.ndim == 1
            and offsets.shape == (n + 1,)
            and offset_values[0] == 0
            and offset_values[-1] == x.shape[0]
            and np.all(np.diff(offset_values) == length_values)
            and np.all(length_values > 0)
            and scores.ndim == 2
            and top.shape == (n, 1)
            and keyword_score.shape == (n, 1)
            and filler_score.shape == (n, 1)
            and raw_score.shape == (n, 1)
            and normalized_raw_score.shape == (n, 1)
            and confidence.shape == (n, 1)
            and normalized_confidence.shape == (n, 1)
            and segment_length.shape == (n, 1)
            and margin.shape == (n, 1)
            and winner.shape == scores.shape
            and scores.shape[0] == n
            and scores.shape[1] >= 1
            and np.all(np.isfinite(keyword_score))
            and np.all(np.isfinite(filler_score))
            and np.all(np.isfinite(raw_score))
            and np.all(np.isfinite(normalized_raw_score))
            and np.all(np.isfinite(confidence))
            and np.all(np.isfinite(normalized_confidence))
            and np.all((confidence >= 0.0) & (confidence <= 1.0))
            and np.all((normalized_confidence >= 0.0) & (normalized_confidence <= 1.0))
            and np.all(np.asarray(segment_length, dtype=np.int64) > 0)
        )
        return bool(valid)
    except Exception:
        return False


def _log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(values, axis=-1, keepdims=True)
    return values - maximum - np.log(np.sum(np.exp(values - maximum), axis=-1, keepdims=True))


def _ctc_values(log_probs: np.ndarray, *, blank_id: int) -> np.ndarray:
    """Validate one time-major CTC log-probability matrix."""

    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(
            "CTC scores must have shape [frames, vocabulary] with at least one frame, "
            f"got {values.shape}"
        )
    if blank_id < 0 or blank_id >= values.shape[1]:
        raise ValueError("CTC blank token ID is outside the output vocabulary")
    return values


def _ctc_target_tokens(
    token_ids: Sequence[int],
    *,
    vocabulary_size: int,
    blank_id: int,
    allow_empty: bool,
) -> tuple[int, ...]:
    tokens = tuple(int(item) for item in token_ids)
    if not tokens and not allow_empty:
        raise ValueError("A CTC keyword needs at least one token")
    if any(item < 0 or item >= vocabulary_size for item in tokens):
        raise ValueError("CTC keyword token ID is outside the output vocabulary")
    if blank_id in tokens:
        raise ValueError("CTC keyword token IDs must not include the blank ID")
    return tokens


def ctc_forward_score(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    blank_id: int = 0,
) -> float:
    """Return the standard CTC forward log probability of one token sequence.

    The target is expanded to ``blank, token_1, blank, ... , token_n,
    blank``.  Unlike the stage-1 candidate finder, this sums every legal CTC
    path in log space rather than retaining only the Viterbi predecessor.
    ``token_ids`` may be empty, which represents the all-blank sequence and
    is useful when it is the best non-keyword filler hypothesis.
    """

    values = _ctc_values(log_probs, blank_id=blank_id)
    tokens = _ctc_target_tokens(
        token_ids,
        vocabulary_size=values.shape[1],
        blank_id=blank_id,
        allow_empty=True,
    )
    extended: list[int] = [blank_id]
    for token in tokens:
        extended.extend([token, blank_id])
    state_count = len(extended)
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    previous[0] = values[0, blank_id]
    if state_count > 1:
        previous[1] = values[0, tokens[0]]

    for frame in values[1:]:
        current = np.full(state_count, -np.inf, dtype=np.float64)
        for state, symbol in enumerate(extended):
            predecessors = [previous[state]]  # stay in the same CTC state
            if state > 0:
                predecessors.append(previous[state - 1])  # advance one state
            if state > 1 and state % 2 == 1 and symbol != extended[state - 2]:
                predecessors.append(previous[state - 2])  # skip a blank when CTC permits it
            current[state] = np.logaddexp.reduce(np.asarray(predecessors, dtype=np.float64)) + frame[symbol]
        previous = current

    if state_count == 1:
        return float(previous[0])
    return float(np.logaddexp(previous[-2], previous[-1]))


def _logaddexp_pair(left: float, right: float) -> float:
    """Fast scalar log-add-exp for the small prefix-beam state maps."""

    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    if left < right:
        left, right = right, left
    return left + math.log1p(math.exp(right - left))


@dataclass(frozen=True)
class CtcPrefixBeamHypothesis:
    """One collapsed token sequence retained by CTC prefix beam search."""

    token_ids: tuple[int, ...]
    log_score: float


def _prefix_beam_tokens(
    frame: np.ndarray,
    *,
    blank_id: int,
    token_prune: int | None,
) -> tuple[int, ...]:
    """Return the highest-scoring non-blank token IDs for one beam step."""

    vocabulary_size = int(frame.shape[0])
    nonblank = np.concatenate(
        [np.arange(blank_id, dtype=np.int64), np.arange(blank_id + 1, vocabulary_size, dtype=np.int64)]
    )
    if token_prune is not None:
        if int(token_prune) < 1:
            raise ValueError("CTC prefix beam token_prune must be >= 1 when configured")
        if int(token_prune) < nonblank.size:
            local = np.argpartition(frame[nonblank], -int(token_prune))[-int(token_prune):]
            nonblank = nonblank[local]
    return tuple(sorted((int(item) for item in nonblank), key=lambda item: (-float(frame[item]), item)))


def ctc_prefix_beam_search(
    log_probs: np.ndarray,
    *,
    blank_id: int = 0,
    beam_size: int = 16,
    token_prune: int | None = None,
) -> list[CtcPrefixBeamHypothesis]:
    """Decode top collapsed CTC token sequences with prefix beam search.

    Scores inside this search are used only to select the strongest decoded
    non-keyword sequence.  Call :func:`ctc_forward_score` afterwards to get
    the full, non-Viterbi CTC probability of the selected sequence.
    """

    values = _ctc_values(log_probs, blank_id=blank_id)
    if int(beam_size) < 1:
        raise ValueError("CTC prefix beam_size must be >= 1")
    beam_width = int(beam_size)
    negative_infinity = -math.inf
    # Prefix -> (ends in blank log probability, ends in non-blank log probability).
    beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, negative_infinity)}
    for frame in values:
        next_blank: dict[tuple[int, ...], float] = {}
        next_nonblank: dict[tuple[int, ...], float] = {}
        token_ids = _prefix_beam_tokens(frame, blank_id=blank_id, token_prune=token_prune)
        blank_log_prob = float(frame[blank_id])
        for prefix, (prob_blank, prob_nonblank) in beam.items():
            prefix_total = _logaddexp_pair(prob_blank, prob_nonblank)
            blank_value = prefix_total + blank_log_prob
            next_blank[prefix] = _logaddexp_pair(next_blank.get(prefix, negative_infinity), blank_value)
            for token_id in token_ids:
                token_value = float(frame[token_id])
                if prefix and token_id == prefix[-1]:
                    # Repeating a label without a separating blank keeps the
                    # same collapsed sequence.  Repeating after a blank adds a
                    # second copy of that label to the collapsed sequence.
                    if prob_nonblank != negative_infinity:
                        next_nonblank[prefix] = _logaddexp_pair(
                            next_nonblank.get(prefix, negative_infinity),
                            prob_nonblank + token_value,
                        )
                    if prob_blank != negative_infinity:
                        extended = prefix + (token_id,)
                        next_nonblank[extended] = _logaddexp_pair(
                            next_nonblank.get(extended, negative_infinity),
                            prob_blank + token_value,
                        )
                elif prefix_total != negative_infinity:
                    extended = prefix + (token_id,)
                    next_nonblank[extended] = _logaddexp_pair(
                        next_nonblank.get(extended, negative_infinity),
                        prefix_total + token_value,
                    )
        ranked = [
            (
                prefix,
                _logaddexp_pair(
                    next_blank.get(prefix, negative_infinity),
                    next_nonblank.get(prefix, negative_infinity),
                ),
            )
            for prefix in set(next_blank) | set(next_nonblank)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        beam = {
            prefix: (next_blank.get(prefix, negative_infinity), next_nonblank.get(prefix, negative_infinity))
            for prefix, _score in ranked[:beam_width]
        }
    return [
        CtcPrefixBeamHypothesis(token_ids=prefix, log_score=float(score))
        for prefix, score in sorted(
            (
                (prefix, _logaddexp_pair(prob_blank, prob_nonblank))
                for prefix, (prob_blank, prob_nonblank) in beam.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _sigmoid(value: float) -> float:
    """Return a finite sigmoid without overflowing for extreme score gaps."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True)
class CtcKeywordVsFillerScore:
    """Forward-score comparison for one selected keyword candidate segment."""

    keyword_score: float
    filler_score: float
    raw_score: float
    normalized_raw_score: float
    confidence: float
    normalized_confidence: float
    segment_length: int
    filler_token_ids: tuple[int, ...]


def ctc_keyword_vs_filler_score(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    blank_id: int = 0,
    beam_size: int = 16,
    token_prune: int | None = 8,
) -> CtcKeywordVsFillerScore:
    """Compare a keyword with the strongest beam-decoded non-keyword sequence.

    Both forward scores use exactly the supplied CTC segment.  The beam search
    chooses a collapsed competitor sequence different from ``token_ids``;
    its final score is then recomputed with the same forward algorithm as the
    keyword, rather than comparing a forward score with a Viterbi score.
    """

    values = _ctc_values(log_probs, blank_id=blank_id)
    tokens = _ctc_target_tokens(
        token_ids,
        vocabulary_size=values.shape[1],
        blank_id=blank_id,
        allow_empty=False,
    )
    if int(beam_size) < 2:
        raise ValueError("CTC keyword-versus-filler beam_size must be >= 2")
    hypotheses = ctc_prefix_beam_search(
        values,
        blank_id=blank_id,
        beam_size=int(beam_size),
        token_prune=token_prune,
    )
    filler_tokens = next((item.token_ids for item in hypotheses if item.token_ids != tokens), None)
    if filler_tokens is None:
        raise RuntimeError("CTC prefix beam search did not retain a non-keyword filler hypothesis")
    keyword_score = ctc_forward_score(values, tokens, blank_id=blank_id)
    filler_score = ctc_forward_score(values, filler_tokens, blank_id=blank_id)
    if not math.isfinite(keyword_score) or not math.isfinite(filler_score):
        raise RuntimeError("CTC keyword-versus-filler forward score is not finite")
    segment_length = int(values.shape[0])
    raw_score = keyword_score - filler_score
    normalized_raw_score = raw_score / segment_length
    return CtcKeywordVsFillerScore(
        keyword_score=keyword_score,
        filler_score=filler_score,
        raw_score=raw_score,
        normalized_raw_score=normalized_raw_score,
        confidence=_sigmoid(raw_score),
        normalized_confidence=_sigmoid(normalized_raw_score),
        segment_length=segment_length,
        filler_token_ids=filler_tokens,
    )


@dataclass(frozen=True)
class CtcKeywordAlignmentTrace:
    """Best score and CTC token boundaries for every possible end frame."""

    scores: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def _better_alignment(
    left: tuple[np.float32, int, int], right: tuple[np.float32, int, int]
) -> tuple[np.float32, int, int]:
    """Choose a deterministic Viterbi predecessor.

    Scores are primary.  For a score tie, a later start produces the shorter,
    more useful wake-word crop; an earlier final token then breaks any remaining
    tie.  This does not alter non-tied score traces from the previous scorer.
    """

    if right[0] > left[0]:
        return right
    if right[0] < left[0]:
        return left
    if right[1] > left[1]:
        return right
    if right[1] < left[1]:
        return left
    return right if right[2] < left[2] else left


def ctc_keyword_alignment_trace(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    blank_id: int = 0,
    max_search_frames: int | None = None,
) -> CtcKeywordAlignmentTrace:
    """Return normalized CTC scores and token start/end frames.

    The dynamic program is the previous restartable Viterbi scorer with
    boundary metadata propagated beside each state.  A boundary ends at the
    final non-blank token, so trailing CTC blanks never become stage-2 audio.
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

    if max_search_frames is not None:
        if int(max_search_frames) < 1:
            raise ValueError("max_search_frames must be >= 1 when configured")
        # Augmentation caps every training waveform at the same maximum range.
        # In that common case the whole matrix already lies inside the rolling
        # horizon, so the original O(T * states) scorer is exact and much
        # faster for large feature-generation corpora.
        if values.shape[0] > int(max_search_frames):
            return BoundedCtcKeywordScorer(
                tokens,
                blank_id=blank_id,
                max_search_frames=int(max_search_frames),
            ).process(values)

    # [blank, token_0, blank, token_1, ..., blank]
    extended: list[int] = [blank_id]
    for token in tokens:
        extended.extend([token, blank_id])
    state_count = len(extended)
    negative_infinity = np.float32(-1.0e30)
    previous_score = np.full(state_count, negative_infinity, dtype=np.float32)
    previous_start = np.full(state_count, -1, dtype=np.int64)
    previous_end = np.full(state_count, -1, dtype=np.int64)
    trace = np.full(values.shape[0], negative_infinity, dtype=np.float32)
    starts = np.full(values.shape[0], -1, dtype=np.int64)
    ends = np.full(values.shape[0], -1, dtype=np.int64)

    for frame_index, frame in enumerate(values):
        current_score = np.full(state_count, negative_infinity, dtype=np.float32)
        current_start = np.full(state_count, -1, dtype=np.int64)
        current_end = np.full(state_count, -1, dtype=np.int64)
        # Restarting here removes arbitrary speech before a possible keyword.
        current_score[0] = np.float32(frame[blank_id])
        for state in range(1, state_count):
            symbol = extended[state]
            best = (previous_score[state], int(previous_start[state]), int(previous_end[state]))
            best = _better_alignment(
                best,
                (previous_score[state - 1], int(previous_start[state - 1]), int(previous_end[state - 1])),
            )
            if state == 1:
                best = _better_alignment(best, (np.float32(0.0), -1, -1))
            elif state % 2 == 1 and extended[state] != extended[state - 2]:
                best = _better_alignment(
                    best,
                    (previous_score[state - 2], int(previous_start[state - 2]), int(previous_end[state - 2])),
                )
            current_score[state] = best[0] + np.float32(frame[symbol])
            if state % 2 == 1:  # emitted a token at this frame
                current_start[state] = frame_index if best[1] < 0 else best[1]
                current_end[state] = frame_index
            else:  # emitted a blank: preserve the last non-blank boundary
                current_start[state] = best[1]
                current_end[state] = best[2]

        final = _better_alignment(
            (current_score[-2], int(current_start[-2]), int(current_end[-2])),
            (current_score[-1], int(current_start[-1]), int(current_end[-1])),
        )
        trace[frame_index] = final[0] / np.float32(len(tokens))
        if final[0] > negative_infinity / 2:
            starts[frame_index] = final[1]
            ends[frame_index] = final[2]
        previous_score, previous_start, previous_end = current_score, current_start, current_end
    return CtcKeywordAlignmentTrace(trace, starts, ends)


class BoundedCtcKeywordScorer:
    """Incremental restartable CTC Viterbi scorer with a finite search horizon.

    The frozen ONNX model still runs as one continuous stream.  This class only
    limits which CTC *candidate paths* may be considered at a given output
    frame: their first keyword token must have been emitted within the most
    recent ``max_search_frames`` encoder frames.  This prevents an old prefix
    from influencing a later candidate while preserving all ONNX cache context.

    One dynamic-programming row is retained for each possible first-token
    frame in the horizon.  That is necessary for exact bounded scoring: keeping
    only the single best historical path would lose a lower-scoring path that
    becomes valid after the old best path expires.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        *,
        blank_id: int = 0,
        max_search_frames: int,
    ) -> None:
        self.tokens = tuple(int(item) for item in token_ids)
        if not self.tokens:
            raise ValueError("A CTC keyword needs at least one token")
        if int(max_search_frames) < 1:
            raise ValueError("max_search_frames must be >= 1")
        if blank_id < 0 or blank_id in self.tokens:
            raise ValueError("CTC keyword token IDs must be valid non-blank IDs")
        self.blank_id = int(blank_id)
        self.max_search_frames = int(max_search_frames)
        self.extended: tuple[int, ...] = tuple(
            symbol for token in self.tokens for symbol in (self.blank_id, token)
        ) + (self.blank_id,)
        self.state_count = len(self.extended)
        self.negative_infinity = np.float32(-1.0e30)
        self._scores = np.full(
            (self.max_search_frames, self.state_count), self.negative_infinity, dtype=np.float32
        )
        self._ends = np.full((self.max_search_frames, self.state_count), -1, dtype=np.int64)
        self._starts = np.full(self.max_search_frames, -1, dtype=np.int64)
        self._frame_index = 0
        self._vocabulary_size: int | None = None

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        values = np.asarray(frame, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"CTC frame must have shape [vocabulary], got {values.shape}")
        if self._vocabulary_size is None:
            self._vocabulary_size = int(values.shape[0])
            if (
                self.blank_id >= self._vocabulary_size
                or any(item < 0 or item >= self._vocabulary_size for item in self.tokens)
            ):
                raise ValueError("CTC keyword token ID is outside the output vocabulary")
        elif int(values.shape[0]) != self._vocabulary_size:
            raise ValueError("CTC vocabulary size changed during streaming scoring")
        return values

    @staticmethod
    def _take_better_predecessor(
        best_score: np.ndarray,
        best_end: np.ndarray,
        score: np.ndarray,
        end: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized form of ``_better_alignment`` for one fixed start row."""

        take = (score > best_score) | ((score == best_score) & (end < best_end))
        return np.where(take, score, best_score), np.where(take, end, best_end)

    def push(self, frame: np.ndarray) -> tuple[np.float32, int, int]:
        """Consume one continuous CTC frame and return its bounded best path."""

        values = self._validate_frame(frame)
        frame_index = self._frame_index
        slot = frame_index % self.max_search_frames

        # The circular slot contains exactly the candidate whose first token
        # was emitted ``max_search_frames`` frames ago.  It is out of range at
        # this frame and must disappear before any transition is made.
        self._scores[slot].fill(self.negative_infinity)
        self._ends[slot].fill(-1)
        self._starts[slot] = -1

        previous_score = self._scores
        previous_end = self._ends
        current_score = np.full_like(previous_score, self.negative_infinity)
        current_end = np.full_like(previous_end, -1)

        # State zero is intentionally not stored per candidate.  Before the
        # first keyword token, a direct restart at the current frame always
        # dominates a path containing preceding blank log-probabilities.
        # State one below seeds exactly that direct restart for this frame.
        for state in range(1, self.state_count):
            best_score = previous_score[:, state].copy()
            best_end = previous_end[:, state].copy()
            best_score, best_end = self._take_better_predecessor(
                best_score,
                best_end,
                previous_score[:, state - 1],
                previous_end[:, state - 1],
            )
            if state % 2 == 1 and state > 1 and self.extended[state] != self.extended[state - 2]:
                best_score, best_end = self._take_better_predecessor(
                    best_score,
                    best_end,
                    previous_score[:, state - 2],
                    previous_end[:, state - 2],
                )
            current_score[:, state] = best_score + np.float32(values[self.extended[state]])
            valid = best_score > self.negative_infinity / 2
            if state % 2 == 1:
                current_end[:, state] = np.where(valid, frame_index, -1)
            else:
                current_end[:, state] = np.where(valid, best_end, -1)

        self._scores = current_score
        self._ends = current_end
        # Start a fresh path on the first keyword token at this encoder frame.
        self._starts[slot] = frame_index
        self._scores[slot, 1] = np.float32(values[self.tokens[0]])
        self._ends[slot, 1] = frame_index

        final_score = self._scores[:, -2].copy()
        final_end = self._ends[:, -2].copy()
        final_score, final_end = self._take_better_predecessor(
            final_score,
            final_end,
            self._scores[:, -1],
            self._ends[:, -1],
        )
        valid_slots = np.flatnonzero(final_score > self.negative_infinity / 2)
        self._frame_index += 1
        if valid_slots.size == 0:
            return np.float32(self.negative_infinity / np.float32(len(self.tokens))), -1, -1

        best = (self.negative_infinity, -1, -1)
        for candidate_slot in valid_slots.tolist():
            best = _better_alignment(
                best,
                (
                    np.float32(final_score[candidate_slot]),
                    int(self._starts[candidate_slot]),
                    int(final_end[candidate_slot]),
                ),
            )
        return np.float32(best[0] / np.float32(len(self.tokens))), int(best[1]), int(best[2])

    def process(self, log_probs: np.ndarray) -> CtcKeywordAlignmentTrace:
        """Return the bounded trace for a complete, continuously inferred matrix."""

        values = np.asarray(log_probs, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"CTC scores must have shape [frames, vocabulary], got {values.shape}")
        scores = np.full(values.shape[0], self.negative_infinity, dtype=np.float32)
        starts = np.full(values.shape[0], -1, dtype=np.int64)
        ends = np.full(values.shape[0], -1, dtype=np.int64)
        for index, frame in enumerate(values):
            score, start, end = self.push(frame)
            scores[index] = score
            starts[index] = start
            ends[index] = end
        return CtcKeywordAlignmentTrace(scores, starts, ends)


def ctc_keyword_score_trace(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    blank_id: int = 0,
    max_search_frames: int | None = None,
) -> np.ndarray:
    """Return the best length-normalized CTC alignment score at each frame."""

    return ctc_keyword_alignment_trace(
        log_probs,
        token_ids,
        blank_id=blank_id,
        max_search_frames=max_search_frames,
    ).scores


def ctc_keyword_score_traces(
    log_probs: np.ndarray,
    keywords: Sequence[Keyword],
    *,
    blank_id: int = 0,
    max_search_frames: int | None = None,
) -> np.ndarray:
    """Return ``[frames, keyword_count]`` normalized CTC scores."""

    if not keywords:
        raise ValueError("At least one keyword is required")
    return np.stack(
        [
            ctc_keyword_score_trace(
                log_probs,
                item.token_ids,
                blank_id=blank_id,
                max_search_frames=max_search_frames,
            )
            for item in keywords
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def ctc_keyword_alignment_traces(
    log_probs: np.ndarray,
    keywords: Sequence[Keyword],
    *,
    blank_id: int = 0,
    max_search_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return score, start, and end matrices with shape ``[frames, keywords]``."""

    if not keywords:
        raise ValueError("At least one keyword is required")
    traces = [
        ctc_keyword_alignment_trace(
            log_probs,
            item.token_ids,
            blank_id=blank_id,
            max_search_frames=max_search_frames,
        )
        for item in keywords
    ]
    return (
        np.stack([item.scores for item in traces], axis=1).astype(np.float32, copy=False),
        np.stack([item.starts for item in traces], axis=1).astype(np.int64, copy=False),
        np.stack([item.ends for item in traces], axis=1).astype(np.int64, copy=False),
    )


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


@dataclass(frozen=True)
class CtcCandidate:
    """One best keyword hypothesis for an utterance or augmented window."""

    frame: int
    keyword_index: int
    scores: np.ndarray
    top_score: float
    margin: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class CtcTokenAlignment:
    """One token's Viterbi span inside a selected CTC candidate."""

    token_index: int
    token_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    log_score: float
    normalized_score: float


def ctc_candidate_token_alignments(
    log_probs: np.ndarray,
    token_ids: Sequence[int],
    *,
    candidate_start_frame: int,
    candidate_end_frame: int,
    blank_id: int = 0,
) -> list[CtcTokenAlignment]:
    """Return Viterbi token spans for one already selected CTC candidate.

    ``candidate_start_frame`` and ``candidate_end_frame`` are the non-blank
    boundaries produced by :func:`best_ctc_candidate`.  The alignment is
    constrained to begin with the first keyword token at the former and end
    with the last keyword token at the latter.  A token's normalized score is
    its mean CTC log-probability across the frames assigned to that token.

    This function is intentionally separate from normal candidate scoring so
    feature extraction pays the backtracking cost only when debug alignment
    logging is enabled.
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
    if not 0 <= candidate_start_frame <= candidate_end_frame < values.shape[0]:
        raise ValueError(
            "CTC candidate boundaries must satisfy "
            "0 <= candidate_start_frame <= candidate_end_frame < frame_count"
        )

    candidate = values[candidate_start_frame:candidate_end_frame + 1]
    extended: list[int] = [blank_id]
    for token in tokens:
        extended.extend([token, blank_id])
    state_count = len(extended)
    negative_infinity = np.float32(-1.0e30)
    previous_score = np.full(state_count, negative_infinity, dtype=np.float32)
    predecessors = np.full((candidate.shape[0], state_count), -1, dtype=np.int32)

    # The recorded start boundary is the first non-blank keyword token, so
    # leading CTC blanks are deliberately excluded from this forced alignment.
    previous_score[1] = np.float32(candidate[0, tokens[0]])
    for frame_index in range(1, candidate.shape[0]):
        current_score = np.full(state_count, negative_infinity, dtype=np.float32)
        for state in range(1, state_count):
            symbol = extended[state]
            candidate_states = [state, state - 1]
            if state % 2 == 1 and extended[state] != extended[state - 2]:
                candidate_states.append(state - 2)
            best_state = candidate_states[0]
            best_score = previous_score[best_state]
            for previous_state in candidate_states[1:]:
                if previous_score[previous_state] > best_score:
                    best_score = previous_score[previous_state]
                    best_state = previous_state
            if best_score > negative_infinity / 2:
                current_score[state] = best_score + np.float32(candidate[frame_index, symbol])
                predecessors[frame_index, state] = best_state
        previous_score = current_score

    final_state = state_count - 2
    if previous_score[final_state] <= negative_infinity / 2:
        raise RuntimeError("Selected CTC candidate has no token-level alignment")
    states = np.empty(candidate.shape[0], dtype=np.int32)
    state = final_state
    for frame_index in range(candidate.shape[0] - 1, -1, -1):
        states[frame_index] = state
        state = int(predecessors[frame_index, state])
    if int(states[0]) != 1:
        raise RuntimeError("Selected CTC candidate alignment does not start with its first token")

    alignments: list[CtcTokenAlignment] = []
    for token_index, token_id in enumerate(tokens):
        state_index = 2 * token_index + 1
        local_frames = np.flatnonzero(states == state_index)
        if local_frames.size == 0:
            raise RuntimeError(f"Selected CTC candidate did not emit token index {token_index}")
        token_log_scores = candidate[local_frames, token_id]
        frame_count = int(local_frames.size)
        alignments.append(
            CtcTokenAlignment(
                token_index=token_index,
                token_id=token_id,
                start_frame=int(candidate_start_frame + local_frames[0]),
                end_frame=int(candidate_start_frame + local_frames[-1]),
                frame_count=frame_count,
                log_score=float(token_log_scores.sum()),
                normalized_score=float(token_log_scores.mean()),
            )
        )
    return alignments


def best_ctc_candidate(
    log_probs: np.ndarray,
    keywords: Sequence[Keyword],
    *,
    blank_id: int = 0,
    max_search_frames: int | None = None,
) -> CtcCandidate:
    """Select the one globally best keyword candidate from a CTC matrix."""

    scores, starts, ends = ctc_keyword_alignment_traces(
        log_probs,
        keywords,
        blank_id=blank_id,
        max_search_frames=max_search_frames,
    )
    if scores.shape[0] < 1:
        raise ValueError("CTC output contains no encoder frames")
    # ``argmax`` intentionally chooses the earliest frame for an exact score
    # tie, matching the former feature extractor's deterministic behavior.
    frame = int(np.argmax(np.max(scores, axis=1)))
    row = scores[frame:frame + 1]
    top, margin, winner = rank_keyword_scores(row)
    keyword_index = int(winner[0])
    start_frame = int(starts[frame, keyword_index])
    end_frame = int(ends[frame, keyword_index])
    if start_frame < 0 or end_frame < start_frame:
        raise RuntimeError("Best CTC candidate has no valid non-blank alignment boundary")
    return CtcCandidate(
        frame=frame,
        keyword_index=keyword_index,
        scores=np.asarray(row[0], dtype=np.float32),
        top_score=float(top[0]),
        margin=float(margin[0]),
        start_frame=start_frame,
        end_frame=end_frame,
    )


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


def _onnx_intra_op_threads() -> int:
    """Respect a Slurm/OMP CPU allocation and avoid ORT oversubscription."""

    for name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return 1


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
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = _onnx_intra_op_threads()
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
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


def audio_to_fbank(audio: np.ndarray, contract: Stage1Contract) -> np.ndarray:
    """Compute the fbank matrix described by the stage-1 contract.

    The optional mean/istd values are deliberately contract data instead of
    hidden code.  Put them in the contract only when the exported ONNX graph
    expects CMVN to have already happened outside the graph.
    """

    _torch, torchaudio = _torchaudio()
    # Match wenet.dataset.processor.compute_fbank. load_audio() returns
    # normalized floating-point PCM, but WeNet scales it to the signed-16-bit
    # range before Kaldi fbank extraction.
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
    waveform = waveform * float(contract.waveform_scale)
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


def _atomic_write_npy(path: Path, value: np.ndarray) -> None:
    """Write an NPY payload atomically without NumPy changing the suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
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
    candidate_pre_margin_frames: int = 3,
    candidate_post_margin_frames: int = 0,
    max_search_frames: int | None = None,
    device: str = "cpu",
    overwrite: bool = False,
    index_offset: int = 0,
    debug_alignments: bool = False,
    competitor_beam_size: int = 16,
    competitor_token_prune: int | None = 8,
) -> dict[str, Any]:
    """Generate ragged candidate features and all stage-1 values.

    The feature bundle is intentionally all-row.  A later train step applies
    the keyword-specific stage-1 threshold, so a changed threshold never
    requires expensive stage-1 inference again.
    """

    if not records:
        raise ValueError("Cannot create a CTC-WAC feature bundle from an empty manifest")
    output_file = output_file.resolve()
    paths = feature_bundle_paths(output_file)
    contract = Stage1Contract.from_json(contract_path)
    if not overwrite and feature_bundle_valid(
        output_file,
        expected_stage1_contract_fingerprint=contract.fingerprint(),
        require_debug_alignments=debug_alignments,
    ):
        existing_summary = read_json(paths.summary)
        if bool(existing_summary.get("debug_alignment_enabled")) == debug_alignments:
            return existing_summary
    keywords = load_keywords(keywords_path, require_threshold=False)
    keyword_ids = {item.id for item in keywords}
    keyword_by_text = {item.display_text.strip().casefold(): item.id for item in keywords}
    if candidate_pre_margin_frames < 0 or candidate_post_margin_frames < 0:
        raise ValueError("candidate crop margins must be >= 0")
    if max_search_frames is not None and int(max_search_frames) < 1:
        raise ValueError("max_search_frames must be >= 1 when configured")
    if int(competitor_beam_size) < 2:
        raise ValueError("competitor_beam_size must be >= 2")
    if competitor_token_prune is not None and int(competitor_token_prune) < 1:
        raise ValueError("competitor_token_prune must be >= 1 when configured")
    stage1 = StreamingCtcStage1(model_path, contract, device=device)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = {name: _temporary_path(path) for name, path in asdict(paths).items()}
    temporary = {name: Path(value) for name, value in temporary_paths.items()}
    temporary_debug_alignments = _temporary_path(paths.debug_alignments) if debug_alignments else None
    raw_features = _temporary_path(output_file.with_name(f".{output_file.name}.candidate_features.raw"))
    rows: list[dict[str, Any]] = []
    score_rows: list[np.ndarray] = []
    top_rows: list[float] = []
    keyword_score_rows: list[float] = []
    filler_score_rows: list[float] = []
    raw_score_rows: list[float] = []
    normalized_raw_score_rows: list[float] = []
    confidence_rows: list[float] = []
    normalized_confidence_rows: list[float] = []
    segment_length_rows: list[int] = []
    margin_rows: list[float] = []
    winner_rows: list[np.ndarray] = []
    lengths: list[int] = []
    offsets = [0]
    errors: list[dict[str, Any]] = []
    invalid_alignments: list[dict[str, Any]] = []
    debug_alignment_rows: list[dict[str, Any]] = []
    expected_keyword_counts = {item.id: 0 for item in keywords}
    expected_keyword_invalid_alignment_counts = {item.id: 0 for item in keywords}
    feature_dim: int | None = None
    input_duration_seconds = 0.0
    row = 0

    def debug_alignment_row(
        *,
        record: dict[str, Any],
        source_index: int,
        expected_keyword_id: str | None,
        status: str,
        candidate: CtcCandidate | None = None,
        token_alignments: list[CtcTokenAlignment] | None = None,
        keyword_vs_filler: CtcKeywordVsFillerScore | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "source_index": source_index,
            "id": record.get("id"),
            "path": str(record.get("path")) if record.get("path") else None,
            "label": record.get("label"),
            "expected_keyword_id": expected_keyword_id,
            "status": status,
            "candidate": None,
        }
        if candidate is None:
            return result
        keyword = keywords[candidate.keyword_index]
        result["candidate"] = {
            "keyword_id": keyword.id,
            "keyword_text": keyword.display_text,
            "token_ids": list(keyword.token_ids),
            "candidate_frame": candidate.frame,
            "candidate_start_frame": candidate.start_frame,
            "candidate_end_frame": candidate.end_frame,
            "top_score": candidate.top_score,
            "margin": candidate.margin,
            "score_domain": "normalized_ctc_log_probability_per_keyword",
            "keyword_vs_filler": (
                {
                    "keyword_score": keyword_vs_filler.keyword_score,
                    "filler_score": keyword_vs_filler.filler_score,
                    "raw_score": keyword_vs_filler.raw_score,
                    "normalized_raw_score": keyword_vs_filler.normalized_raw_score,
                    "confidence": keyword_vs_filler.confidence,
                    "normalized_confidence": keyword_vs_filler.normalized_confidence,
                    "segment_length": keyword_vs_filler.segment_length,
                    "filler_token_ids": list(keyword_vs_filler.filler_token_ids),
                    "filler_method": "ctc_prefix_beam_best_nonkeyword",
                    "competitor_beam_size": int(competitor_beam_size),
                    "competitor_token_prune": (
                        int(competitor_token_prune) if competitor_token_prune is not None else None
                    ),
                }
                if keyword_vs_filler is not None
                else None
            ),
            "tokens": [
                {
                    "token_index": item.token_index,
                    "token_id": item.token_id,
                    "start_frame": item.start_frame,
                    "end_frame": item.end_frame,
                    "frame_count": item.frame_count,
                    "log_score": item.log_score,
                    "normalized_score": item.normalized_score,
                    "normalized_score_domain": "mean_log_probability_per_assigned_encoder_frame",
                }
                for item in token_alignments or []
            ],
        }
        return result

    try:
        with raw_features.open("ab") as raw_handle:
            for local_index, record in enumerate(records):
                index = int(index_offset) + local_index
                path_value = record.get("path")
                expected_keyword_id: str | None = None
                try:
                    try:
                        is_positive = int(record.get("label", 0)) == 1
                    except (TypeError, ValueError):
                        raise ValueError(f"record {index} has an invalid label {record.get('label')!r}")
                    if is_positive:
                        expected_keyword_id = _expected_keyword_id(
                            record,
                            keyword_ids=keyword_ids,
                            keyword_by_text=keyword_by_text,
                        )
                        if expected_keyword_id is None:
                            raise ValueError(
                                f"positive record {index} has no expected keyword ID and text does not match keywords"
                            )
                        expected_keyword_counts[expected_keyword_id] += 1
                    if not path_value:
                        raise ValueError("normalized manifest row has no path")
                    audio = load_audio(Path(str(path_value)), contract.sample_rate)
                    input_duration_seconds += audio.size / float(contract.sample_rate)
                    fbank = audio_to_fbank(audio, contract)
                    encoder, ctc = stage1.infer_fbank(fbank)
                    if encoder.ndim != 2 or ctc.ndim != 2:
                        raise RuntimeError("stage-1 inference did not return two time-major matrices")
                    if encoder.shape[0] != ctc.shape[0]:
                        raise RuntimeError("stage-1 encoder and CTC output frame counts differ")
                    if contract.encoder_output_size is not None and encoder.shape[1] != contract.encoder_output_size:
                        raise RuntimeError(
                            f"stage-1 encoder dimension {encoder.shape[1]} does not match contract "
                            f"encoder_output_size {contract.encoder_output_size}"
                        )
                    if contract.vocab_size is not None and ctc.shape[1] != contract.vocab_size:
                        raise RuntimeError(
                            f"stage-1 CTC vocabulary {ctc.shape[1]} does not match contract vocab_size {contract.vocab_size}"
                        )
                    if feature_dim is None:
                        feature_dim = int(encoder.shape[1])
                    elif int(encoder.shape[1]) != feature_dim:
                        raise RuntimeError(
                            f"stage-1 encoder dimension changed from {feature_dim} to {encoder.shape[1]}"
                        )
                    candidate = best_ctc_candidate(
                        ctc,
                        keywords,
                        blank_id=contract.blank_id,
                        max_search_frames=max_search_frames,
                    )
                    token_alignments = (
                        ctc_candidate_token_alignments(
                            ctc,
                            keywords[candidate.keyword_index].token_ids,
                            candidate_start_frame=candidate.start_frame,
                            candidate_end_frame=candidate.end_frame,
                            blank_id=contract.blank_id,
                        )
                        if debug_alignments
                        else None
                    )
                    crop_start = max(0, candidate.start_frame - int(candidate_pre_margin_frames))
                    crop_end = min(
                        int(encoder.shape[0]),
                        candidate.end_frame + 1 + int(candidate_post_margin_frames),
                    )
                    if crop_start >= crop_end:
                        if expected_keyword_id is not None:
                            expected_keyword_invalid_alignment_counts[expected_keyword_id] += 1
                        invalid_alignments.append(
                            {
                                "source_index": index,
                                "id": record.get("id"),
                                "path": str(path_value),
                                "reason": "empty_candidate_crop",
                                "expected_keyword_id": expected_keyword_id,
                                "candidate_start_frame": candidate.start_frame,
                                "candidate_end_frame": candidate.end_frame,
                            }
                        )
                        if debug_alignments:
                            debug_alignment_rows.append(
                                debug_alignment_row(
                                    record=record,
                                    source_index=index,
                                    expected_keyword_id=expected_keyword_id,
                                    status="empty_candidate_crop",
                                    candidate=candidate,
                                    token_alignments=token_alignments,
                                )
                            )
                        continue
                    keyword_vs_filler = ctc_keyword_vs_filler_score(
                        ctc[candidate.start_frame:candidate.end_frame + 1],
                        keywords[candidate.keyword_index].token_ids,
                        blank_id=contract.blank_id,
                        beam_size=int(competitor_beam_size),
                        token_prune=competitor_token_prune,
                    )
                    crop = np.asarray(encoder[crop_start:crop_end], dtype=np.float32)
                    crop.tofile(raw_handle)
                    candidate_length = int(crop.shape[0])
                    score_rows.append(candidate.scores)
                    top_rows.append(candidate.top_score)
                    keyword_score_rows.append(keyword_vs_filler.keyword_score)
                    filler_score_rows.append(keyword_vs_filler.filler_score)
                    raw_score_rows.append(keyword_vs_filler.raw_score)
                    normalized_raw_score_rows.append(keyword_vs_filler.normalized_raw_score)
                    confidence_rows.append(keyword_vs_filler.confidence)
                    normalized_confidence_rows.append(keyword_vs_filler.normalized_confidence)
                    segment_length_rows.append(keyword_vs_filler.segment_length)
                    margin_rows.append(candidate.margin)
                    onehot = np.zeros((len(keywords),), dtype=np.float32)
                    onehot[candidate.keyword_index] = 1.0
                    winner_rows.append(onehot)
                    lengths.append(candidate_length)
                    offsets.append(offsets[-1] + candidate_length)
                    rows.append(
                        {
                            "row": row,
                            "source_index": index,
                            "id": record.get("id"),
                            "path": str(path_value),
                            "label": record.get("label"),
                            "expected_keyword_id": expected_keyword_id,
                            "keyword_id": keywords[candidate.keyword_index].id,
                            "candidate_frame": candidate.frame,
                            "candidate_start_frame": candidate.start_frame,
                            "candidate_end_frame": candidate.end_frame,
                            "crop_start_frame": crop_start,
                            "crop_end_frame": crop_end,
                            "candidate_length_frames": candidate_length,
                            "candidate_duration_ms": candidate_length * contract.encoder_frame_shift_ms,
                            "top_score": candidate.top_score,
                            "normalized_ctc_score": candidate.top_score,
                            "keyword_score": keyword_vs_filler.keyword_score,
                            "filler_score": keyword_vs_filler.filler_score,
                            "raw_score": keyword_vs_filler.raw_score,
                            "normalized_raw_score": keyword_vs_filler.normalized_raw_score,
                            "confidence": keyword_vs_filler.confidence,
                            "normalized_confidence": keyword_vs_filler.normalized_confidence,
                            "segment_length": keyword_vs_filler.segment_length,
                            "filler_token_ids": list(keyword_vs_filler.filler_token_ids),
                            "filler_method": "ctc_prefix_beam_best_nonkeyword",
                            "competitor_beam_size": int(competitor_beam_size),
                            "competitor_token_prune": (
                                int(competitor_token_prune) if competitor_token_prune is not None else None
                            ),
                            "margin": candidate.margin,
                        }
                    )
                    if debug_alignments:
                        debug_alignment_rows.append(
                            debug_alignment_row(
                                record=record,
                                source_index=index,
                                expected_keyword_id=expected_keyword_id,
                                status="ok",
                                candidate=candidate,
                                token_alignments=token_alignments,
                                keyword_vs_filler=keyword_vs_filler,
                            )
                        )
                    row += 1
                except RuntimeError as exc:
                    # A CTC matrix with no complete keyword path is expected to
                    # be rare, but it must not produce a fake all-zero crop.
                    if "no valid non-blank alignment boundary" in str(exc):
                        if expected_keyword_id is not None:
                            expected_keyword_invalid_alignment_counts[expected_keyword_id] += 1
                        invalid_alignments.append(
                            {
                                "source_index": index,
                                "id": record.get("id"),
                                "path": str(path_value) if path_value else None,
                                "reason": "no_valid_ctc_alignment",
                                "expected_keyword_id": expected_keyword_id,
                            }
                        )
                        if debug_alignments:
                            debug_alignment_rows.append(
                                debug_alignment_row(
                                    record=record,
                                    source_index=index,
                                    expected_keyword_id=expected_keyword_id,
                                    status="no_valid_ctc_alignment",
                                )
                            )
                        continue
                    errors.append(
                        {
                            "index": index,
                            "id": record.get("id"),
                            "path": str(path_value) if path_value else None,
                            "error": repr(exc),
                        }
                    )
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
        if feature_dim is None:
            raise RuntimeError("CTC-WAC feature generation did not receive a usable stage-1 encoder output")
        if debug_alignments and len(debug_alignment_rows) != len(records):
            raise RuntimeError("CTC-WAC debug alignment log does not cover every input record")
        total_frames = int(offsets[-1])
        if total_frames:
            raw = np.memmap(raw_features, mode="r", dtype=np.float32, shape=(total_frames, feature_dim))
            feature_mmap = open_memmap(
                temporary["features"],
                mode="w+",
                dtype=np.float32,
                shape=(total_frames, feature_dim),
            )
            try:
                for start in range(0, total_frames, 65536):
                    feature_mmap[start:start + 65536] = raw[start:start + 65536]
                feature_mmap.flush()
            finally:
                del feature_mmap, raw
        else:
            _atomic_write_npy(temporary["features"], np.empty((0, feature_dim), dtype=np.float32))
        _atomic_write_npy(temporary["offsets"], np.asarray(offsets, dtype=np.int64))
        _atomic_write_npy(temporary["lengths"], np.asarray(lengths, dtype=np.int32))
        _atomic_write_npy(temporary["all_scores"], np.asarray(score_rows, dtype=np.float32).reshape(row, len(keywords)))
        _atomic_write_npy(temporary["top_score"], np.asarray(top_rows, dtype=np.float32).reshape(row, 1))
        _atomic_write_npy(temporary["keyword_score"], np.asarray(keyword_score_rows, dtype=np.float32).reshape(row, 1))
        _atomic_write_npy(temporary["filler_score"], np.asarray(filler_score_rows, dtype=np.float32).reshape(row, 1))
        _atomic_write_npy(temporary["raw_score"], np.asarray(raw_score_rows, dtype=np.float32).reshape(row, 1))
        _atomic_write_npy(
            temporary["normalized_raw_score"],
            np.asarray(normalized_raw_score_rows, dtype=np.float32).reshape(row, 1),
        )
        _atomic_write_npy(temporary["confidence"], np.asarray(confidence_rows, dtype=np.float32).reshape(row, 1))
        _atomic_write_npy(
            temporary["normalized_confidence"],
            np.asarray(normalized_confidence_rows, dtype=np.float32).reshape(row, 1),
        )
        _atomic_write_npy(
            temporary["segment_length"],
            np.asarray(segment_length_rows, dtype=np.int32).reshape(row, 1),
        )
        _atomic_write_npy(temporary["margin"], np.asarray(margin_rows, dtype=np.float32).reshape(row, 1))
        _atomic_write_npy(temporary["winner_onehot"], np.asarray(winner_rows, dtype=np.float32).reshape(row, len(keywords)))
        length_values = np.asarray(lengths, dtype=np.int64)
        summary = {
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "output_file": str(output_file),
            "feature_count": row,
            "feature_storage_shape": [total_frames, feature_dim],
            "feature_dim": feature_dim,
            "candidate_length_frames": {
                "min": int(length_values.min()) if length_values.size else None,
                "max": int(length_values.max()) if length_values.size else None,
            },
            "keyword_count": len(keywords),
            "keyword_ids": [item.id for item in keywords],
            "keyword_token_fingerprint": keyword_token_fingerprint(keywords),
            "expected_keyword_counts": expected_keyword_counts,
            "expected_keyword_invalid_alignment_counts": expected_keyword_invalid_alignment_counts,
            "stage1_contract_fingerprint": contract.fingerprint(),
            "stage1_model": str(model_path),
            "stage1_contract": str(contract_path),
            "keyword_tokens": str(keywords_path),
            "stage1_providers": stage1.providers,
            "sample_rate": contract.sample_rate,
            "score_domain": "normalized_log_probability",
            "keyword_vs_filler": {
                "filler_method": "ctc_prefix_beam_best_nonkeyword",
                "keyword_score_domain": "ctc_forward_log_probability",
                "filler_score_domain": "ctc_forward_log_probability",
                "raw_score_domain": "keyword_score_minus_filler_score",
                "normalized_raw_score_domain": "raw_score_per_selected_candidate_encoder_frame",
                "confidence_domain": "sigmoid(keyword_score_minus_filler_score)",
                "normalized_confidence_domain": "sigmoid(raw_score_per_selected_candidate_encoder_frame)",
                "competitor_beam_size": int(competitor_beam_size),
                "competitor_token_prune": (
                    int(competitor_token_prune) if competitor_token_prune is not None else None
                ),
            },
            "encoder_frame_shift_ms": contract.encoder_frame_shift_ms,
            "input_count": len(records),
            "input_duration_seconds": input_duration_seconds,
            "debug_alignment_enabled": debug_alignments,
            "debug_alignment_jsonl": str(paths.debug_alignments) if debug_alignments else None,
            "debug_alignment_rows": len(debug_alignment_rows) if debug_alignments else 0,
            "invalid_alignment_rows": len(invalid_alignments),
            "invalid_alignments": invalid_alignments[:50],
            "candidate_pre_margin_frames": int(candidate_pre_margin_frames),
            "candidate_post_margin_frames": int(candidate_post_margin_frames),
            "max_search_frames": int(max_search_frames) if max_search_frames is not None else None,
            "max_search_seconds": (
                float(max_search_frames) * contract.encoder_frame_shift_ms / 1000.0
                if max_search_frames is not None
                else None
            ),
            "index_offset": int(index_offset),
            "error_count": 0,
            "errors": [],
            "stage1_gate_applied": False,
        }
        _atomic_write_rows(temporary["rows"], rows)
        if temporary_debug_alignments is not None:
            write_jsonl(temporary_debug_alignments, debug_alignment_rows)
        _atomic_write_json(temporary["summary"], summary)
        for name, destination in asdict(paths).items():
            os.replace(temporary[name], Path(destination))
        if temporary_debug_alignments is not None:
            os.replace(temporary_debug_alignments, paths.debug_alignments)
        else:
            paths.debug_alignments.unlink(missing_ok=True)
        return summary
    finally:
        # A failed extraction never overwrites a last known-good bundle.
        for path in temporary.values():
            path.unlink(missing_ok=True)
        if temporary_debug_alignments is not None:
            temporary_debug_alignments.unlink(missing_ok=True)
        raw_features.unlink(missing_ok=True)


@dataclass
class CtcWacFeatureBlock:
    """Memory-mapped feature bundle plus rows retained by the stage-1 gate."""

    name: str
    label: int
    split: str
    paths: FeatureBundlePaths
    features: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
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
        offsets = np.load(paths.offsets, mmap_mode="r")
        lengths = np.load(paths.lengths, mmap_mode="r")
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
            offsets=offsets,
            lengths=lengths,
            all_scores=scores,
            top_score=top,
            margin=margin,
            winner_onehot_values=onehot,
            retained_indices=retained_stage1_indices(scores, keywords),
        )

    @property
    def input_count(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def retained_count(self) -> int:
        return int(self.retained_indices.shape[0])

    @property
    def feature_shape(self) -> tuple[int, int]:
        return (int(np.max(self.lengths)) if self.lengths.size else 0, int(self.features.shape[1]))

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def candidate(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.features[start:end], dtype=np.float32)

    def retained_bucket_indices(self, bucket_width_frames: int) -> dict[int, np.ndarray]:
        if bucket_width_frames < 1:
            raise ValueError("bucket_width_frames must be >= 1")
        values = np.asarray(self.lengths[self.retained_indices], dtype=np.int64)
        buckets: dict[int, list[int]] = {}
        for index, length in zip(self.retained_indices.tolist(), values.tolist()):
            bucket = (int(length) - 1) // int(bucket_width_frames)
            buckets.setdefault(bucket, []).append(int(index))
        return {key: np.asarray(value, dtype=np.int64) for key, value in buckets.items()}

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.size < 1:
            raise ValueError("Cannot construct an empty CTC-WAC batch")
        lengths = np.asarray(self.lengths[selected], dtype=np.int64)
        max_frames = int(lengths.max())
        values = np.zeros((selected.shape[0], max_frames, self.feature_dim), dtype=np.float32)
        mask = np.zeros((selected.shape[0], max_frames), dtype=np.float32)
        for row_index, source_index in enumerate(selected.tolist()):
            candidate = self.candidate(int(source_index))
            values[row_index, :candidate.shape[0]] = candidate
            mask[row_index, :candidate.shape[0]] = 1.0
        return (
            values,
            mask,
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
            "min_retained_frames": int(np.min(self.lengths[self.retained_indices])) if self.retained_count else 0,
            "max_retained_frames": int(np.max(self.lengths[self.retained_indices])) if self.retained_count else 0,
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
        frame_mask: torch.Tensor,
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
            if frame_mask.shape != encoder_features.shape[:2]:
                raise ValueError("frame_mask must have shape [batch, frames]")
            if winner_onehot_values.ndim != 2 or winner_onehot_values.shape[-1] != self.keyword_count:
                raise ValueError("winner_onehot has the wrong keyword dimension")
        frame_values = self.frame_net(self.frame_norm(encoder_features))
        mask = frame_mask.to(dtype=frame_values.dtype).unsqueeze(-1)
        pooled = (frame_values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        normalized_score = (top_score.reshape(-1, 1) - self.score_mean) / self.score_std
        normalized_margin = (margin.reshape(-1, 1) - self.margin_mean) / self.margin_std
        context = torch.cat([pooled, normalized_score, normalized_margin, winner_onehot_values], dim=1)
        return self.decoder(self.context_norm(context))

    def forward(
        self,
        encoder_features: torch.Tensor,
        frame_mask: torch.Tensor,
        top_score: torch.Tensor,
        margin: torch.Tensor,
        winner_onehot_values: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward_logits(encoder_features, frame_mask, top_score, margin, winner_onehot_values)
        )


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
    expected = {"encoder_features", "frame_mask", "top_score", "margin", "winner_onehot"}
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
