"""WAC-compatible FIR loading and deterministic waveform convolution."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ConfigurationError


@dataclass(frozen=True)
class FirFilter:
    """One validated WAC FIR filter."""

    path: Path
    coefficients: np.ndarray
    sample_rate: int

    @property
    def taps(self) -> int:
        return int(self.coefficients.size)


def fir_paths_from_list(list_path: Path) -> list[Path]:
    """Read a FIR list, resolving relative entries beside the list file."""

    try:
        lines = list_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"Could not read FIR list {list_path}: {exc}") from exc

    paths: list[Path] = []
    for line_number, raw_line in enumerate(lines, start=1):
        text = raw_line.split("#", 1)[0].strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = list_path.parent / path
        resolved = path.resolve()
        if not resolved.is_file():
            raise ConfigurationError(
                f"FIR file from {list_path}:{line_number} does not exist: {resolved}"
            )
        paths.append(resolved)
    if not paths:
        raise ConfigurationError(f"FIR list is empty: {list_path}")
    return paths


def load_wac_fir(path: Path, *, expected_sample_rate: int) -> FirFilter:
    """Load ``int32 taps + float32 coefficients + int32 Fs`` WAC data."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"Could not read FIR file {path}: {exc}") from exc
    if len(payload) < 12:
        raise ConfigurationError(f"FIR file is too short: {path}")

    (tap_count,) = struct.unpack_from("<i", payload, 0)
    if tap_count < 1:
        raise ConfigurationError(f"FIR file has invalid tap count {tap_count}: {path}")
    expected_bytes = 4 + tap_count * 4 + 4
    if len(payload) != expected_bytes:
        raise ConfigurationError(
            f"FIR file size does not match its {tap_count} taps "
            f"(expected {expected_bytes} bytes, found {len(payload)}): {path}"
        )

    coefficients = np.frombuffer(payload, dtype="<f4", count=tap_count, offset=4).copy()
    if not np.isfinite(coefficients).all():
        raise ConfigurationError(f"FIR file contains non-finite coefficients: {path}")
    if not np.any(coefficients):
        raise ConfigurationError(f"FIR file contains only zero coefficients: {path}")

    (sample_rate,) = struct.unpack_from("<i", payload, 4 + tap_count * 4)
    if sample_rate != expected_sample_rate:
        raise ConfigurationError(
            f"FIR sample rate is {sample_rate}, but augmentation uses "
            f"{expected_sample_rate}: {path}"
        )
    return FirFilter(path=path, coefficients=coefficients, sample_rate=sample_rate)


def load_fir_bank(list_path: Path, *, expected_sample_rate: int) -> list[FirFilter]:
    return [
        load_wac_fir(path, expected_sample_rate=expected_sample_rate)
        for path in fir_paths_from_list(list_path)
    ]


def fir_bank_signature(list_path: Path, filters: list[FirFilter]) -> str:
    """Fingerprint the list and validated coefficient values."""

    digest = hashlib.sha256()
    digest.update(list_path.resolve().read_bytes())
    for item in filters:
        digest.update(str(item.path).encode("utf-8"))
        digest.update(struct.pack("<i", item.sample_rate))
        digest.update(item.coefficients.tobytes())
    return digest.hexdigest()


def apply_fir(
    audio: torch.Tensor,
    fir_filter: FirFilter,
    *,
    peak_dbfs: float = -10.0,
) -> torch.Tensor:
    """Apply causal FIR convolution and preserve the input sample count."""

    if audio.ndim != 1:
        raise ValueError(f"FIR augmentation expects mono 1-D audio, got shape {tuple(audio.shape)}")
    if audio.numel() < 1:
        raise ValueError("FIR augmentation cannot process empty audio")

    values = audio.detach().cpu().to(torch.float32)
    coefficients = torch.from_numpy(fir_filter.coefficients)
    full_samples = int(values.numel() + coefficients.numel() - 1)
    fft_samples = 1 << (full_samples - 1).bit_length()
    reverberated = torch.fft.irfft(
        torch.fft.rfft(values, n=fft_samples)
        * torch.fft.rfft(coefficients, n=fft_samples),
        n=fft_samples,
    )[: values.numel()].to(torch.float32)
    peak = float(reverberated.abs().max().item())
    if peak > 1e-12:
        target_peak = 10.0 ** (float(peak_dbfs) / 20.0)
        reverberated *= target_peak / peak
    return reverberated
