"""Small Silero VAD adapter for candidate-level test gating."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


VAD_SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 480
VAD_FRAME_SECONDS = VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE


class VadScorer:
    """Score one independent audio clip and query candidate intervals."""

    def __init__(self, model_path: Path, *, threads: int = 1) -> None:
        if threads < 1:
            raise ValueError("VAD threads must be >= 1")
        # Keep the dependency lazy so VAD-disabled testing does not import ORT
        # through openWakeWord's VAD module or require a model resource.
        from openwakeword.vad import VAD

        self.model_path = Path(model_path).resolve()
        self.model: Any = VAD(model_path=str(self.model_path), n_threads=threads)

    def frame_scores(self, audio: np.ndarray) -> np.ndarray:
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        clipped = np.clip(values, -1.0, 1.0)
        pcm = np.rint(clipped * 32767.0).astype(np.int16)
        remainder = pcm.size % VAD_FRAME_SAMPLES
        if remainder:
            pcm = np.pad(pcm, (0, VAD_FRAME_SAMPLES - remainder))
        self.model.reset_states()
        scores = [
            float(self.model.predict(pcm[start:start + VAD_FRAME_SAMPLES], frame_size=VAD_FRAME_SAMPLES))
            for start in range(0, pcm.size, VAD_FRAME_SAMPLES)
        ]
        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def interval_score(
        scores: np.ndarray,
        *,
        start_seconds: float,
        end_seconds: float,
        audio_seconds: float,
        padding_ms: float,
    ) -> tuple[float, float, float]:
        padding_seconds = float(padding_ms) / 1000.0
        start = max(0.0, float(start_seconds) - padding_seconds)
        end = min(float(audio_seconds), float(end_seconds) + padding_seconds)
        if end < start:
            end = start
        if scores.size == 0 or end <= start:
            return 0.0, start, end
        first = max(0, int(math.floor(start / VAD_FRAME_SECONDS)))
        stop = min(int(scores.size), int(math.ceil(end / VAD_FRAME_SECONDS)))
        if stop <= first:
            return 0.0, start, end
        return float(np.max(scores[first:stop])), start, end

