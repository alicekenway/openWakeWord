"""Persistent, content-aware completion checkpoints for pipeline stages."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from .artifacts import file_signature, hash_payload, read_json, write_json


def step_slug(step: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", step).replace(".", "_")


class CheckpointManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def complete_path(self, step: str) -> Path:
        return self.root / f"{step_slug(step)}.done.json"

    def failure_path(self, step: str) -> Path:
        return self.root / f"{step_slug(step)}.failed.json"

    def read_complete(self, step: str) -> dict[str, Any] | None:
        path = self.complete_path(step)
        if not path.exists():
            return None
        try:
            state = read_json(path)
        except Exception:
            return None
        return state if isinstance(state, dict) else None

    def completion_digest(self, step: str) -> str | None:
        state = self.read_complete(step)
        return hash_payload(state) if state is not None else None

    def is_complete(
        self,
        step: str,
        *,
        fingerprint: str,
        input_signature: dict[str, Any],
        outputs: list[Path],
        output_validator: Callable[[], bool],
    ) -> tuple[bool, str]:
        state = self.read_complete(step)
        if state is None:
            return False, "no completion checkpoint"
        if state.get("fingerprint") != fingerprint:
            return False, "configuration changed"
        if state.get("input_signature") != input_signature:
            return False, "input or dependency changed"
        if not output_validator():
            return False, "declared outputs are incomplete"
        current_outputs = [file_signature(path) for path in outputs]
        if state.get("outputs") != current_outputs:
            return False, "declared outputs changed"
        return True, "complete"

    def mark_complete(
        self,
        step: str,
        *,
        fingerprint: str,
        input_signature: dict[str, Any],
        outputs: list[Path],
        result: dict[str, Any],
    ) -> None:
        state = {
            "step": step,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fingerprint": fingerprint,
            "input_signature": input_signature,
            "outputs": [file_signature(path) for path in outputs],
            "result": result,
        }
        write_json(self.complete_path(step), state)
        failed = self.failure_path(step)
        if failed.exists():
            failed.unlink()

    def mark_failed(self, step: str, error: BaseException) -> None:
        write_json(
            self.failure_path(step),
            {
                "step": step,
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": repr(error),
            },
        )
