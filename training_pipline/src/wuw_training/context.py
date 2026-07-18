"""Runtime context passed to every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import IniConfig


@dataclass
class StageContext:
    config: IniConfig
    step: str
    section: dict[str, str]
    experiment_dir: Path
    work_dir: Path
    force: bool = False
    execution_role: str = "controller"

    @property
    def section_name(self) -> str:
        return self.step

    @property
    def is_slurm_worker(self) -> bool:
        return self.execution_role == "slurm_worker"
