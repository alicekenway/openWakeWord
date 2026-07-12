"""Access the legacy helper module without duplicating its audio primitives."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def get_legacy_module() -> ModuleType:
    expected = Path(__file__).resolve().parents[1] / "wuw_pipeline.py"
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if isinstance(main_module, ModuleType) and main_file and Path(main_file).resolve() == expected:
        return main_module
    return importlib.import_module("wuw_pipeline")
