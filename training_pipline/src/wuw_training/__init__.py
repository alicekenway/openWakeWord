"""INI-driven orchestration for the local openWakeWord training workflow.

The package deliberately keeps pipeline orchestration separate from the legacy
``wuw_pipeline.py`` command module.  Low-level audio helpers are reused where
appropriate, while configuration, checkpoints, and stage dispatch live here.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path


OPENWAKEWORD_ROOT = Path(__file__).resolve().parents[3]
if str(OPENWAKEWORD_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENWAKEWORD_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wuw_mpl_config")

# The acoustics dependency imported by openwakeword.train still imports the
# deprecated SciPy name at module import time. Keep the compatibility patch in
# the modular package as well as the legacy command entrypoint.
try:
    import scipy.special as scipy_special

    if not hasattr(scipy_special, "sph_harm") and hasattr(scipy_special, "sph_harm_y"):
        def _sph_harm_compat(m: int, n: int, theta: object, phi: object) -> object:
            return scipy_special.sph_harm_y(n, m, phi, theta)

        scipy_special.sph_harm = _sph_harm_compat  # type: ignore[attr-defined]
except Exception:
    pass


PIPELINE_VERSION = "1"
