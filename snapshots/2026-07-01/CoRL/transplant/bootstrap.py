"""Path and environment bootstrap for the transplant package.

This module is intentionally the only place that wires the HSP source tree and
the active Risked Overcooked environment into Python import resolution.  It
does not modify either source tree; it only adds their import roots to sys.path
and points HSP's POLICY_POOL
lookup at transplant/policy_pool by default.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

TRANSPLANT_CODE_ROOT = Path(__file__).resolve().parent
TRANSPLANT_ROOT = Path(os.environ.get("TRANSPLANT_OUTPUT_ROOT", TRANSPLANT_CODE_ROOT)).resolve()
CORL_ROOT = TRANSPLANT_CODE_ROOT.parent
HSP_ROOT = Path(os.environ.get("HSP_ROOT", CORL_ROOT / "HSP")).resolve()
RISKED_ROOT = Path(os.environ.get("RISKED_ROOT", CORL_ROOT / "risked_overcooked")).resolve()
RISKED_SRC = RISKED_ROOT / "src"
# Backwards-compatible aliases for older transplant code paths.
RISKY_ROOT = RISKED_ROOT
RISKY_SRC = RISKED_SRC
COMPAT_ROOT = TRANSPLANT_CODE_ROOT / "compat"
POLICY_POOL_ROOT = Path(
    os.environ.get("POLICY_POOL", TRANSPLANT_ROOT / "policy_pool")
).resolve()


def ensure_paths() -> dict[str, Path]:
    """Make HSP and Risked Overcooked importable from transplant scripts."""

    os.environ.setdefault("POLICY_POOL", str(POLICY_POOL_ROOT))
    old_risky_src = str((CORL_ROOT / "risky" / "src").resolve())
    sys.path[:] = [path for path in sys.path if path != old_risky_src]
    for path in (COMPAT_ROOT, CORL_ROOT, HSP_ROOT, RISKED_SRC):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return {
        "transplant_root": TRANSPLANT_ROOT,
        "transplant_code_root": TRANSPLANT_CODE_ROOT,
        "corl_root": CORL_ROOT,
        "hsp_root": HSP_ROOT,
        "risked_root": RISKED_ROOT,
        "risked_src": RISKED_SRC,
        "risky_root": RISKY_ROOT,
        "risky_src": RISKY_SRC,
        "compat_root": COMPAT_ROOT,
        "policy_pool_root": POLICY_POOL_ROOT,
    }


PATHS = ensure_paths()
