"""Path and environment bootstrap for the transplant package.

This module is intentionally the only place that wires the two original
repositories into Python import resolution.  It does not modify either source
tree; it only adds their import roots to sys.path and points HSP's POLICY_POOL
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

TRANSPLANT_ROOT = Path(__file__).resolve().parent
CORL_ROOT = TRANSPLANT_ROOT.parent
HSP_ROOT = Path(os.environ.get("HSP_ROOT", CORL_ROOT / "HSP")).resolve()
RISKY_ROOT = Path(os.environ.get("RISKY_ROOT", CORL_ROOT / "risky")).resolve()
RISKY_SRC = RISKY_ROOT / "src"
COMPAT_ROOT = TRANSPLANT_ROOT / "compat"
POLICY_POOL_ROOT = Path(
    os.environ.get("POLICY_POOL", TRANSPLANT_ROOT / "policy_pool")
).resolve()


def ensure_paths() -> dict[str, Path]:
    """Make HSP and Risky Overcooked importable from transplant scripts."""

    os.environ.setdefault("POLICY_POOL", str(POLICY_POOL_ROOT))
    for path in (COMPAT_ROOT, CORL_ROOT, HSP_ROOT, RISKY_SRC):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return {
        "transplant_root": TRANSPLANT_ROOT,
        "corl_root": CORL_ROOT,
        "hsp_root": HSP_ROOT,
        "risky_root": RISKY_ROOT,
        "risky_src": RISKY_SRC,
        "compat_root": COMPAT_ROOT,
        "policy_pool_root": POLICY_POOL_ROOT,
    }


PATHS = ensure_paths()
