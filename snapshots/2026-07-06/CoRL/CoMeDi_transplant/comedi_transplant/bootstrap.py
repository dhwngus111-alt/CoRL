"""Bootstrap paths for CoMeDi_transplant without modifying source repos."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


COMEDI_TRANSPLANT_ROOT = Path(__file__).resolve().parents[1]
CORL_ROOT = COMEDI_TRANSPLANT_ROOT.parent
HSP_ROOT = Path(os.environ.get("HSP_ROOT", CORL_ROOT / "HSP")).resolve()
RISKED_ROOT = Path(os.environ.get("RISKED_ROOT", CORL_ROOT / "risked_overcooked")).resolve()
POLICY_POOL_ROOT = Path(
    os.environ.get("POLICY_POOL", COMEDI_TRANSPLANT_ROOT / "policy_pool")
).resolve()


def ensure_paths() -> dict[str, Path]:
    """Make transplant, HSP, and Risky Overcooked importable."""

    os.environ.setdefault("TRANSPLANT_OUTPUT_ROOT", str(COMEDI_TRANSPLANT_ROOT))
    os.environ.setdefault("POLICY_POOL", str(POLICY_POOL_ROOT))
    os.environ.setdefault("HSP_ROOT", str(HSP_ROOT))
    os.environ.setdefault("RISKED_ROOT", str(RISKED_ROOT))
    os.environ.setdefault("RISKY_ROOT", str(RISKED_ROOT))

    for path in (COMEDI_TRANSPLANT_ROOT, CORL_ROOT, HSP_ROOT, RISKED_ROOT / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # transplant.bootstrap reads env vars at import time. Reload only if some
    # earlier import captured a different output root in this Python process.
    if "transplant.bootstrap" in sys.modules:
        import transplant.bootstrap as transplant_bootstrap

        if Path(transplant_bootstrap.TRANSPLANT_ROOT).resolve() != COMEDI_TRANSPLANT_ROOT:
            transplant_bootstrap = importlib.reload(transplant_bootstrap)
    else:
        import transplant.bootstrap as transplant_bootstrap

    transplant_bootstrap.ensure_paths()
    return {
        "comedi_transplant_root": COMEDI_TRANSPLANT_ROOT,
        "corl_root": CORL_ROOT,
        "hsp_root": HSP_ROOT,
        "risked_root": RISKED_ROOT,
        "policy_pool_root": POLICY_POOL_ROOT,
    }


PATHS = ensure_paths()

