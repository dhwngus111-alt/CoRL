"""Path and output-root bootstrap for the FCP_transplant overlay.

FCP_transplant reuses ``transplant``'s adapter/runner/common/train drivers
verbatim.  Those modules resolve their *output* locations (results dir and
policy pool) from environment variables read at import time by
``transplant.bootstrap``:

    TRANSPLANT_OUTPUT_ROOT  ->  results/ root
    POLICY_POOL             ->  policy_pool/ root

To keep FCP artifacts separate from the HSP transplant run, we default those
env vars to this directory *before* importing ``transplant.bootstrap``.  The
shell pipeline (``common.sh``) exports the same variables so both direct
``python`` invocations and ``python -m transplant.*`` calls agree.

This module does not modify the HSP or transplant source trees; it only wires
import paths and points the output roots at ``FCP_transplant/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FCP_CODE_ROOT = Path(__file__).resolve().parent
CORL_ROOT = FCP_CODE_ROOT.parent

# Make the sibling ``transplant`` package importable before we touch it.
if str(CORL_ROOT) not in sys.path:
    sys.path.insert(0, str(CORL_ROOT))

# Default FCP outputs to this directory unless the caller overrode them.
FCP_OUTPUT_ROOT = Path(os.environ.setdefault("TRANSPLANT_OUTPUT_ROOT", str(FCP_CODE_ROOT))).resolve()
os.environ.setdefault("POLICY_POOL", str(FCP_OUTPUT_ROOT / "policy_pool"))

# transplant.bootstrap reads the env vars above at import time and wires
# hsp/, risked_overcooked/src, and transplant/compat onto sys.path.
from transplant.bootstrap import (  # noqa: E402
    POLICY_POOL_ROOT,
    TRANSPLANT_ROOT as OUTPUT_ROOT,
    ensure_paths as _ensure_transplant_paths,
)


def ensure_paths() -> dict:
    """Wire transplant/hsp/risked import paths and return resolved roots."""
    paths = _ensure_transplant_paths()
    paths["fcp_code_root"] = FCP_CODE_ROOT
    paths["fcp_output_root"] = OUTPUT_ROOT
    paths["policy_pool_root"] = POLICY_POOL_ROOT
    return paths


PATHS = ensure_paths()
