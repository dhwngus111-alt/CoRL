#!/usr/bin/env python
"""[02] Extract FCP Stage-1 self-play checkpoints into the FCP policy pool.

Thin CLI wrapper over ``FCP_transplant.extract_fcp_s1`` so the numbered pipeline
mirrors ``transplant``.  Run by file path (leading-digit module names are not
importable via ``-m``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FCP_transplant.extract_fcp_s1 import main  # noqa: E402

if __name__ == "__main__":
    main()
