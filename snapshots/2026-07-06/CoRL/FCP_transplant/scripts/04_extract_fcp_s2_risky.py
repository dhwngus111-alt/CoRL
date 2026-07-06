#!/usr/bin/env python
"""[04] Extract the trained FCP Stage-2 adaptive checkpoint + write eval YAML.

Thin CLI wrapper over ``FCP_transplant.extract_fcp_s2``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FCP_transplant.extract_fcp_s2 import main  # noqa: E402

if __name__ == "__main__":
    main()
