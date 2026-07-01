"""Small tensorboardX fallback used by transplant when tensorboardX is absent.

The original HSP code imports ``SummaryWriter`` from tensorboardX.  Some local
envs here have the rest of HSP's dependencies but not tensorboardX itself, so
this module provides the tiny subset HSP calls during local runs.
"""

from __future__ import annotations

import json
from pathlib import Path


try:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
except Exception:  # pragma: no cover - dependency-dependent fallback
    _TorchSummaryWriter = None


class SummaryWriter:
    def __init__(self, logdir=None, *args, **kwargs):
        self.logdir = Path(logdir or ".")
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        if _TorchSummaryWriter is not None:
            try:
                self._writer = _TorchSummaryWriter(str(self.logdir), *args, **kwargs)
            except Exception:
                self._writer = None

    def add_scalars(self, *args, **kwargs):
        if self._writer is not None:
            return self._writer.add_scalars(*args, **kwargs)
        return None

    def add_scalar(self, *args, **kwargs):
        if self._writer is not None:
            return self._writer.add_scalar(*args, **kwargs)
        return None

    def log(self, info, step=None):
        if isinstance(info, dict):
            for key, value in info.items():
                self.add_scalar(key, value, step)

    def export_scalars_to_json(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self._writer is not None and hasattr(self._writer, "export_scalars_to_json"):
            return self._writer.export_scalars_to_json(str(output))
        output.write_text(json.dumps({}))
        return None

    def flush(self):
        if self._writer is not None:
            return self._writer.flush()
        return None

    def close(self):
        if self._writer is not None:
            return self._writer.close()
        return None

