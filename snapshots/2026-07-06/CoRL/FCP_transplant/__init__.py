"""FCP (Fictitious Co-Play) baseline for Risked Overcooked.

This package is a thin orchestration overlay on top of ``transplant`` (which
already ports the HSP MAPPO + population machinery to the Risked Overcooked
environment).  It adds only the FCP-specific pipeline:

  Stage 1  self-play population of N seeds  (reuses transplant's runner.run())
  Extract  init / mid / final checkpoint per seed  -> 3*N frozen partners
  Stage 2  best-response adaptive agent vs the frozen pool (uniform sampling)

No code under ``HSP/`` or ``transplant/`` is modified.
"""
