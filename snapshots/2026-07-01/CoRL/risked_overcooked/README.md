# Risky Overcooked (environment)

Cooperative multi-agent Overcooked with **water-slip risk** and **subgoal buttons**
(pressing a subgoal disables nearby water for `subgoal_disable_steps`). Environment
only — import and plug into any algorithm.

## Install
```bash
pip install -e .
```

## Use
```python
from risky_overcooked_py.mdp.overcooked_mdp import OvercookedGridworld
from risky_overcooked_py.mdp.overcooked_env import OvercookedEnv

mdp = OvercookedGridworld.from_layout_name('risky_spiral_subgoal', neglect_boarders=True)
env = OvercookedEnv.from_mdp(mdp, horizon=200)
env.reset()
```

## Layouts (subgoal)
`risky_spiral_subgoal`, `risky_tree_subgoal`, `risky_shortcuts_subgoal`,
`risky_mixed_coordination_subgoal` (+ others under `data/layouts/`).
