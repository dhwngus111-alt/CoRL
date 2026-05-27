"""HSP-style hidden utility schema for Risky Multipath.

The Risky environment exposes raw EVENT_TYPES.  HSP S1 hidden utility uses the
original Overcooked category order, with Risky-specific risk/handoff events
appended for multipath.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transplant.bootstrap import ensure_paths

ensure_paths()

from risky_overcooked_py.mdp.overcooked_mdp import EVENT_TYPES  # noqa: E402


HSP_CORE_CATEGORY_KEYS = (
    "put_onion_on_X",
    "put_tomato_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_tomato_from_X",
    "pickup_tomato_from_T",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP",
    "SOUP_PICKUP",
    "PLACEMENT_IN_POT",
    "viable_placement",
    "optimal_placement",
    "catastrophic_placement",
    "useless_placement",
    "potting_onion",
    "potting_tomato",
    "delivery",
)

HSP_MANY_ORDERS_CORE_W0_SPEC = (
    "0",
    "0",
    "0",
    "0",
    "0",
    "r[-5:5:3]",
    "0",
    "r[-5:5:3]",
    "0",
    "r[0:5:2]",
    "0",
    "0",
    "r[-5:5:3]",
    "0",
    "r[-10:10:3]",
    "r[-10:0:2]",
    "r[0:10:2]",
    "0",
    "r[-3:3:3]",
    "r[-3:3:3]",
    "r[-10:0:2]",
)

# Risky env에서 hidden utility selection 대상으로 추가한 것들.
# Slip events are stochastic outcomes, so they stay out of hidden utility.
RISKY_RISKED_EVENT_KEYS = (
    "tomato_risked",
    "onion_risked",
    "dish_risked",
    "soup_risked",
)

RISKY_HANDOFF_EVENT_KEYS = (
    "tomato_handoff",
    "onion_handoff",
    "dish_handoff",
    "soup_handoff",
)

RISKY_MULTIPATH_EVENT_KEYS = RISKY_RISKED_EVENT_KEYS + RISKY_HANDOFF_EVENT_KEYS

HIDDEN_UTILITY_KEYS = HSP_CORE_CATEGORY_KEYS + RISKY_MULTIPATH_EVENT_KEYS
RISKY_EVENT_KEYS = tuple(EVENT_TYPES)

RISKY_TO_HSP_CATEGORY = {
    "onion_drop": ("put_onion_on_X",),
    "tomato_drop": ("put_tomato_on_X",),
    "dish_drop": ("put_dish_on_X",),
    "soup_drop": ("put_soup_on_X",),
    "onion_pickup": ("pickup_onion_from_O",),
    "tomato_pickup": ("pickup_tomato_from_T",),
    "dish_pickup": ("pickup_dish_from_D",),
    "soup_pickup": ("pickup_soup_from_X", "SOUP_PICKUP"),
    "useful_dish_pickup": ("USEFUL_DISH_PICKUP",),
    "potting_onion": ("PLACEMENT_IN_POT", "potting_onion"),
    "potting_tomato": ("PLACEMENT_IN_POT", "potting_tomato"),
    "optimal_onion_potting": ("optimal_placement",),
    "optimal_tomato_potting": ("optimal_placement",),
    "viable_onion_potting": ("viable_placement",),
    "viable_tomato_potting": ("viable_placement",),
    "catastrophic_onion_potting": ("catastrophic_placement",),
    "catastrophic_tomato_potting": ("catastrophic_placement",),
    "useless_onion_potting": ("useless_placement",),
    "useless_tomato_potting": ("useless_placement",),
    "soup_delivery": ("delivery",),
    **{key: (key,) for key in RISKY_MULTIPATH_EVENT_KEYS}, # risky env에서의 추가 event들 매핑
}


def _validate_schema() -> None:
    missing_raw = sorted(set(RISKY_TO_HSP_CATEGORY) - set(RISKY_EVENT_KEYS))
    if missing_raw:
        raise RuntimeError(f"Risky EVENT_TYPES missing mapped raw events: {missing_raw}")

    missing_categories = sorted(
        {
            category
            for categories in RISKY_TO_HSP_CATEGORY.values()
            for category in categories
            if category not in HIDDEN_UTILITY_KEYS
        }
    )
    if missing_categories:
        raise RuntimeError(f"Unknown HSP hidden utility categories: {missing_categories}")

    if len(set(HIDDEN_UTILITY_KEYS)) != len(HIDDEN_UTILITY_KEYS):
        raise RuntimeError("Duplicate HSP hidden utility keys.")

    if len(HSP_MANY_ORDERS_CORE_W0_SPEC) != len(HSP_CORE_CATEGORY_KEYS):
        raise RuntimeError(
            "HSP many_orders core w0 spec must align with HSP core category keys."
        )


_validate_schema()


def ordered_zero_category_dict() -> OrderedDict[str, float]:
    return OrderedDict((key, 0.0) for key in HIDDEN_UTILITY_KEYS)


def event_flag(event_infos: dict, risky_key: str, agent_idx: int) -> float:
    values = event_infos.get(risky_key)
    if values is None:
        return 0.0
    return float(bool(values[agent_idx]))


def risky_events_to_hsp_category_info(event_infos: dict) -> list[OrderedDict[str, float]]:
    by_agent = [ordered_zero_category_dict(), ordered_zero_category_dict()]
    for risky_key, categories in RISKY_TO_HSP_CATEGORY.items():
        for agent_idx in range(2):
            occurred = event_flag(event_infos, risky_key, agent_idx)
            if not occurred:
                continue
            for category in categories:
                by_agent[agent_idx][category] += occurred
    return by_agent


def build_hsp_w0_spec(
    core_spec: str | None = None,
    risky_spec: str | None = None,
    risked_spec: str | None = None,
    handoff_spec: str | None = None,
    sparse_weight: str = "r[0:1:2]",
) -> str:
    if core_spec is None:
        specs = list(HSP_MANY_ORDERS_CORE_W0_SPEC)
    else:
        specs = [core_spec] * len(HSP_CORE_CATEGORY_KEYS)
    risked_spec = (
        risked_spec if risked_spec is not None else risky_spec or "r[-5:5:3]"
    )
    handoff_spec = (
        handoff_spec if handoff_spec is not None else risky_spec or "r[0:5:2]"
    )
    specs.extend([risked_spec] * len(RISKY_RISKED_EVENT_KEYS))
    specs.extend([handoff_spec] * len(RISKY_HANDOFF_EVENT_KEYS))
    return ",".join(specs + [sparse_weight])


def build_hsp_w1_spec(sparse_weight: str = "1") -> str:
    return ",".join(["0"] * len(HIDDEN_UTILITY_KEYS) + [sparse_weight])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["w0", "w1"], required=True)
    parser.add_argument("--core-spec", default=None)
    parser.add_argument("--risky-spec", default=None)
    parser.add_argument("--risked-spec", default=None)
    parser.add_argument("--handoff-spec", default=None)
    parser.add_argument("--sparse-weight", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.kind == "w0":
        print(
            build_hsp_w0_spec(
                core_spec=args.core_spec,
                risky_spec=args.risky_spec,
                risked_spec=args.risked_spec,
                handoff_spec=args.handoff_spec,
                sparse_weight=args.sparse_weight or "r[0:1:2]",
            )
        )
    else:
        print(build_hsp_w1_spec(sparse_weight=args.sparse_weight or "1"))


if __name__ == "__main__":
    main()
