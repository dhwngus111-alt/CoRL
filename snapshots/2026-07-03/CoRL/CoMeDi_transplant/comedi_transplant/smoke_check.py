"""Small import/config smoke check for CoMeDi_transplant."""

from __future__ import annotations

import argparse

from comedi_transplant.bootstrap import PATHS, ensure_paths
from comedi_transplant.policy_configs import DEFAULT_LAYOUTS, build_policy_configs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", action="append", dest="layouts")
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--build-policy-configs", action="store_true")
    args = parser.parse_args(argv)

    paths = ensure_paths()
    layouts = tuple(args.layouts or DEFAULT_LAYOUTS)
    print("CoMeDi_transplant paths:")
    for key, value in paths.items():
        print(f"  {key}: {value}")

    if args.build_policy_configs:
        for layout in layouts:
            build_policy_configs(layout, episode_length=args.episode_length)
            print(f"built policy configs for {layout}")

    import hsp  # noqa: F401
    import risky_overcooked_py  # noqa: F401
    import transplant  # noqa: F401

    print("smoke_check ok")


if __name__ == "__main__":
    main()

