"""Minimal stub for environments without icecream installed."""


def ic(*args, **kwargs):
    if len(args) == 0:
        return None
    if len(args) == 1:
        return args[0]
    return args

