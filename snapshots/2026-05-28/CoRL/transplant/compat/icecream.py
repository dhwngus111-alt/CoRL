"""Minimal icecream.ic fallback for local transplant runs."""


def ic(*args):
    if not args:
        print("ic|")
        return None
    print("ic|", *args)
    return args[0] if len(args) == 1 else args

