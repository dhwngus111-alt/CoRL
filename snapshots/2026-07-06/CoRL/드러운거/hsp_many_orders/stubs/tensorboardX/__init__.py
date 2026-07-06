"""Minimal stub for environments without tensorboardX installed."""


class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        return None

    def export_scalars_to_json(self, *args, **kwargs):
        return None

    def close(self):
        return None

