"""Minimal stub for environments without slackweb installed."""


class Slack:
    def __init__(self, url=None):
        self.url = url

    def notify(self, text=None, **kwargs):
        return None

