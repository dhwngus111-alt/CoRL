"""No-op slackweb fallback for local transplant runs."""


class Slack:
    def __init__(self, url=None, *args, **kwargs):
        self.url = url

    def notify(self, *args, **kwargs):
        return None

