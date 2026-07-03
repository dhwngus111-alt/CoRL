"""Minimal stub for environments without setproctitle installed."""

_TITLE = ""


def setproctitle(title):
    global _TITLE
    _TITLE = str(title)
    return None


def getproctitle():
    return _TITLE

