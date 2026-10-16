"""Marker used by every function that is left for you to implement."""

from typing import NoReturn


def todo(milestone: str, hint: str = "") -> NoReturn:
    message = f"[{milestone}] not implemented yet"
    if hint:
        message += f": {hint}"
    raise NotImplementedError(message)
