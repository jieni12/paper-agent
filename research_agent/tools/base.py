"""Shared tool abstractions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseTool:
    """Base class for lightweight tools.

    Tools remain intentionally small: a name, a description, and a ``run``
    method. This mirrors the interface the ReAct loop expects.
    """

    name: str
    description: str

    def run(self, input: str) -> str:  # pragma: no cover - interface method
        raise NotImplementedError

