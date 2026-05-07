"""Optional planning helpers.

The production agent can work without a separate planner, but keeping this
module gives the system a clean extension point for query decomposition.
"""

from __future__ import annotations


class SimplePlanner:
    """Small rule-based planner used before the ReAct loop when desired."""

    def plan(self, query: str) -> list[str]:
        """Return coarse research steps for a user query."""

        lowered = query.lower()
        steps = ["clarify research objective"]
        if "latest" in lowered or "recent" in lowered:
            steps.append("search recent papers")
        steps.extend(["retrieve relevant context", "summarize findings"])
        return steps

