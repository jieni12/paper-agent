"""Simple persistent memory for user interests and read papers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from research_agent.config import settings


@dataclass
class ResearchMemory:
    """JSON-backed memory store.

    This implementation is intentionally small. For production multi-user
    deployments, replace the file store with Postgres or another transactional
    backend while preserving these method contracts.
    """

    path: Path = settings.memory_path
    user_interests: list[str] = field(default_factory=list)
    read_papers: dict[str, dict[str, object]] = field(default_factory=dict)

    def load(self) -> "ResearchMemory":
        if not self.path.exists():
            return self
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.user_interests = list(data.get("user_interests", []))
        self.read_papers = dict(data.get("read_papers", {}))
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "user_interests": self.user_interests,
                    "read_papers": self.read_papers,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def add_interest(self, interest: str) -> None:
        interest = interest.strip()
        if interest and interest not in self.user_interests:
            self.user_interests.append(interest)
            self.save()

    def mark_read(self, paper_id: str, metadata: dict[str, object]) -> None:
        self.read_papers[paper_id] = metadata
        self.save()

    def has_read(self, paper_id: str) -> bool:
        return paper_id in self.read_papers

    def filter_unread(self, papers: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            paper
            for paper in papers
            if not self.has_read(str(paper.get("id") or paper.get("paper_id") or ""))
        ]

