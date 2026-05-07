"""Structured paper summarization tool."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from research_agent.tools.base import BaseTool


@dataclass
class SummarizerTool(BaseTool):
    """Extract a structured summary from paper text or parsed sections."""

    name: str = "summarize"
    description: str = "Summarize paper text into title/problem/method/contribution/limitations."

    def run(self, input: str) -> str:
        payload = self._parse_input(input)
        title = str(payload.get("title", ""))
        text = str(payload.get("text", input))
        sections = payload.get("sections", {})
        if not isinstance(sections, dict):
            sections = {}
        structured = self.summarize(text=text, title=title, sections=sections)
        return json.dumps(structured, ensure_ascii=False)

    def summarize(
        self,
        text: str,
        title: str = "",
        sections: dict[str, object] | None = None,
    ) -> dict[str, str]:
        sections = sections or {}
        abstract = str(sections.get("abstract", "")) or self._first_sentences(text, 4)
        method = (
            str(sections.get("method", ""))
            or str(sections.get("methods", ""))
            or str(sections.get("methodology", ""))
        )
        limitations = str(sections.get("limitations", ""))
        contribution = self._extract_contribution(text)

        return {
            "title": title or self._infer_title(text),
            "problem": self._first_sentences(abstract, 2),
            "method": self._first_sentences(method or abstract, 3),
            "contribution": contribution,
            "limitations": self._first_sentences(limitations, 3) if limitations else "TODO: Limitations not explicitly found in parsed text.",
        }

    def _extract_contribution(self, text: str) -> str:
        patterns = [
            r"(?is)(?:our contributions are|we make the following contributions)(.+?)(?:\n\n|introduction|method)",
            r"(?is)(?:we propose|this paper proposes)(.+?)(?:\.|\n)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._first_sentences(match.group(0), 3)
        return self._first_sentences(text, 3)

    def _infer_title(self, text: str) -> str:
        for line in text.splitlines():
            cleaned = line.strip()
            if len(cleaned) > 10:
                return cleaned
        return ""

    def _first_sentences(self, text: str, count: int) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(sentences[:count]).strip()

    def _parse_input(self, input: str) -> dict[str, object]:
        try:
            parsed = json.loads(input)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

