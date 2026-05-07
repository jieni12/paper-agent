"""PDF parsing tool based on PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

from research_agent.tools.base import BaseTool


@dataclass
class PaperParseTool(BaseTool):
    """Extract text and coarse sections from a PDF."""

    name: str = "paper_parse"
    description: str = "Parse a PDF and extract structured sections. Input JSON: pdf_path."

    def run(self, input: str) -> str:
        payload = self._parse_input(input)
        pdf_path = Path(str(payload.get("pdf_path", input))).expanduser()
        parsed = self.parse_pdf(pdf_path)
        return json.dumps(parsed, ensure_ascii=False)

    def parse_pdf(self, pdf_path: Path) -> dict[str, object]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install PyMuPDF to parse PDFs: pip install pymupdf") from exc

        pages: list[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        text = "\n".join(pages)
        sections = self.extract_sections(text)
        return {
            "pdf_path": str(pdf_path),
            "title": self._infer_title(text),
            "sections": sections,
            "text": text,
        }

    def extract_sections(self, text: str) -> dict[str, str]:
        headings = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "method",
            "methods",
            "methodology",
            "experiments",
            "results",
            "discussion",
            "limitations",
            "conclusion",
        ]
        pattern = re.compile(
            r"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?(" + "|".join(map(re.escape, headings)) + r")\s*$"
        )
        matches = list(pattern.finditer(text))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            key = match.group(1).lower()
            sections[key] = text[start:end].strip()
        if "abstract" not in sections:
            sections["abstract"] = self._extract_abstract_fallback(text)
        return sections

    def _infer_title(self, text: str) -> str:
        for line in text.splitlines():
            cleaned = line.strip()
            if len(cleaned) > 10 and not cleaned.lower().startswith("arxiv"):
                return cleaned
        return ""

    def _extract_abstract_fallback(self, text: str) -> str:
        match = re.search(r"(?is)abstract\s*(.+?)(?:\n\s*1\s+introduction|\n\s*introduction)", text)
        return match.group(1).strip() if match else ""

    def _parse_input(self, input: str) -> dict[str, object]:
        try:
            parsed = json.loads(input)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

