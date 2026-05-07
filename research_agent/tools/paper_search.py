"""Paper search tool for arXiv and OpenReview."""

from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from research_agent.config import settings
from research_agent.tools.base import BaseTool


@dataclass
class PaperSearchTool(BaseTool):
    """Search recent research papers.

    The arXiv path uses only the standard library. OpenReview is left as a TODO
    because its API/client behavior differs across venues and deployments.
    """

    name: str = "paper_search"
    description: str = "Search latest papers from arXiv. Input JSON: query, max_results."

    def run(self, input: str) -> str:
        payload = self._parse_input(input)
        query = payload.get("query", input).strip()
        max_results = int(payload.get("max_results", settings.arxiv_max_results))
        papers = self.search_arxiv(query=query, max_results=max_results)
        return json.dumps({"source": "arxiv", "papers": papers}, ensure_ascii=False)

    def search_arxiv(self, query: str, max_results: int) -> list[dict[str, str]]:
        encoded = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": settings.arxiv_sort_by,
                "sortOrder": settings.arxiv_sort_order,
            }
        )
        url = f"https://export.arxiv.org/api/query?{encoded}"
        with urllib.request.urlopen(url, timeout=settings.request_timeout_seconds) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[dict[str, str]] = []
        for entry in root.findall("atom:entry", ns):
            links = entry.findall("atom:link", ns)
            pdf_url = ""
            for link in links:
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
            papers.append(
                {
                    "id": self._text(entry, "atom:id", ns),
                    "title": " ".join(self._text(entry, "atom:title", ns).split()),
                    "summary": " ".join(self._text(entry, "atom:summary", ns).split()),
                    "published": self._text(entry, "atom:published", ns),
                    "pdf_url": pdf_url,
                }
            )
        return papers

    def _text(self, entry: ET.Element, path: str, ns: dict[str, str]) -> str:
        node = entry.find(path, ns)
        return node.text.strip() if node is not None and node.text else ""

    def _parse_input(self, input: str) -> dict[str, object]:
        try:
            parsed = json.loads(input)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


# TODO: Add OpenReview search support behind the same tool interface.
