"""FAISS-backed paper chunk index."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from research_agent.config import settings
from research_agent.rag.embedding import Embedder, build_embedder


@dataclass
class PaperChunk:
    """A retrievable chunk of a paper."""

    paper_id: str
    title: str
    section: str
    text: str
    source_url: str = ""


class FaissPaperIndex:
    """Thin FAISS index wrapper with JSONL metadata sidecar."""

    def __init__(
        self,
        index_dir: Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.index_dir = index_dir or settings.index_dir
        self.index_path = self.index_dir / "papers.faiss"
        self.metadata_path = self.index_dir / "metadata.jsonl"
        self.embedder = embedder or build_embedder()
        self._index = None
        self._metadata: list[PaperChunk] = []

    def add_chunks(self, chunks: Iterable[PaperChunk]) -> int:
        """Embed and add chunks to FAISS."""

        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        faiss = self._import_faiss()
        vectors = self.embedder.embed([chunk.text for chunk in chunk_list])
        index = self._load_or_create_index(faiss)
        index.add(self._to_float32_array(vectors))
        self._metadata.extend(chunk_list)
        self._index = index
        self.save()
        return len(chunk_list)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Return top-k chunks ranked by inner product similarity."""

        faiss = self._import_faiss()
        index = self._load_or_create_index(faiss)
        if index.ntotal == 0:
            return []

        self._load_metadata()
        query_vector = self.embedder.embed([query])
        scores, indices = index.search(self._to_float32_array(query_vector), top_k)

        results: list[dict[str, object]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            chunk = self._metadata[int(idx)]
            item = asdict(chunk)
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self) -> None:
        """Persist FAISS index and metadata."""

        if self._index is None:
            return

        faiss = self._import_faiss()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            for chunk in self._metadata:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    def _load_or_create_index(self, faiss):
        if self._index is not None:
            return self._index

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            return self._index

        self._index = faiss.IndexFlatIP(self.embedder.dim)
        return self._index

    def _load_metadata(self) -> None:
        if self._metadata or not self.metadata_path.exists():
            return
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self._metadata = [
                PaperChunk(**json.loads(line)) for line in handle if line.strip()
            ]

    def _to_float32_array(self, vectors: list[list[float]]):
        import numpy as np

        return np.array(vectors, dtype="float32")

    def _import_faiss(self):
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is required for vector search. Install it with: pip install faiss-cpu"
            ) from exc
        return faiss


def chunks_from_structured_paper(paper: dict[str, object]) -> list[PaperChunk]:
    """Create section-aware chunks from parsed/structured paper data."""

    paper_id = str(paper.get("id") or paper.get("paper_id") or paper.get("title") or "")
    title = str(paper.get("title", ""))
    source_url = str(paper.get("source_url") or paper.get("pdf_url") or "")
    sections = paper.get("sections", {})
    chunks: list[PaperChunk] = []

    if isinstance(sections, dict):
        for section, value in sections.items():
            text = str(value).strip()
            if text:
                chunks.extend(_split_long_section(paper_id, title, str(section), text, source_url))

    summary_fields = ["problem", "method", "contribution", "limitations"]
    for field in summary_fields:
        text = str(paper.get(field, "")).strip()
        if text:
            chunks.append(PaperChunk(paper_id=paper_id, title=title, section=field, text=text, source_url=source_url))

    return chunks


def _split_long_section(
    paper_id: str,
    title: str,
    section: str,
    text: str,
    source_url: str,
    max_words: int | None = None,
) -> list[PaperChunk]:
    max_words = max_words or settings.chunk_max_words
    words = text.split()
    if len(words) <= max_words:
        return [PaperChunk(paper_id=paper_id, title=title, section=section, text=text, source_url=source_url)]

    chunks: list[PaperChunk] = []
    for index, start in enumerate(range(0, len(words), max_words)):
        part = " ".join(words[start : start + max_words])
        chunks.append(
            PaperChunk(
                paper_id=paper_id,
                title=title,
                section=f"{section}:{index + 1}",
                text=part,
                source_url=source_url,
            )
        )
    return chunks
