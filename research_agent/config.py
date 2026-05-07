"""Application configuration.

The defaults intentionally support local development without external services.
Production deployments should configure these values through environment
variables or a secrets manager.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency may be absent in minimal envs
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
else:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the research agent system."""

    app_env: str = os.getenv("APP_ENV", "development")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    data_dir: Path = Path(os.getenv("RESEARCH_AGENT_DATA_DIR", "data"))
    index_dir: Path = Path(os.getenv("RESEARCH_AGENT_INDEX_DIR", "data/faiss"))
    paper_dir: Path = Path(os.getenv("RESEARCH_AGENT_PAPER_DIR", "data/papers"))
    metadata_path: Path = Path(
        os.getenv("RESEARCH_AGENT_METADATA_PATH", "data/papers.jsonl")
    )
    memory_path: Path = Path(os.getenv("RESEARCH_AGENT_MEMORY_PATH", "data/memory.json"))

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local-hash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))

    llm_provider: str = os.getenv("LLM_PROVIDER", "stub")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_model: str = os.getenv(
        "OPENAI_MODEL",
        os.getenv("MODEL_NAME", "gpt-4.1-mini"),
    )
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    vector_db: str = os.getenv("VECTOR_DB", "faiss")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    chunk_max_words: int = int(os.getenv("CHUNK_MAX_WORDS", "350"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))

    arxiv_max_results: int = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
    arxiv_sort_by: str = os.getenv("ARXIV_SORT_BY", "submittedDate")
    arxiv_sort_order: str = os.getenv("ARXIV_SORT_ORDER", "descending")
    openreview_base_url: str = os.getenv("OPENREVIEW_BASE_URL", "https://api2.openreview.net")
    openreview_username: str | None = os.getenv("OPENREVIEW_USERNAME") or None
    openreview_password: str | None = os.getenv("OPENREVIEW_PASSWORD") or None

    pdf_parser: str = os.getenv("PDF_PARSER", "pymupdf")
    grobid_url: str = os.getenv("GROBID_URL", "http://localhost:8070")

    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "6"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    max_pdf_mb: int = int(os.getenv("MAX_PDF_MB", "50"))
    user_agent: str = os.getenv("USER_AGENT", "research-agent/0.1")


settings = Settings()
