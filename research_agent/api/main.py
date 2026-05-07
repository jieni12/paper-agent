"""FastAPI entry point for the research agent backend."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from research_agent.agent.core import AgentResult, ReActAgent
from research_agent.pipeline.ingest import PaperIngestionPipeline
from research_agent.tools.paper_parse import PaperParseTool
from research_agent.tools.paper_search import PaperSearchTool
from research_agent.tools.rag_retriever import RAGRetrieveTool
from research_agent.tools.summarizer import SummarizerTool


app = FastAPI(
    title="Research Agent API",
    version="0.1.0",
    description="Backend API for a modular AI research assistant.",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    steps: list[dict[str, str | None]]


class IngestRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int | None = Field(default=None, ge=1, le=50)


class IngestResponse(BaseModel):
    query: str
    fetched: int
    ingested: int
    skipped_duplicates: int
    papers: list[dict[str, str]]


def build_agent(max_steps: int | None = None) -> ReActAgent:
    """Create a ReAct agent with the required tool set."""

    tools = [
        PaperSearchTool(),
        PaperParseTool(),
        RAGRetrieveTool(),
        SummarizerTool(),
    ]
    return ReActAgent(tools=tools, max_steps=max_steps)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest) -> QueryResponse:
    """Run the ReAct agent for a user research question."""

    try:
        result: AgentResult = build_agent(max_steps=request.max_steps).run(request.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        steps=[asdict(step) for step in result.steps],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_papers(request: IngestRequest) -> IngestResponse:
    """Trigger the paper ingestion pipeline."""

    try:
        result = PaperIngestionPipeline().run(
            query=request.query,
            max_results=request.max_results,
        )
    except Exception as exc:  # pragma: no cover - API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(**asdict(result))
