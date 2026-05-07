# Research Agent

Production-oriented backend skeleton for an AI Research Assistant.

## Project Structure

```text
research_agent/
├── agent/
│   ├── core.py
│   └── planner.py
├── tools/
│   ├── base.py
│   ├── paper_search.py
│   ├── paper_parse.py
│   ├── rag_retriever.py
│   └── summarizer.py
├── rag/
│   ├── index.py
│   ├── embedding.py
│   └── retriever.py
├── pipeline/
│   └── ingest.py
├── memory/
│   └── memory.py
├── api/
│   └── main.py
└── config.py
```

## Run

```bash
pip install -r requirements.txt
uvicorn research_agent.api.main:app --reload
```

## API

- `POST /query`: run the ReAct agent.
- `POST /ingest`: fetch, parse, summarize, and index recent papers.
- `GET /health`: service health check.

## Notes

- Default LLM provider is a deterministic stub for local smoke tests.
- Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` for production LLM reasoning.
- Default embedding provider is `local-hash` for development. Use OpenAI or add
  a sentence-transformers BGE backend for production semantic retrieval.
- OpenReview support is marked as a TODO in `paper_search.py` because venue API
  details often require deployment-specific configuration.

