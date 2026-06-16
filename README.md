# SurveyFlow

[![CI](https://github.com/<your-username>/SurveyFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/SurveyFlow/actions/workflows/ci.yml) · [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/) · [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) · [![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**SurveyFlow** is a deterministic, retrieval-augmented research survey writing system powered by [LangGraph](https://langchain-ai.github.io/langgraph/). It takes a research topic as input, automatically retrieves relevant papers from arXiv and Semantic Scholar, generates a structured literature survey, and exposes everything via a FastAPI service with real-time WebSocket streaming.

---

## Key Features

- **Deterministic pipeline** — nine-node `StateGraph` with a single conditional edge; identical execution across runs, fully inspectable at every step.
- **Multi-source retrieval** — pulls papers from arXiv and Semantic Scholar, converts PDFs to structured markdown via MinerU, deduplicates, and chunks for embedding.
- **Gap-aware generation** — optional research gap analysis (Theoretical, Methodological, Data, Application, Evaluation, Perspective) drives a more focused survey.
- **Citation-grounded writing** — retrieved paper content is injected as evidence; the system outputs APA-style references.
- **Real-time streaming** — WebSocket endpoint streams node completions, progress updates, and log entries as the pipeline runs.
- **Reproducible benchmarks** — a five-phase citation-accuracy evaluation framework (`generate → parse → verify → attribute → judge`) with per-seed runs and significance-ready result files.
- **Three system modes** — SurveyFlow, Naive RAG, and LLM-only baselines share the same corpus for clean ablation.

---

## Architecture

### Tech Stack

| Component | Technology |
|---|---|
| Workflow framework | LangGraph (`StateGraph`) |
| LLM | Zhipu AI GLM-4 (OpenAI-compatible API) |
| Embedding | GLM-Embedding-3 |
| Vector store | ChromaDB |
| PDF processing | MinerU |
| API layer | FastAPI (REST + WebSocket) |
| Package manager | `uv` |

### Pipeline

The graph runs nine nodes in a fixed order. One conditional edge (after the Splitter node) decides whether the optional Gap node executes.

```
query_node → download_node → splitter_node
                                    │
                    [conditional: should_run_gap?]
                         ↙              ↘
                   gap_node          (skip gap)
                       ↓                   ↓
                 retriever_node ←←←←←←←←←←←
                       ↓
               classifier_node
                       ↓
                section_node
                       ↓
                outline_node
                       ↓
                writer_node → END
```

| Node | Responsibility |
|---|---|
| **Query Node** | Generates strict and generic search queries for arXiv and Semantic Scholar |
| **Download Node** | Downloads PDFs, converts PDF → Markdown → JSON via MinerU; deduplicates results |
| **Splitter Node** | Chunks text, embeds with GLM-Embedding-3, stores in ChromaDB; builds citation metadata |
| **Gap Node** *(conditional)* | Analyzes research gaps based on the selected `gap_type` and retrieves relevant evidence |
| **Retriever Node** | Generates paper descriptions; retrieves citation-grounding chunks per paper |
| **Classifier Node** | Clusters papers into categories using LLM-driven label assignment |
| **Section Node** | Generates section titles per category |
| **Outline Node** | Produces a structured survey outline with hierarchical headings |
| **Writer Node** | Writes abstract, introduction, category sections, future directions, conclusion, and references |

---

## Project Structure

```
SurveyFlow/
├── api/                     # FastAPI application
│   ├── main.py              # Entry point, router wiring
│   ├── schemas.py           # Pydantic request/response models
│   ├── deps.py              # Dependency injection (store, graph)
│   └── routes/
│       ├── survey.py        # /survey/generate, /survey/{id}/status, /survey/{id}/result
│       ├── streaming.py     # WebSocket streaming routes
│       └── health.py        # /health
├── core/
│   ├── state.py             # SurveyState Pydantic model + enums
│   ├── llm_client.py        # Zhipu AI / OpenAI-compatible LLM client
│   ├── embedder.py          # GLM-Embedding-3 wrapper
│   ├── chroma_client.py     # ChromaDB client
│   ├── arxiv_client.py      # arXiv API client
│   ├── s2_client.py         # Semantic Scholar API client
│   ├── pdf_processor.py     # MinerU PDF → JSON pipeline
│   ├── mineru_client.py     # MinerU API wrapper
│   ├── dedup.py             # Paper deduplication
│   ├── tsv_manager.py       # TSV state persistence
│   └── constants.py         # Shared constants
├── graph/
│   ├── builder.py           # StateGraph assembly and compilation
│   ├── nodes.py             # Node entry points (wraps individual node modules)
│   └── edges.py             # Conditional edge logic
├── nodes/                   # Per-node business logic
│   ├── query_node.py
│   ├── download_node.py
│   ├── splitter_node.py
│   ├── gap_node.py
│   ├── retriever_node.py
│   ├── classifier_node.py
│   ├── section_node.py
│   ├── outline_node.py
│   └── writer_node.py
├── prompts/                 # LLM prompt templates per node
│   ├── query_prompts.py
│   ├── gap_prompts.py
│   ├── classification_prompts.py
│   ├── section_prompts.py
│   ├── outline_prompts.py
│   └── writer_prompts.py
├── storage/
│   ├── survey_store.py      # Survey state persistence (file-based)
│   └── file_manager.py      # File I/O utilities
├── streaming/
│   └── manager.py           # WebSocket broadcast manager
├── evaluation_citation/      # Citation accuracy benchmark suite
│   ├── run_benchmark.py      # Phase orchestrator
│   ├── configs.py           # Topics, systems, seeds, thresholds
│   ├── corpus_cache.py      # Shared per-topic corpus for ablation
│   ├── generators/          # One runner per system (surveyflow, naive_rag, llm_only)
│   ├── judges/              # LLM-as-judge: faithfulness + attribution
│   ├── verifiers/           # arXiv / S2 reference resolution
│   ├── citation_parser.py   # Survey.md → reference list
│   ├── attribution_parser.py # ALCE sentence-level claim/citation extraction
│   ├── failure_classifier.py # Resolution failure taxonomy
│   └── aggregate.py         # Per-cell + cross-cell metric aggregation
├── scripts/                 # Utility scripts
├── tests/                   # Pytest test suite
└── pyproject.toml
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install uv
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Zhipu AI API key and other settings
```

### 3. Start the API server

```bash
uv run uvicorn api.main:app --reload --port 8000
```

API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Generate a survey

```bash
curl -X POST http://localhost:8000/survey/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Large Language Models in Recommendation Systems",
    "gap_type": "Methodological Gap",
    "paper_count": 20
  }'
```

The response includes a `survey_id`. Poll its status or connect via WebSocket to watch progress:

```bash
# REST polling
curl http://localhost:8000/survey/{survey_id}/status

# WebSocket streaming
# Connect to: ws://localhost:8000/survey/{survey_id}/stream
```

Fetch the completed survey when done:

```bash
curl http://localhost:8000/survey/{survey_id}/result
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/survey/generate` | Start a new survey generation task |
| `GET` | `/survey/{survey_id}/status` | Poll current status, node, progress, logs |
| `GET` | `/survey/{survey_id}/result` | Retrieve the completed survey |
| `WS` | `/survey/{survey_id}/stream` | Real-time WebSocket log and status stream |
| `GET` | `/health` | Liveness check |

---

## Request Schema

```json
{
  "topic": "Large Language Models in Recommendation Systems",
  "gap_type": "Methodological Gap",
  "paper_count": 20,
  "year_range": ["2020-01-01", ""],
  "enable_citations": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `topic` | `string` | *required* | Research topic (natural language) |
| `gap_type` | `string` | `""` | One of: Theoretical Gap, Methodological Gap, Data Gap, Application Gap, Evaluation Gap, Perspective Gap |
| `paper_count` | `int` | `20` | Number of papers to download |
| `year_range` | `[string, string]` | `["2020-01-01", ""]` | ISO date range filter |
| `enable_citations` | `bool` | `true` | Inject APA citations from retrieved papers |

---

## Benchmark Suite

The `evaluation_citation/` module provides a five-phase citation-accuracy evaluation framework comparing **SurveyFlow** against **Naive RAG** and **LLM-only** baselines:

| Phase | Description |
|---|---|
| **generate** | Produce one survey per (system, topic, seed) cell |
| **parse** | Extract `Reference:` lines from each survey.md |
| **verify** | Resolve each reference via arXiv + S2; classify FOUND / MISMATCH / MISSING_ID / NOT_FOUND |
| **attribute** | ALCE sentence-level attribution: Citation Recall / Precision + Bare Assertion Rate |
| **judge** | LLM-as-judge topic faithfulness per resolved reference |
| **classify** | Decompose verification failures into failure categories |

Results are written to `evaluation_citation/results/{system}/{topic}__seed{k}__{phase}.json`.

Run all phases for a cell:

```bash
python -m evaluation_citation.run_benchmark --phase all --system SURVEYFLOW --topic neural_audio_codec --seed 0
```

Run a single phase:

```bash
python -m evaluation_citation.run_benchmark --phase verify --system SURVEYFLOW
```

Resume a partial run — each phase is independently idempotent and resumable:

```bash
python -m evaluation_citation.run_benchmark --phase attribute
```
