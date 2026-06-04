# HeartSafe RAG: Guideline-Driven Cardiology Agent

A Retrieval-Augmented Generation (RAG) system for zero-hallucination heart failure decision support, grounded exclusively in AHA/ACC Heart Failure Guidelines.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)
[![Langfuse](https://img.shields.io/badge/Langfuse-Observability-orange)](https://langfuse.com)

## Features

- **Evidence-Based Responses**: Every response is grounded in 2022 AHA/ACC Heart Failure Guidelines.
- **Zero Hallucinations**: Strict retrieval enforcement + structured output parsing + VLM-guard validation pipeline ensures no made-up information; the model refuses to answer if guidelines are missing.
- **Hybrid Retrieval**: Combines semantic search (FAISS) with keyword matching (BM25) for high-precision context fetching.
- **Domain-Optimized Embeddings**: PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO) for biomedical semantic understanding.
- **Query Rewriting**: HyDE (Hypothetical Document Embeddings) + Multi-Query generation expand queries for better retrieval coverage.
- **OCR Support**: Tesseract OCR for scanned PDFs and image extraction from guideline documents.
- **Table & Image Extraction**: Tables extracted via pdfplumber, images extracted via PyMuPDF with OCR fallback.
- **VLM-Guard Validation Pipeline**: 8 rule-based validation engines cross-check every response against guideline COR/LOE tables, LVEF thresholds, drug-class relationships, and answer consistency before returning.
- **Structured Reasoning Output**: Each response includes explicit reasoning steps with claim types (diagnosis, recommendation, threshold, contraindication, etc.) and source chunk references.
- **Evaluation Pipeline**: Integrated Langfuse experiment runner with LLM-as-a-Judge for measuring clinical accuracy on a 40-question golden dataset.
- **Production-Ready**: FastAPI backend with health checks, structured JSON logging, response caching, and Docker support.
- **Chat Interface**: Built-in web UI served via FastAPI + Jinja2 templates with validation audit trail display.

## Architecture

```mermaid
flowchart TB
    subgraph OFFLINE["Offline Ingestion"]
        A[Guideline PDFs] --> B[PDF Processing<br/>Text + OCR + Tables + Images]
        B --> C[Section-Aware Chunking<br/>GuidelineChunker]
        C --> D[Embedding<br/>PubMedBERT]
        D --> E[FAISS Index + Chunks]
        C --> F[Chunk Store<br/>for BM25 rebuild]
    end

    subgraph ONLINE["Online Serving"]
        G[User Query] --> H[Query Rewriting<br/>HyDE + Multi-Query]
        H --> I[Hybrid Retrieval<br/>FAISS + BM25 Ensemble]
        I --> J{Guard Check:<br/>Docs Retrieved?}
        J -->|Yes| K[Structured LLM Generation<br/>Llama-3.1-8b (Groq)]
        J -->|No| N[Refusal Response]
        K --> L[VLM-Guard Validation<br/>8 Rule Engines]
        L --> M{Validation<br/>Passed?}
        M -->|Yes| O[Response with<br/>Reasoning Steps + Citations]
        M -->|Blocked| N[Refusal/Corrected Response]
        M -->|Flagged| P[Flagged Response<br/>with Audit Trail]
    end
```

### Validation Pipeline (VLM-Guard)

Each generated response is validated by 8 rule engines before returning:

| Rule | Function |
|------|----------|
| **CORLevelRule** | Validates Class of Recommendation against hardcoded guideline knowledge base (33 entries) |
| **CSVCorRule** | Data-driven COR validation against `data/rules/cor_lookup.csv` (34 therapies) |
| **LVEFThresholdRule** | Validates LVEF ranges (HFrEF ≤40, HFmrEF 41-49, HFpEF ≥50); auto-corrects misclassification |
| **DrugClassRule** | Validates GDMT drug-class relationships (5 classes: ARNi, ACEi/ARB, Beta Blocker, MRA, SGLT2i) |
| **ContraindicationRule** | Flags contraindications (e.g., ACEi/ARB with potassium >5.0, MRA with CrCl <30) |
| **ValueStatementRule** | Validates high/low value statements against guideline thresholds |
| **AnswerConsistencyRule** | Cross-validates final answer against all claims; detects contradictions with Class 1 recommendations |
| **AnswerCORCrossCheckRule** | Cross-validates COR in final answer against CSV lookup; catches non-standard formats |

## Installation

### Clone the repository

```bash
git clone https://github.com/MohamedFakhry2007/heart_safe_rag.git
cd heart_safe_rag
```

### Install dependencies

```bash
# Install Poetry if you don't have it
pip install poetry

# Install project dependencies
poetry install
```

### Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
GROQ_API_KEY=gsk_...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: Log level
LOG_LEVEL=INFO
```

## Quick Start

### 1. Ingest Guidelines

Parse PDFs and build the vector index. Supports single file or directory:

```bash
# Ingest all PDFs from data/guidelines/
make ingest
# or manually:
poetry run python ingest.py data/guidelines --output data/vector_store

# Ingest a single PDF
poetry run python ingest.py path/to/guideline.pdf
```

### 2. Start the API Server

Launch the backend and the web Chat UI.

```bash
make run
# or:
make api
# or manually:
poetry run uvicorn heartsafe_rag.api:app --reload
```

- **Chat UI**: Open http://localhost:8000
- **API Docs** (Swagger): Open http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Run Evaluations

Upload the golden dataset to Langfuse and run the evaluation experiment:

```bash
# Upload golden dataset to Langfuse (one-time setup)
poetry run python -m heartsafe_rag.upload_dataset

# Run evaluation experiment with LLM-as-a-Judge
poetry run python -m heartsafe_rag.evaluate --dataset-name heartsafe_golden_dataset_v1 --delay 5
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `make install` | Install poetry dependencies |
| `make setup` | Install deps + create data directories |
| `make ingest` | Run ingestion pipeline on `data/guidelines/` |
| `make run` / `make api` | Start FastAPI server (localhost:8000) |
| `make format` | Auto-format code with ruff |
| `make check` | Lint with ruff + type-check with mypy |
| `make test` | Run pytest suite |
| `poetry run python -m heartsafe_rag.evaluate` | Run golden dataset evaluation |
| `poetry run python -m heartsafe_rag.upload_dataset` | Upload dataset to Langfuse |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI (HTML) |
| `/health` | GET | Health check (returns service status) |
| `/chat` | POST | Ask a clinical question |

### `/chat` Example

```json
{
  "query": "What are the four medication classes in GDMT for HFrEF?"
}
```

Response:

```json
{
  "answer": "The four medication classes are...",
  "sources": [
    { "content": "GDMT for HFrEF includes...", "source": "2022-AHA-HF-Guideline.pdf", "chunk_index": 0 }
  ],
  "reasoning_steps": [
    { "step": "HFrEF GDMT consists of...", "claim_type": "recommendation", "source_indices": [0, 1] }
  ],
  "validation": {
    "status": "passed",
    "rules_fired": [
      { "rule_name": "CORLevelRule", "action_type": "pass", "message": "COR validated", "severity": "info", "claim_index": 0, "modified_fields": null }
    ]
  }
}
```

## Testing

```bash
make test
# or:
poetry run pytest
```

The test suite includes:
- **Unit tests**: chunking boundaries, ingestion (OCR, table extraction, image extraction, single/batch PDF processing), generation with mock LLM
- **Integration tests**: PDF-to-chunks pipeline with synthetic PDFs
- **VLM-Guard rule tests**: implicit COR detection, CSV rule matching, answer cross-check (format, valid/invalid/wrong COR)
- **Fixtures**: synthetic PDFs (text-only, multi-page, tables, images, scanned, unicode, corrupt)

## Project Structure

```
HeartSafe-RAG/
├── ingest.py                       # CLI entry point for ingestion
├── Makefile                        # Command shortcuts
├── pyproject.toml                  # Poetry config & dependencies (v0.1.0)
├── docker-compose.yml              # Container orchestration
├── Dockerfile                      # Container image
├── .github/workflows/ci.yml        # CI pipeline (ruff + mypy + pytest)
├── data/
│   ├── guidelines/                 # PDF guideline source files
│   ├── raw_pdfs/                   # Raw PDF copies
│   ├── vector_store/               # FAISS index + chunks + BM25 index
│   │   └── images/                 # Extracted page images
│   ├── bm25_index.pkl              # BM25 index (legacy location)
│   └── rules/
│       └── cor_lookup.csv          # 34-entry COR/LOE reference table
├── eval/
│   ├── data/golden_dataset.json    # 40 QA pairs for evaluation
│   └── results/                    # Evaluation reports
├── faiss_index/                    # Legacy index directory
├── prompts/
│   ├── system_prompt.txt           # Plain-text mode system prompt
│   └── system_prompt_structured.txt # Structured JSON output mode (preferred)
├── src/
│   └── heartsafe_rag/
│       ├── __init__.py             # Package init (v0.2.0)
│       ├── api.py                  # FastAPI server + endpoints
│       ├── config.py               # Pydantic settings (.env)
│       ├── schemas.py              # Request/response models
│       ├── chunking.py             # Guideline-aware text chunker
│       ├── ingestion.py            # PDF processing pipeline
│       ├── retrieval.py            # Hybrid retriever (FAISS+BM25+HyDE+Multi-Query)
│       ├── generation.py           # Guarded generation chain
│       ├── evaluate.py             # Langfuse evaluation experiment runner
│       ├── upload_dataset.py       # Upload golden dataset to Langfuse
│       ├── exceptions.py           # Custom exception hierarchy
│       ├── templates/
│       │   └── chat.html           # Chat UI template with validation display
│       ├── validation/
│       │   ├── __init__.py         # Re-exports all rules + service
│       │   ├── rules.py            # 8 VLM-guard validation rule engines
│       │   └── service.py          # ValidationService orchestrator
│       └── utils/
│           ├── logger.py           # Structured JSON logging
│           └── callbacks.py        # LangChain callback handler
└── tests/
    ├── conftest.py                 # Fixtures (synthetic PDFs, test index)
    ├── test_chunking.py
    ├── test_generation.py
    ├── test_ingestion.py           # 8 test classes (OCR, tables, images, batch)
    ├── test_integration.py
    ├── test_implicit_cor.py        # 10 VLM-guard rule validation tests
    └── test_retrieval.py
```

## Configuration

Key settings (from `src/heartsafe_rag/config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `ENVIRONMENT` | `"development"` | App environment |
| `LOG_LEVEL` | `"INFO"` | Logging level |
| `LOG_FILE` | `None` | Optional log file path |
| `DATA_DIR` | `data` | Data directory |
| `VECTOR_DB_PATH` | `data/vector_store` | FAISS index folder |
| `BM25_PATH` | `data/bm25_index.pkl` | BM25 index path |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq LLM for generation |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `EMBEDDING_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | Biomedical embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size (chars) |
| `CHUNK_OVERLAP` | `100` | Chunk overlap (chars) |
| `CHUNK_SEPARATORS` | `["\n\n", "\n", ". ", " ", ""]` | Chunk separators |
| `INITIAL_RETRIEVAL_K` | `7` | Top-K candidates from ensemble |
| `HYBRID_WEIGHTS` | `[0.4, 0.6]` | [BM25 weight, FAISS weight] |
| `ENABLE_HYDE` | `true` | Hypothetical Document Embeddings |
| `ENABLE_MULTI_QUERY` | `true` | Multi-Query expansion |
| `MULTI_QUERY_COUNT` | `3` | Number of query variants |
| `ENABLE_VALIDATION` | `true` | VLM-guard validation pipeline |
| `VALIDATION_MAX_RETRIES` | `1` | Max retries on blocked output |
| `ENABLE_VLM_GUARD` | `true` | VLM guard structured parsing |
| `LLM_TIMEOUT` | `30` | LLM timeout in seconds |
| `TESSERACT_CMD` | `tesseract` | Tesseract OCR command |
| `OCR_ENABLED` | `true` | Enable OCR for scanned PDFs |

## Docker Deployment

```bash
docker compose up --build
```

This starts the FastAPI server on port 8000, mounting `./data` and `./src` as volumes.

## Tech Stack

- **LLM**: Llama-3.1-8b-instant (via Groq) for low-latency medical reasoning
- **Embeddings**: PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO) via sentence-transformers
- **Vector Store**: FAISS (dense) + BM25 (sparse) with EnsembleRetriever
- **Query Expansion**: HyDE + Multi-Query generation
- **Validation**: VLM-guard (8 rule engines: COR, LVEF, drug class, contraindications, answer consistency)
- **Backend**: FastAPI + Jinja2 templates (response caching with TTLCache)
- **Observability**: Full trace logging via Langfuse + structured JSON logging
- **PDF Processing**: PyMuPDF (text/images) + pdfplumber (tables) + Tesseract OCR
- **Code Quality**: Ruff (linting + formatting) + mypy (strict type checking)
- **Dependencies**: Poetry

## Data Files

- `data/rules/cor_lookup.csv` — 34-entry CSV mapping therapies to Class of Recommendation and Level of Evidence
- `eval/data/golden_dataset.json` — 40 curated QA pairs covering GDMT quadruple therapy, HF classifications, ICD indications, biomarker thresholds, contraindications, and more
- `prompts/system_prompt_structured.txt` — Preferred structured JSON output prompt with reasoning steps
- `prompts/system_prompt.txt` — Fallback plain-text output prompt

## Acknowledgments

American Heart Association (AHA) and American College of Cardiology (ACC) for the clinical source material.
