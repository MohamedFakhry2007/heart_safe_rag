# HeartSafe RAG: Guideline-Driven Cardiology Agent

A Retrieval-Augmented Generation (RAG) system for zero-hallucination heart failure decision support, grounded exclusively in AHA/ACC Heart Failure Guidelines.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Langfuse](https://img.shields.io/badge/Langfuse-Observability-orange)](https://langfuse.com)

## 🚀 Features

- **Evidence-Based Responses**: Every response is grounded in 2022 AHA/ACC Heart Failure Guidelines.
- **Zero Hallucinations**: Strict retrieval enforcement ensures no made-up information; the model refuses to answer if guidelines are missing.
- **Hybrid Retrieval**: Combines semantic search (FAISS) with keyword matching (BM25) for high-precision context fetching.
- **Evaluation Pipeline**: Integrated "LLM-as-a-Judge" workflow using Langfuse for tracking correctness, safety, and guideline adherence over time.
- **Production-Ready**: FastAPI backend with health checks, structured logging, and Docker support.
- **User Interface**: Built-in chat interface for easy interaction.

## 🏗️ Architecture

```mermaid
%% ---------------
%% RAG pipeline
%% ---------------
flowchart TB
    %% -------- OFFLINE --------
    subgraph OFFLINE ["📥 Offline Processing"]
        direction TB
        A[Guideline PDFs] --> B[Document Loading]
        B --> C[Recursive Chunking]
        C --> D[Embedding<br/>HuggingFace]
        D --> E[FAISS Index + BM25]
    end

    %% -------- ONLINE ---------
    subgraph ONLINE ["🌐 Online Serving"]
        direction TB
        F[User Query] --> G[Router<br/>Classifier]
        G -->|HF Related| H[Hybrid Retrieval]
        G -->|General|   K[Direct Response]
        H --> I[LLM Generation<br/>Llama-3]
        I --> J[Response with<br/>Citations]
    end

    %% ---- optional legend ----
    style OFFLINE fill:#ffeaa7,stroke:#fdcb6e
    style ONLINE  fill:#74b9ff,stroke:#0984e3
```

## 🛠️ Installation

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
```

## 🚀 Quick Start

### 1. Ingest Guidelines
Parse the PDFs and build the vector index.

```bash
poetry run python src/heartsafe_rag/ingest.py
```

### 2. Start the API Server
Launch the backend and the Chat UI.

```bash
poetry run uvicorn heartsafe_rag.api:app --reload
```

- **Chat UI**: Open http://localhost:8000 in your browser
- **API Docs**: Open http://localhost:8000/docs

### 3. Run Evaluations (The "Judge")
Run the Golden Dataset against the current pipeline and log results to Langfuse.

```bash
poetry run python src/heartsafe_rag/evaluate.py
```

## 🧪 Testing

```bash
# Run unit and integration tests
poetry run pytest
```

## 📂 Project Structure

.
├── data/                   # Data storage
│   ├── raw_pdfs/           # PDF Guidelines source
│   ├── vector_store/       # Generated FAISS Index
│   └── bm25_index.pkl      # Generated Keyword Index
├── eval/                   # Evaluation Suite
│   ├── data/               # Golden Datasets (QA pairs)
│   └── results/            # Local evaluation reports
├── src/                    # Source Code
│   └── heartsafe_rag/
│       ├── api.py          # FastAPI endpoints & UI
│       ├── config.py       # Pydantic Settings
│       ├── evaluate.py     # LLM Judge & Langfuse Experiment Runner
│       ├── generation.py   # LLM Chains & Routing Logic
│       ├── ingest.py       # ETL Pipeline
│       ├── retrieval.py    # Hybrid Search Logic
│       └── utils/          # Logger & Helpers
├── templates/              # HTML Templates for UI
├── tests/                  # Pytest Suite
└── docker-compose.yml      # Container Orchestration

## 📚 Documentation

- **LLM**: Llama-3-70b (via Groq) for high-fidelity medical reasoning
- **Embeddings**: all-MiniLM-L6-v2 for efficient semantic search
- **Observability**: Full trace logging via Langfuse

## 🙏 Acknowledgments

American Heart Association (AHA) and American College of Cardiology (ACC) for the clinical source material.