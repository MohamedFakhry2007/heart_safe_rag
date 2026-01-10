# Requirements Document

## Project Title
**HeartSafe RAG: Guideline-Driven Cardiology Agent**  
**Zero-Hallucination Heart Failure Decision Support**

---

## 1. Introduction

HeartSafe RAG is a guideline-driven Retrieval-Augmented Generation (RAG) system designed to support **heart failure clinical decision-making** using authoritative **AHA/ACC Heart Failure Guidelines**.  
The system is explicitly engineered to **eliminate hallucinations** by grounding every response in verifiable guideline content and exposing citations transparently.

This document defines the functional, technical, and architectural requirements necessary to build a **production-ready, reproducible GitHub repository** that can be deployed locally, on Streamlit, or in cloud environments with **zero re-ingestion cost**.

---

## 2. Goals & Non-Goals

### Goals
- Provide **accurate, guideline-grounded answers** for heart failure management
- Enforce **retrieval-first reasoning** (no free-form clinical speculation)
- Support **offline ingestion + online inference**
- Enable **hybrid search (dense + sparse)** for clinical recall and precision
- Ensure **deterministic, reproducible builds** across environments

### Non-Goals
- This system does **not** provide medical diagnosis
- This system does **not** replace clinician judgment
- This system does **not** fine-tune LLMs
- This system does **not** store or process PHI

---

## 3. Glossary

- **HeartSafe_RAG**: The cardiology-focused RAG system
- **AHA/ACC**: American Heart Association / American College of Cardiology
- **HF**: Heart Failure
- **RAG**: Retrieval-Augmented Generation
- **FAISS**: Facebook AI Similarity Search (vector index)
- **BM25**: Sparse keyword-based ranking algorithm
- **Hybrid_Search**: Combined dense (FAISS) + sparse (BM25) retrieval
- **Groq_API**: High-performance inference API for LLMs
- **Chunks**: Text segments created during ingestion
- **EnsembleRetriever**: LangChain retriever combining FAISS and BM25
- **LLM**: Large Language Model

---

## 4. System Architecture Overview

┌──────────────┐
│ AHA/ACC Docs │ (PDF, PPTX, DOCX, TXT)
└──────┬───────┘
│
▼
┌─────────────────────┐
│ ingest.py │
│ - Load documents │
│ - Chunk text │
│ - Create embeddings │
│ - Save FAISS index │
│ - Save text chunks │
└──────┬──────────────┘
│
▼
┌──────────────────────────┐
│ faiss_index/ │
│ - index.faiss │
│ - index.pkl │
│ - chunks.pkl │
└──────┬───────────────────┘
│
▼
┌──────────────────────────┐
│ app.py (Streamlit) │
│ - Load FAISS index │
│ - Rebuild BM25 │
│ - Hybrid retrieval │
│ - LLM response │
└──────────┬───────────────┘
▼
┌──────────────────────────┐
│ llama-3.3-70b-versatile │
│ (Groq Console API) │
└──────────────────────────┘

yaml
Copy code

---

## 5. Requirements

---

## Requirement 1: Guideline-Only Knowledge Base

**User Story:**  
As a clinician, I want answers strictly derived from AHA/ACC heart failure guidelines, so that responses are evidence-based and trustworthy.

### Acceptance Criteria

1. THE HeartSafe_RAG SHALL ingest only guideline-related documents
2. THE system SHALL support:
   - PDF
   - PPTX
   - DOCX
   - TXT
3. EACH document SHALL store source metadata (title, section, guideline year)
4. THE system SHALL reject unsupported or non-guideline files
5. ALL responses SHALL cite retrieved guideline chunks
6. IF no relevant guideline content is found, THE system SHALL respond:
   > “No guideline-supported answer found.”

---

## Requirement 2: Deterministic Offline Data Ingestion

**User Story:**  
As a developer, I want to ingest guidelines once and reuse them everywhere, so deployment is instant and cost-free.

### Acceptance Criteria

1. `ingest.py` SHALL:
   - Load documents from `/data/guidelines/`
   - Split text into chunks
   - Generate embeddings
   - Persist outputs locally
2. FAISS indexes SHALL be saved to:
faiss_index/
├── index.faiss
├── index.pkl

css
Copy code
3. Text chunks SHALL be saved separately:
faiss_index/chunks.pkl

yaml
Copy code
4. The ingestion process SHALL NOT run at app startup
5. The repository SHALL include the prebuilt `faiss_index/` directory
6. Streamlit deployment SHALL load indexes instantly without recomputation

---

## Requirement 3: Chunking Strategy Optimized for Embeddings

**User Story:**  
As an ML engineer, I want chunk sizes optimized for the embedding model, so semantic meaning is preserved.

### Acceptance Criteria

1. The embedding model SHALL be:
sentence-transformers/all-MiniLM-L6-v2

yaml
Copy code
2. Chunk size SHALL be ~500 characters
3. Chunk overlap SHALL be configurable (default: 50–100 characters)
4. No chunk SHALL exceed the embedding model’s effective token limit
5. Sentence boundaries SHOULD be preserved where possible

---

## Requirement 4: Hybrid Search with Persistence

**User Story:**  
As a clinician, I want both semantic and keyword-based retrieval, so I can find exact guideline phrases and conceptual answers.

### Acceptance Criteria

1. THE system SHALL use LangChain `EnsembleRetriever`
2. Dense retrieval SHALL be implemented using FAISS
3. Sparse retrieval SHALL be implemented using BM25
4. Because BM25 is in-memory:
- Chunks MUST be persisted during ingestion
- BM25 MUST be rebuilt at runtime from saved chunks
5. Hybrid weighting SHALL be configurable
6. Retrieval results SHALL include relevance scores

---

## Requirement 5: Hallucination Prevention & Safety Guards

**User Story:**  
As a clinician, I want zero hallucinations, so I can trust the system.

### Acceptance Criteria

1. The LLM SHALL only answer using retrieved context
2. System prompts SHALL explicitly forbid:
- External medical knowledge
- Speculation
- Clinical advice beyond guidelines
3. Responses SHALL:
- Quote guideline language where possible
- Include section-level citations
4. If retrieval confidence is low:
- The system SHALL refuse to answer
5. Temperature SHALL default to ≤ 0.2

---

## Requirement 6: LLM Configuration

**User Story:**  
As a system architect, I want fast, reliable inference at scale.

### Acceptance Criteria

1. The chatbot LLM SHALL be:
llama-3.3-70b-versatile

yaml
Copy code
2. Inference SHALL use Groq Console API
3. API keys SHALL be loaded from environment variables
4. Token usage and latency SHOULD be logged
5. The system SHALL fail gracefully if the API is unavailable

---

## Requirement 7: Streamlit Application

**User Story:**  
As a user, I want an intuitive UI to query guidelines interactively.

### Acceptance Criteria

1. The UI SHALL be built with Streamlit
2. The app SHALL:
- Load FAISS and chunks on startup
- Rebuild BM25 retriever
- Initialize EnsembleRetriever
3. The UI SHALL show:
- User question
- Answer
- Retrieved guideline excerpts
- Source citations
4. No ingestion SHALL occur in the UI
5. Startup time SHALL be < 5 seconds

---

## Requirement 8: Repository Structure

**User Story:**  
As an open-source contributor, I want a clean, reproducible repo.

### Acceptance Criteria

HeartSafe-RAG/
├── app.py
├── ingest.py
├── requirements.txt
├── requirements.md
├── README.md
├── data/
│ └── guidelines/
├── faiss_index/
│ ├── index.faiss
│ ├── index.pkl
│ └── chunks.pkl
└── prompts/
└── system_prompt.txt

yaml
Copy code

---

## Requirement 9: Explainability & Transparency

**User Story:**  
As a clinician, I want to see *why* an answer was given.

### Acceptance Criteria

1. Each answer SHALL display:
   - Guideline section
   - Retrieved text chunks
2. The system SHALL allow inspection of retrieved context
3. The system SHALL not hide or summarize away evidence

---

## Requirement 10: Production Readiness

**User Story:**  
As a platform owner, I want a system ready for real deployment.

### Acceptance Criteria

1. The system SHALL support `.env` configuration
2. Logging SHALL be structured and readable
3. Errors SHALL be non-fatal where possible
4. The system SHALL be deployable on:
   - Local machine
   - Streamlit Cloud
5. No proprietary data SHALL be committed

---

## 6. Success Criteria

The project is considered successful when:

- Answers are **100% grounded in guidelines**
- No hallucinations are observed
- Streamlit loads instantly without ingestion
- Hybrid retrieval improves recall and precision
- Clinicians can trace every answer to its source

---

## 7. Guiding Principle

> **If it’s not in the guideline, it’s not in the answer.**

## 11. Engineering Standards (Added)

**User Story:** As a Lead Engineer, I want the codebase to be maintainable, typed, and automatically tested.

### Acceptance Criteria
1. **Type Safety**: All code MUST pass `mypy --strict`.
2. **Linting**: All code MUST adhere to `ruff` standard configuration.
3. **Evaluation**: An `eval/` directory MUST exist containing:
    - `golden_dataset.json`: Minimum 50 QA pairs.
    - `evaluation.py`: Script to run the RAG against the dataset and output accuracy %.
4. **CI/CD**: GitHub Actions MUST trigger on push to validate types and linting.
5. **Configuration**: Use `pydantic-settings` for environment variables.