# Implementation Plan: HeartSafe RAG

## Overview

This implementation plan describes the concrete engineering tasks required to build **HeartSafe RAG**, a guideline-driven cardiology agent with **zero hallucinations**.  
The plan follows a **build-once, validate-early** philosophy and maps directly to the **requirements.md** and **design.md** documents.

The system is intentionally simpler than typical healthcare platforms:  
there is **no PHI**, **no fine-tuning**, and **no runtime ingestion**.  
Complexity is concentrated where it matters most: **retrieval correctness, determinism, and safety**.

---

## Tasks

---
- [ ] **0. Engineering Setup**
  - [ ] Initialize Git repository
  - [ ] Initialize Poetry (`poetry init`)
  - [ ] Create `standards.md`
  - [ ] Create `Makefile`
  - [ ] Create `.github/workflows/ci.yml`

- [ ] **0.5 Evaluation Framework (The "Senior" Touch)**
  - [ ] Create `eval/` directory
  - [ ] Compile `data/golden_dataset.json` (50 items) based on Guidelines
  - [ ] Write `evaluation.py` logic (Load data -> RAG -> Score)
  - [ ] Add `make eval` command


- [ ] **1. Initialize repository structure**
  - Create top-level folders:
    - `data/guidelines/`
    - `faiss_index/`
    - `prompts/`
  - Add placeholder files:
    - `README.md`
    - `requirements.md`
    - `design.md`
    - `tasks.md`
  - _Requirements: Repo Structure_

---

- [ ] **2. Define system prompt for hallucination prevention**
  - Create `prompts/system_prompt.txt`
  - Explicitly forbid:
    - External medical knowledge
    - Speculation
    - Advice not supported by guidelines
  - Include refusal template for insufficient context
  - _Requirements: Hallucination Prevention_

---

- [ ] **3. Set up dependencies**
  - Add core dependencies to `requirements.txt`:
    - `langchain`
    - `langchain-community`
    - `sentence-transformers`
    - `faiss-cpu`
    - `streamlit`
    - `groq`
    - `pypdf`
    - `python-docx`
    - `python-pptx`
  - Pin versions for reproducibility
  - _Requirements: Deterministic Builds_

---

- [ ] **4. Implement Multimodal Ingestion Pipeline**
  - Update `requirements.txt`: Add `pymupdf`, `pillow`.
  - Update `ingest.py` to use `fitz` (PyMuPDF).
  - Implement `analyze_image` function using `meta-llama/llama-4-scout-17b-16e-instruct` (Groq).
  - Add logic to filter out small icons (logos/footers).
  - Merge image descriptions into page text before chunking.

---

- [ ] **5. Implement chunking strategy**
  - Split documents into chunks:
    - ~500 characters
    - Configurable overlap (default 50–100 chars)
  - Preserve metadata:
    - Source file
    - Section headers (if available)
    - Guideline year
  - _Requirements: Chunking Strategy_

---

- [ ] **6. Generate embeddings**
  - Load embedding model:
    - `sentence-transformers/all-MiniLM-L6-v2`
  - Generate embeddings for all chunks
  - Validate no chunk exceeds model limits
  - _Requirements: Embedding Layer_

---

- [ ] **7. Persist FAISS index**
  - Create FAISS vector store from embeddings
  - Save artifacts to:
    - `faiss_index/index.faiss`
    - `faiss_index/index.pkl`
  - _Requirements: Offline Ingestion_

---

- [ ] **8. Persist text chunks for BM25**
  - Serialize chunk objects using `pickle`
  - Save to:
    - `faiss_index/chunks.pkl`
  - Ensure chunks include metadata
  - _Requirements: Hybrid Search Persistence_

---

- [ ] **9. Validate ingestion artifacts**
  - Confirm repository can be cloned with:
    - No ingestion required
    - All artifacts present
  - Fail fast if any artifact missing
  - _Requirements: Reproducibility_

---

- [ ] **10. Implement FAISS retriever**
  - Load FAISS index from disk
  - Wrap with LangChain retriever interface
  - _Requirements: Dense Retrieval_

---

- [ ] **11. Implement BM25 retriever**
  - Load chunks from `chunks.pkl`
  - Rebuild BM25 index at runtime
  - Validate search correctness
  - _Requirements: Sparse Retrieval_

---

- [ ] **12. Implement EnsembleRetriever**
  - Combine FAISS and BM25 retrievers
  - Configure hybrid weights
  - Return ranked results with scores
  - _Requirements: Hybrid Search_

---

- [ ] **13. Implement hallucination guard logic**
  - Enforce:
    - No retrieved docs → no answer
    - Retrieved context only → LLM input
  - Add refusal behavior for low confidence
  - _Requirements: Safety Guards_

---

- [ ] **14. Integrate Groq LLM**
  - Configure Groq client
  - Use model:
    - `llama-3.3-70b-versatile`
  - Load API key from environment
  - Set temperature ≤ 0.2
  - _Requirements: LLM Configuration_

---

- [ ] **15. Implement guarded LLM chain**
  - Inject:
    - System prompt
    - Retrieved guideline chunks
  - Disable any tool or memory usage
  - _Requirements: Retrieval-First Reasoning_

---

- [ ] **16. Build Streamlit application**
  - Create `app.py`
  - On startup:
    - Load FAISS index
    - Load chunks
    - Rebuild BM25
    - Initialize EnsembleRetriever
  - _Requirements: Streamlit Application_

---

- [ ] **17. Implement UI components**
  - Text input for clinician questions
  - Answer display panel
  - Retrieved guideline excerpts
  - Source citations
  - _Requirements: Explainability_

---

- [ ] **18. Enforce zero ingestion at runtime**
  - Ensure:
    - No document loading
    - No embedding generation
    - No index creation
  - Fail startup if ingestion artifacts missing
  - _Requirements: Performance_

---

- [ ] **19. Add structured logging**
  - Log:
    - Query
    - Retrieved chunks
    - Refusal events
    - LLM errors
  - Exclude any sensitive data
  - _Requirements: Production Readiness_

---

- [ ] **20. Write unit tests**
  - Chunk size enforcement
  - Embedding consistency
  - FAISS index loading
  - BM25 rebuild correctness
  - _Requirements: Correctness_

---

- [ ]* **21. Write property-based tests**
  - Retrieval determinism
  - Refusal on missing context
  - Context-only generation
  - _Requirements: Safety Guarantees_

---

- [ ] **22. End-to-end integration test**
  - Query → Retrieval → Answer
  - Verify citations always present
  - Verify refusal behavior
  - _Requirements: System Integration_

---

- [ ] **23. Deployment validation**
  - Run locally
  - Run on Streamlit Cloud
  - Confirm startup < 5 seconds
  - _Requirements: Deployability_

---

- [ ] **24. Documentation finalization**
  - Update `README.md` with:
    - Architecture summary
    - How ingestion works
    - How to deploy
    - Safety guarantees
  - _Requirements: Developer Experience_

---

- [ ] **25. Final checkpoint**
  - All tests passing
  - No hallucinations observed
  - All answers traceable to guidelines
  - Ready for public release

---

## Notes

- Tasks marked with `*` are optional for MVP
- This plan intentionally excludes:
  - PHI handling
  - Authentication
  - Fine-tuning
- The system’s credibility depends on **refusal correctness**, not answer verbosity

> **In HeartSafe RAG, silence is safer than speculation.**
