# Engineering Standards & Contribution Guidelines

## 1. Core Philosophy

**"If it isn't tested, it doesn't work. If it isn't typed, it isn't Python."**

We prioritize **readability**, **reproducibility**, and **type safety** over clever one-liners. This project uses a strict subset of Python to ensure production readiness in clinical environments.

---

## 2. Technology Stack

- **Language**: Python 3.11+
- **Dependency Manager**: `poetry` (Standardized environment)
- **Linter/Formatter**: `ruff` (Replaces Black, Isort, Flake8)
- **Static Analysis**: `mypy` (Strict mode)
- **Testing**: `pytest`

---

## 3. Coding Conventions

### A. Type Hinting

**Requirement**: All function signatures must be fully type-hinted.  
**Reason**: Clinical systems cannot afford `NoneType` errors at runtime.

```python
# ❌ BAD
def search(query, k=5):
    ...

# ✅ GOOD
def search(query: str, k: int = 5) -> list[RetrievalResult]:
    ...
```

---

### B. Documentation (Docstrings)

**Requirement**: Google-style docstrings for all public modules, classes, and methods.  
**Reason**: Explain *why* a decision was made, not just what the code does.

```python
def retrieve_guidelines(query: str) -> list[str]:
    """
    Retrieves relevant chunks using hybrid search (FAISS + BM25).

    Args:
        query (str): The clinician's natural language query.

    Returns:
        list[str]: A list of text chunks sorted by relevance score.
    """
    ...
```

---

### C. Error Handling

**Requirement**: Never catch generic Exception without re-raising or logging stack traces.  
Use custom exception classes for domain errors.

```python
# ✅ GOOD
class GuidelineNotFoundError(Exception):
    """Raised when no valid guideline content matches the query."""
    pass
```

---

## 4. Operational Excellence

### A. Configuration

No hardcoded paths or API keys.  
Use `pydantic-settings` to load from `.env`.

### B. Logging

Use the centralized logger in `src.utils`.

- INFO: High-level flow (e.g., `"Ingestion complete"`).
- DEBUG: Detailed steps (e.g., `"Chunking file X, size Y"`).
- ERROR: System failures (must include stack trace).

### C. Git Workflow

- **Commits**: Conventional Commits  
  Example:  
  `feat: add hybrid search`  
  `fix: token limit overflow`
- **Pre-commit**: Run `make check` before pushing.
