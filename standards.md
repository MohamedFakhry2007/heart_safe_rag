# Engineering Standards & Contribution Guidelines

## 1. Core Philosophy
**"Reliability is not an afterthought; it is the primary feature."**

We prioritize **readability**, **reproducibility**, and **type safety** over clever one-liners. This project uses a strict subset of Python to ensure production readiness in clinical environments. Code should be written as if the person maintaining it is a homicidal psychopath who knows where you live—make it clear, safe, and obvious.

## 2. Technology Stack
- **Language**: Python 3.11+
- **Dependency Manager**: `poetry` (Standardized environment)
- **Linter/Formatter**: `ruff` (Replaces Black, Isort, Flake8)
- **Static Analysis**: `mypy` (Strict mode)
- **Testing**: `pytest`

## 3. Architecture & Design Patterns

### A. SOLID Principles
- **Single Responsibility**: Each class or function should do one thing and do it well. If a function has `and` in its name, it's likely doing too much.
- **Dependency Inversion**: High-level modules should not import low-level modules. Both should depend on abstractions (Protocol or ABC).

### B. Data Validation (Pydantic)
**Requirement**: Use Pydantic models for all data crossing boundaries (API requests, database records, config files).
**Reason**: Fail fast at the boundary rather than deep in the logic.

```python
# ✅ GOOD
from pydantic import BaseModel, Field

class PatientQuery(BaseModel):
    text: str = Field(..., min_length=5, description="Clinical query")
    max_results: int = Field(default=3, ge=1, le=10)
```

### C. Functional Core, Imperative Shell
Keep business logic pure (no side effects) where possible. Push I/O (DB calls, API requests) to the edges of the application.

## 4. Coding Conventions

### A. Type Hinting
**Requirement**: All function signatures must be fully type-hinted. Use `typing.Optional`, `typing.List`, or modern `|` syntax.
**Reason**: Clinical systems cannot afford `NoneType` errors at runtime.

```python
# ❌ BAD
def search(query, k=5):
    ...

# ✅ GOOD
def search(query: str, k: int = 5) -> list[RetrievalResult]:
    ...
```

### B. Documentation (Docstrings)
**Requirement**: Google-style docstrings for all public modules, classes, and methods.
**Reason**: Explain *why* a decision was made, not just *what* the code does.

```python
def retrieve_guidelines(query: str) -> list[str]:
    """
    Retrieves relevant chunks using hybrid search (FAISS + BM25).

    We use hybrid search to balance semantic understanding (Dense) with
    keyword exact matching (Sparse), which is critical for specific medical terminology.

    Args:
        query (str): The clinician's natural language query.

    Returns:
        list[str]: A list of text chunks sorted by relevance score.
    """
    ...
```

### C. Naming Conventions
- **Variables/Functions**: `snake_case`. Functions should be verbs (`calculate_risk`, `fetch_patient`).
- **Classes**: `PascalCase`. Should be nouns (`RiskCalculator`, `PatientRepository`).
- **Constants**: `UPPER_CASE`.
- **Privates**: `_leading_underscore` to indicate internal use.

### D. Error Handling
**Requirement**: Never catch generic `Exception` without re-raising or logging stack traces. Use custom exception classes for domain errors.

```python
# ✅ GOOD
class GuidelineNotFoundError(Exception):
    """Raised when no valid guideline content matches the query."""
    pass

try:
    result = vector_store.search(query)
except VectorStoreConnectionError as e:
    logger.error(f"Vector store unavailable: {e}")
    raise SystemUnavailableError from e
```

## 5. Testing Standards

### A. The Testing Pyramid
1.  **Unit Tests (80%)**: Test individual functions/classes in isolation. Mock external dependencies.
2.  **Integration Tests (15%)**: Test interactions between modules (e.g., API -> Service -> DB).
3.  **E2E Tests (5%)**: Test the full flow from user input to output.

### B. Guidelines
- **No Global State**: Tests must be independent. Use `pytest` fixtures.
- **Mock External Calls**: Never hit real APIs in unit tests. Use `unittest.mock` or libraries like `respx`.
- **Coverage**: Aim for high branch coverage, especially in logic-heavy components.

## 6. Security & Privacy

### A. Data Handling
- **PII**: Never log Patient Health Information (PHI) or PII. Sanitize logs.
- **Input Sanitization**: Validate all inputs using Pydantic. Prevent injection attacks.

### B. Secrets
- **No Hardcoded Secrets**: Credentials must strictly come from environment variables.
- **.gitignore**: Ensure `.env` and `__pycache__` are ignored.

## 7. Performance & Efficiency

- **Lazy Loading**: Import heavy libraries (like `torch` or `transformers`) only when needed or at the module level if used globally but consider startup time.
- **Generators**: Use generators (`yield`) for processing large datasets to save memory.
- **Async I/O**: Use `async`/`await` for I/O-bound operations (network, disk) in FastAPI routes.

## 8. Operational Excellence

### A. Configuration
No hardcoded paths or API keys. Use `pydantic-settings` to load from `.env`.

### B. Logging
Use the centralized logger in `src.utils`.
- **INFO**: High-level flow (e.g., "Ingestion complete").
- **DEBUG**: Detailed steps (e.g., "Chunking file X, size Y").
- **ERROR**: System failures (must include stack trace).

### C. Git Workflow
- **Commits**: Conventional Commits (e.g., `feat: add hybrid search`, `fix: token limit overflow`).
- **Pre-commit**: Run `make check` (format + lint + test) before pushing.
