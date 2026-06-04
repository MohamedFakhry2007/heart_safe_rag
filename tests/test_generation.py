from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from heartsafe_rag.generation import GenerationService


@pytest.fixture
def mock_docs() -> list[Document]:
    return [
        Document(
            page_content="Lasix (furosemide) is a loop diuretic used in Heart Failure. "
            "Typical starting dose is 40 mg once daily.",
            metadata={"source": "guidelines.pdf", "page": 42, "chunk_id": 0},
        ),
    ]


@patch("heartsafe_rag.generation.ChatGroq")
def test_generation_rag_path_with_context(
    mock_chat_groq: MagicMock, mock_docs: list[Document]
) -> None:
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()
    service.rag_chain = MagicMock()

    service.rag_chain.invoke.return_value = '{"reasoning_steps": [{"step": "Lasix is dosed at 40mg.", "claim_type": "recommendation", "sources": [0]}], "answer": "Lasix dose is 40mg."}'

    response = service.generate_response("What is the dose of Lasix?", mock_docs)

    assert response.answer == "Lasix dose is 40mg."
    assert len(response.reasoning_steps) == 1
    assert response.sources[0].chunk_index == 0
    service.rag_chain.invoke.assert_called_once()


@patch("heartsafe_rag.generation.ChatGroq")
def test_generation_refuses_without_context(mock_chat_groq: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()

    response = service.generate_response("What is the dose of Lasix?", context_docs=[])

    assert "cannot provide a response" in response.answer
    assert "do not contain information" in response.answer


@patch("heartsafe_rag.generation.ChatGroq")
def test_generation_default_callbacks(
    mock_chat_groq: MagicMock, mock_docs: list[Document]
) -> None:
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()
    service.rag_chain = MagicMock()

    service.rag_chain.invoke.return_value = '{"reasoning_steps": [{"step": "The dose is 40mg.", "claim_type": "recommendation"}], "answer": "answer"}'

    response = service.generate_response("test query", mock_docs)

    assert response.answer == "answer"


@patch("heartsafe_rag.generation.ChatGroq")
def test_source_grounding_in_reasoning_steps(
    mock_chat_groq: MagicMock, mock_docs: list[Document]
) -> None:
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()
    service.rag_chain = MagicMock()

    service.rag_chain.invoke.return_value = (
        '{"reasoning_steps": ['
        '{"step": "Lasix is dosed at 40mg [Chunk 0].", "claim_type": "recommendation", "sources": [0]},'
        '{"step": "It can be titrated up to 80mg if needed [Chunk 1].", "claim_type": "recommendation", "sources": [1]}'
        '], "answer": "Lasix dose is 40mg."}'
    )

    response = service.generate_response("What is the dose of Lasix?", mock_docs)

    assert len(response.reasoning_steps) == 2
    assert response.reasoning_steps[0].source_indices == [0]
    assert response.reasoning_steps[1].source_indices == [1]
