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
            metadata={"source": "guidelines.pdf", "page": 42},
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
    service.guard_chain = MagicMock()

    service.rag_chain.invoke.return_value = "Lasix dose is 40mg."
    service.guard_chain.invoke.return_value = {"is_grounded": True, "reason": "all claims supported"}

    response = service.generate_response("What is the dose of Lasix?", mock_docs)

    assert response == "Lasix dose is 40mg."
    service.rag_chain.invoke.assert_called_once()
    service.guard_chain.invoke.assert_called_once()


@patch("heartsafe_rag.generation.ChatGroq")
def test_generation_refuses_without_context(mock_chat_groq: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()

    response = service.generate_response("What is the dose of Lasix?", context_docs=[])

    assert "cannot provide a response" in response.lower()
    assert "no relevant guidelines" in response.lower()


@patch("heartsafe_rag.generation.ChatGroq")
def test_output_guard_rejects_ungrounded_answer(
    mock_chat_groq: MagicMock, mock_docs: list[Document]
) -> None:
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()
    service.rag_chain = MagicMock()
    service.guard_chain = MagicMock()

    service.rag_chain.invoke.return_value = "Lasix dose is 200mg."
    service.guard_chain.invoke.return_value = {
        "is_grounded": False,
        "reason": "The answer states 200mg but the context says 40mg",
    }

    response = service.generate_response("What is the dose of Lasix?", mock_docs)

    assert "cannot provide a response" in response.lower()
    assert "Lasix dose is 200mg" not in response


@patch("heartsafe_rag.generation.ChatGroq")
def test_generation_default_callbacks(
    mock_chat_groq: MagicMock, mock_docs: list[Document]
) -> None:
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()
    service.rag_chain = MagicMock()
    service.guard_chain = MagicMock()

    service.rag_chain.invoke.return_value = "answer"
    service.guard_chain.invoke.return_value = {"is_grounded": True, "reason": "ok"}

    response = service.generate_response("test query", mock_docs)

    assert response == "answer"
