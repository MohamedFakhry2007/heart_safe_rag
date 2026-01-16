"""
Tests for the generation module with routing logic.
"""

from unittest.mock import MagicMock, patch, ANY

import pytest
from langchain_core.documents import Document

from heartsafe_rag.generation import GenerationService


@pytest.fixture
def mock_docs():
    return [
        Document(page_content="Lasix is a diuretic used in Heart Failure.", metadata={"source": "guidelines"}),
    ]


@patch("heartsafe_rag.generation.ChatGroq")
def test_generation_routing_rag(mock_chat_groq, mock_docs):
    """Test that HF queries use the RAG path."""
    # Setup
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()

    # Mock the chains individually to control output
    service.router_chain = MagicMock()
    service.rag_chain = MagicMock()
    service.direct_chain = MagicMock()

    # 1. Simulate Router returning HF_RELATED
    service.router_chain.invoke.return_value = "HF_RELATED"

    # 2. Simulate RAG chain response
    service.rag_chain.invoke.return_value = "Lasix dose is 40mg."

    # Execute
    query = "What is the dose of Lasix?"
    response = service.generate_response(query, mock_docs)

    # Assertions
    assert response == "Lasix dose is 40mg."
    service.router_chain.invoke.assert_called_with(
        {"question": query}, config={"callbacks": ANY}
    )
    # Ensure RAG chain was called
    service.rag_chain.invoke.assert_called_once()
    # Ensure Direct chain was NOT called
    service.direct_chain.invoke.assert_not_called()


@patch("heartsafe_rag.generation.ChatGroq")
def test_generation_routing_general(mock_chat_groq, mock_docs):
    """Test that General queries use the Direct path."""
    # Setup
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm

    service = GenerationService()

    # Mock chains
    service.router_chain = MagicMock()
    service.rag_chain = MagicMock()
    service.direct_chain = MagicMock()

    # 1. Simulate Router returning GENERAL
    service.router_chain.invoke.return_value = "GENERAL"

    # 2. Simulate Direct chain response
    service.direct_chain.invoke.return_value = "Hello! How can I help?"

    # Execute
    query = "Hi there"
    # Even if docs are passed (by accident), they should be ignored
    response = service.generate_response(query, mock_docs)

    # Assertions
    assert response == "Hello! How can I help?"
    service.direct_chain.invoke.assert_called_once()
    service.rag_chain.invoke.assert_not_called()
