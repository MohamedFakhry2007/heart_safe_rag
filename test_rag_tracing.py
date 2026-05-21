# test_rag_tracing.py
import asyncio
import sys
from pathlib import Path

from langchain_core.documents import Document

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from langfuse import get_client, observe

from heartsafe_rag.generation import GenerationService
from heartsafe_rag.retrieval import RetrievalService


@observe(name="retrieval", as_type="span")
def trace_retrieval(retrieval_service, query: str) -> list[Document]:
    """Helper function to trace the retrieval step."""
    langfuse = get_client()
    try:
        # Use the standard get_relevant_documents method
        docs = retrieval_service.retriever.get_relevant_documents(query)

        # Get sample sources (up to 3 unique sources)
        sample_sources = []
        seen_sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                sample_sources.append(source)
                if len(sample_sources) >= 3:
                    break

        # Update the span with input and output
        langfuse.update_current_span(
            input={"query": query},
            output={
                "retrieved_docs_count": len(docs),
                "sample_sources": sample_sources,
                "sample_content": [doc.page_content[:100] + "..." for doc in docs[:3]]
            }
        )

        return docs

    except Exception as e:
        error_msg = f"Retrieval failed: {e!s}"
        langfuse.update_current_span(
            level="ERROR",
            status_message=error_msg
        )
        raise Exception(error_msg) from e

@observe(name="generation", as_type="span")
def trace_generation(generation_service, query: str, context_docs: list[Document]) -> str:
    """Helper function to trace the generation step."""
    langfuse = get_client()
    try:
        response = generation_service.generate_response(
            query=query,
            context_docs=context_docs
        )
        langfuse.update_current_span(
            input={
                "query": query,
                "context_docs_count": len(context_docs)
            },
            output={"response": response}
        )
        return response
    except Exception as e:
        langfuse.update_current_span(status_message=f"Generation failed: {e!s}")
        raise

@observe(name="rag-pipeline")
async def test_rag_pipeline(query: str):
    """Test the complete RAG pipeline with tracing."""
    langfuse = get_client()
    try:
        # Initialize services
        retrieval_service = RetrievalService()
        generation_service = GenerationService()

        # 1. Retrieve documents with tracing
        docs = trace_retrieval(retrieval_service, query)

        # 2. Generate response with tracing
        response = trace_generation(generation_service, query, docs)


        langfuse.update_current_trace(
            input={"query": query},
            output={"response": response}
        )

        return response

    except Exception:
        raise

if __name__ == "__main__":
    # Initialize Langfuse
    langfuse = get_client()

    QUERY = "when to use beta blockers in heart failure management?"

    try:
        # Run the pipeline
        response = asyncio.run(test_rag_pipeline(QUERY))

        # Ensure all data is flushed to Langfuse
        langfuse.flush()


    except Exception:
        pass
    finally:
        # Clean up
        langfuse.flush()
