import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.status import HTTP_200_OK, HTTP_408_REQUEST_TIMEOUT, HTTP_500_INTERNAL_SERVER_ERROR

from heartsafe_rag.config import settings
from heartsafe_rag.generation import GenerationService
from heartsafe_rag.retrieval import HybridRetriever
from heartsafe_rag.schemas import ChatRequest, ChatResponse, SourceDocument
from heartsafe_rag.utils.logger import logger

services: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        logger.info("Starting up HeartSafe RAG API...")
        services["retrieval"] = HybridRetriever()
        services["generation"] = GenerationService()
        logger.info("Services initialized successfully.")
        yield
    except Exception:
        logger.exception("Startup failed")
        raise
    finally:
        logger.info("Shutting down HeartSafe RAG API...")
        services.clear()


app = FastAPI(
    title="HeartSafe RAG API",
    version="1.0.0",
    description="Clinical AI Assistant for Heart Failure Management",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="src/heartsafe_rag/templates")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health", status_code=HTTP_200_OK)
async def health_check() -> dict[str, str | list[str]]:
    return {"status": "healthy", "services": list(services.keys())}


@app.post("/chat", response_model=ChatResponse, status_code=HTTP_200_OK)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    request_id = uuid.uuid4().hex[:8]
    query = request.query
    log_extra = {"request_id": request_id}
    t_start = time.perf_counter()

    logger.info(f"Chat request [{request_id}] query='{query}'", extra=log_extra)

    try:
        gen_service: GenerationService = services["generation"]
        ret_service: HybridRetriever = services["retrieval"]

        context_docs = await asyncio.wait_for(
            asyncio.to_thread(ret_service.retrieve, query),
            timeout=settings.LLM_TIMEOUT,
        )

        answer = await asyncio.wait_for(
            asyncio.to_thread(gen_service.generate_response, query, context_docs),
            timeout=settings.LLM_TIMEOUT,
        )

        sources_response = [
            SourceDocument(
                content=doc.page_content[:200] + "...",
                source=doc.metadata.get("source", "unknown"),
            )
            for doc in context_docs
        ]

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Chat response [{request_id}] in {elapsed:.2f}s "
            f"answer_preview='{answer[:100]}...' sources={len(sources_response)}",
            extra=log_extra,
        )

        return ChatResponse(answer=answer, sources=sources_response)

    except TimeoutError:
        elapsed = time.perf_counter() - t_start
        logger.error(
            f"Chat timeout [{request_id}] after {elapsed:.2f}s",
            extra=log_extra,
        )
        raise HTTPException(
            status_code=HTTP_408_REQUEST_TIMEOUT,
            detail="Request timed out. Please try again.",
        ) from None

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        logger.error(
            f"Chat error [{request_id}] after {elapsed:.2f}s: {e!s}",
            extra=log_extra,
            exc_info=True,
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request.",
        ) from e
