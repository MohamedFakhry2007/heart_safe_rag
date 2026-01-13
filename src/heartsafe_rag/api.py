"""
FastAPI application for HeartSafe RAG.
Exposes endpoints for chat, health checks, and a web UI.
"""
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.status import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR

from heartsafe_rag.generation import GenerationService
from heartsafe_rag.retrieval import RetrievalService
from heartsafe_rag.schemas import ChatRequest, ChatResponse, SourceDocument
from heartsafe_rag.utils.logger import logger

# Global services (initialized in lifespan)
services: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes heavy services (LLM, Vector DB) once.
    """
    try:
        logger.info("Starting up HeartSafe RAG API...")
        services["retrieval"] = RetrievalService()
        services["generation"] = GenerationService()
        logger.info("Services initialized successfully.")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise e
    finally:
        logger.info("Shutting down HeartSafe RAG API...")
        services.clear()

app = FastAPI(
    title="HeartSafe RAG API",
    version="1.0.0",
    description="Clinical AI Assistant for Heart Failure Management",
    lifespan=lifespan
)

# Setup Templates
templates = Jinja2Templates(directory="src/heartsafe_rag/templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    """Serve the Chat UI."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/health", status_code=HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "services": list(services.keys())}

@app.post("/chat", response_model=ChatResponse, status_code=HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint.
    Routes queries to RAG or General pipeline based on intent.
    """
    try:
        query = request.query
        gen_service: GenerationService = services["generation"]
        ret_service: RetrievalService = services["retrieval"]

        # 1. Route Query
        category = gen_service.route_query(query)
        
        context_docs = []
        
        # 2. Conditional Retrieval
        if category == "HF_RELATED":
            logger.info(f"Retrieving context for HF query: {query}")
            context_docs = ret_service.retrieve(query)
        
        # 3. Generate Response
        # We pass docs (empty or populated) to the generator
        answer = gen_service.generate_response(query, context_docs)
        
        # 4. Format Sources for Response
        sources_response = [
            SourceDocument(content=doc.page_content[:200] + "...", source=doc.metadata.get("source", "unknown"))
            for doc in context_docs
        ]

        return ChatResponse(
            answer=answer,
            category=category,
            sources=sources_response
        )

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An internal error occurred while processing your request."
        )