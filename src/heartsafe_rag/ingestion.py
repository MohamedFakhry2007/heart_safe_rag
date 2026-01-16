"""
Ingestion logic for HeartSafe RAG.
Handles PDF processing, Vision-LLM analysis, and Vector Store creation.
"""

import base64
import io
import time
from pathlib import Path

import fitz  # type: ignore
from langchain.schema import Document, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from PIL import Image

from heartsafe_rag.chunking import GuidelineChunker
from heartsafe_rag.config import settings
from heartsafe_rag.exceptions import DocumentProcessingError
from heartsafe_rag.utils.logger import logger

# Initialize Vision LLM
# Note: ChatGroq v0.2+ supports passing SecretStr directly for api_key
vision_llm = ChatGroq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,  # Passing SecretStr directly to satisfy type check
    temperature=settings.LLM_TEMPERATURE,
)


def _encode_image(pil_image: Image.Image) -> str:
    """Helper: Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _analyze_image_content(pil_image: Image.Image) -> str:
    """Helper: Send image to Groq Vision for description."""
    try:
        base64_image = _encode_image(pil_image)
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this medical image/chart. Transcribe logic/tables in detail. If decorative, say 'Decorative'.",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        response = vision_llm.invoke([msg])
        if not response or not response.content:
            return ""
        return str(response.content)
    except Exception as e:
        logger.error(f"Vision API failed: {e!s}")
        return ""


def process_pdf_page(doc: fitz.Document, page_num: int) -> str:
    """Extracts text and image descriptions from a single page."""
    try:
        page = doc.load_page(page_num)
        text_content = page.get_text() or ""

        image_descriptions = []
        image_list = page.get_images(full=True)

        if image_list:
            for img in image_list:
                try:
                    # Extract and process each image
                    base_image = doc.extract_image(img[0])
                    image_bytes = base_image["image"]

                    # Process Image
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    desc = _analyze_image_content(pil_img)

                    if desc and "Decorative" not in desc and desc.strip():
                        image_descriptions.append(f"\n[IMAGE CONTEXT]\n{desc}\n[/IMAGE CONTEXT]\n")

                    time.sleep(1.0)  # Rate limit kindness
                except Exception as e:
                    logger.error(f"Error processing image on page {page_num + 1}: {e}")
                    continue

        return text_content + "".join(image_descriptions)
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {e}")
        return ""


def run_ingestion_pipeline(pdf_path: Path, output_dir: Path) -> None:
    """Orchestrates the full ingestion flow: Load -> Chunk -> Embed -> Save."""

    # 1. Validation
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"🚀 Starting ingestion for: {pdf_path.name}")

    # 2. Extract Raw Content (Text + Vision)
    raw_docs: list[Document] = []
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            for i in range(total_pages):
                logger.info(f"Processing page {i + 1}/{total_pages}...")
                content = process_pdf_page(doc, i)

                meta = {"source": pdf_path.name, "page": i + 1}
                raw_docs.append(Document(page_content=content, metadata=meta))
    except Exception as e:
        raise DocumentProcessingError(f"Failed to read PDF: {e!s}")

    # 3. Chunking
    logger.info("✂️ Chunking documents...")
    chunker = GuidelineChunker()
    chunks = chunker.split_documents(raw_docs)

    # 4. Save Chunks (Pickle for BM25)
    import pickle

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"💾 Saved {len(chunks)} chunks to {chunks_path}")

    # 5. Embedding (FAISS)
    logger.info("🧠 Generating Embeddings & FAISS Index...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(output_dir))
    logger.info(f"✅ FAISS index saved to {output_dir}")
