"""PDF ingestion module for HeartSafe RAG system.

This module provides functionality to process PDF documents, extract text and images,
and generate rich document representations with image descriptions using Groq Vision.
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, BinaryIO

import fitz  # PyMuPDF
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PIL import Image
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.docstore.document import Document

from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger
from heartsafe_rag.chunking import GuidelineChunker
from heartsafe_rag.exceptions import (
    DocumentProcessingError,
    ImageProcessingError,
    LLMError,
    ValidationError,
)

# Initialize Groq client
vision_llm = ChatGroq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=settings.LLM_TEMPERATURE,
)


def encode_image(pil_image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded string for the Groq Vision API.

    Args:
        pil_image: PIL Image object to encode.

    Returns:
        Base64-encoded string representation of the image in JPEG format.

    Raises:
        ImageProcessingError: If there's an error encoding the image.
    """
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        error_msg = f"Failed to encode image: {str(e)}"
        logger.error(error_msg)
        raise ImageProcessingError(error_msg) from e


def analyze_image(pil_image: Image.Image) -> str:
    """Analyze an image using Groq Vision to generate a text description.

    Args:
        pil_image: PIL Image object to analyze.

    Returns:
        Textual description of the image content.

    Raises:
        ImageProcessingError: If the image cannot be processed.
        LLMError: If there's an error communicating with the Groq API.
    """
    try:
        base64_image = encode_image(pil_image)
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Analyze this medical image/chart. If it is a flowchart or table, "
                        "transcribe its logic in detail. If it is decorative, say 'Decorative'."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
        
        logger.debug("Sending image to Groq Vision for analysis")
        response = vision_llm.invoke([msg])
        return response.content
        
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if isinstance(e, ImageProcessingError):
            raise
        raise LLMError(error_msg) from e


def process_image(
    doc: fitz.Document,
    img: tuple,
    min_size_bytes: int = 5000,
    rate_limit_delay: float = 1.0,
) -> Optional[str]:
    """Process a single image from a PDF page.

    Args:
        doc: The PDF document containing the image.
        img: The image tuple from PyMuPDF's get_images() method.
        min_size_bytes: Minimum size in bytes for an image to be processed.
        rate_limit_delay: Delay in seconds between API calls to respect rate limits.

    Returns:
        Optional[str]: Description of the image if it was processed, None otherwise.
    """
    xref = img[0]
    try:
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        # Skip small images (likely icons/logos)
        if len(image_bytes) < min_size_bytes:
            logger.debug(f"Skipping small image (size: {len(image_bytes)} bytes < {min_size_bytes} bytes)")
            return None

        pil_image = Image.open(io.BytesIO(image_bytes))
        description = analyze_image(pil_image)

        # Respect rate limiting
        time.sleep(rate_limit_delay)

        if "Decorative" not in description:
            return f"\n[IMAGE DESCRIPTION START]\n{description}\n[IMAGE DESCRIPTION END]\n"
        return None

    except Exception as e:
        logger.warning(f"Failed to process image (xref: {xref}): {str(e)}")
        return None


def ingest_pdf_with_vision(file_path: Path) -> List[Document]:
    """Process a PDF file, extracting text and enhancing it with image descriptions.

    Args:
        file_path: Path to the PDF file to process.

    Returns:
        List of LangChain Document objects containing page content and metadata.

    Raises:
        DocumentProcessingError: If there's an error processing the PDF.
        ValidationError: If the input file is invalid.
    """
    if not file_path.exists():
        error_msg = f"PDF file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if file_path.suffix.lower() != ".pdf":
        error_msg = f"Unsupported file format: {file_path}. Only PDF files are supported."
        logger.error(error_msg)
        raise ValidationError(error_msg)

    logger.info(f"Processing PDF: {file_path}")
    documents: List[Document] = []

    try:
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            logger.info(f"Document contains {total_pages} pages")

            for page_num, page in enumerate(doc, start=1):
                logger.debug(f"Processing page {page_num}/{total_pages}")
                
                # 1. Extract Text
                text_content = page.get_text()
                
                # 2. Process Images
                image_descriptions = []
                image_list = page.get_images(full=True)
                
                if image_list:
                    logger.debug(f"Found {len(image_list)} images on page {page_num}")
                    
                    for img_index, img in enumerate(image_list, start=1):
                        logger.debug(f"Processing image {img_index}/{len(image_list)} on page {page_num}")
                        description = process_image(doc, img)
                        if description:
                            image_descriptions.append(description)
                
                # 3. Combine Text + Image Context
                full_page_content = text_content + "".join(image_descriptions)
                
                # 4. Create Document
                metadata = {
                    "source": file_path.name,
                    "page": page_num,
                    "total_pages": total_pages,
                    "has_images": len(image_descriptions) > 0,
                }
                
                documents.append(Document(
                    page_content=full_page_content,
                    metadata=metadata,
                ))
                
                logger.debug(f"Completed processing page {page_num}/{total_pages}")

        logger.info(f"Successfully processed {len(documents)} pages from {file_path}")
        return documents

    except Exception as e:
        error_msg = f"Error processing PDF {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise DocumentProcessingError(error_msg) from e


def create_faiss_index(documents: List[Document], output_dir: str = "faiss_index") -> None:
    """Create and save a FAISS vector store from document chunks.
    
    Args:
        documents: List of document chunks to index
        output_dir: Directory to save the FAISS index
    """
    try:
        logger.info("Creating FAISS index...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        
        # Create FAISS index from documents
        db = FAISS.from_documents(documents, embeddings)
        
        # Save the index
        db.save_local(output_dir)
        
        logger.info(f"Saved FAISS index to {output_dir}")
        
    except Exception as e:
        error_msg = f"Error creating FAISS index: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise DocumentProcessingError(error_msg) from e


def main() -> int:
    """Main entry point for the PDF ingestion script.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    import argparse
    import pickle
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Process a PDF file, extract text and images, and create a searchable index."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for FAISS index and chunks",
        default="faiss_index",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save processed chunks to JSON file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logger.setLevel("DEBUG")
    
    try:
        file_path = Path(args.file_path).resolve()
        logger.info(f"Starting ingestion of {file_path}")
        
        # Process PDF and extract content
        documents = ingest_pdf_with_vision(file_path)
        
        # Chunk the documents using our context-aware chunker
        logger.info(f"Chunking {len(documents)} pages into semantic chunks...")
        chunker = GuidelineChunker()
        chunks = chunker.split_documents(documents)
        
        # Update documents to use the chunked content
        documents = chunks
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks as pickle for BM25
        chunks_path = output_dir / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Saved {len(documents)} chunks to {chunks_path}")
        
        # Create and save FAISS index
        create_faiss_index(documents, str(output_dir))
        
        # Optionally save as JSON if requested
        if args.save_json:
            json_path = output_dir / "chunks.json"
            serializable_docs = []
            for doc in documents:
                serializable_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                })
            
            import json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(documents)} chunks to {json_path}")
        
        logger.info(f"Successfully processed {len(documents)} chunks from {file_path}")
        logger.info(f"FAISS index and chunks saved to {output_dir}")
        return 0
        
    except Exception as e:
        logger.critical(f"Fatal error during PDF processing: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())