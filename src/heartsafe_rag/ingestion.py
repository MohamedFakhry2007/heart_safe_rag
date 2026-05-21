import pickle
from pathlib import Path

import fitz
import pdfplumber
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from heartsafe_rag.chunking import GuidelineChunker
from heartsafe_rag.config import settings
from heartsafe_rag.exceptions import DocumentProcessingError
from heartsafe_rag.utils.logger import logger


def _extract_tables_from_page(pdf_path: str, page_num: int) -> list[str]:
    tables_text: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    rows = []
                    for row in table:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    table_str = "\n".join(rows)
                    tables_text.append(f"\n[TABLE {table_idx + 1}]\n{table_str}\n[/TABLE]")
    except Exception as e:
        logger.warning(f"Table extraction failed on page {page_num + 1}: {e}")
    return tables_text


def process_pdf_page(pdf_path: str, page_num: int) -> str:
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        text_content = page.get_text() or ""
        doc.close()
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1} of {pdf_path}: {e}")
        return ""

    table_content = _extract_tables_from_page(pdf_path, page_num)

    if table_content:
        return text_content + "\n" + "\n".join(table_content)

    return text_content


def process_single_pdf(pdf_path: Path, output_dir: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Starting ingestion for: {pdf_path.name}")

    raw_docs: list[Document] = []
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            for i in range(total_pages):
                logger.info(f"Processing page {i+1}/{total_pages}...")
                content = process_pdf_page(str(pdf_path), i)
                meta = {"source": pdf_path.name, "page": i + 1}
                raw_docs.append(Document(page_content=content, metadata=meta))
    except Exception as e:
        raise DocumentProcessingError(f"Failed to read PDF: {e!s}") from e

    _build_and_save_indices(raw_docs, output_dir)


def process_batch(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")

    all_docs: list[Document] = []
    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as doc:
                for i in range(len(doc)):
                    content = process_pdf_page(str(pdf_path), i)
                    meta = {"source": pdf_path.name, "page": i + 1}
                    all_docs.append(Document(page_content=content, metadata=meta))
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            continue

    if not all_docs:
        logger.warning("No content extracted from any PDF.")
        return

    logger.info(f"Extracted {len(all_docs)} pages total.")
    _build_and_save_indices(all_docs, output_dir)


def _build_and_save_indices(docs: list[Document], output_dir: Path) -> None:
    logger.info("Chunking documents...")
    chunker = GuidelineChunker()
    chunks = chunker.split_documents(docs)

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.pkl"
    with chunks_path.open("wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")

    logger.info("Generating Embeddings & FAISS Index...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(output_dir))
    logger.info(f"FAISS index saved to {output_dir}")

    logger.info("Building BM25 index...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25_path = output_dir / "bm25_index.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index saved to {bm25_path}")

    logger.info("Ingestion complete.")
