import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from heartsafe_rag.config import settings
from heartsafe_rag.exceptions import DocumentProcessingError
from heartsafe_rag.utils.logger import logger

SECTION_HEADER_PATTERN = re.compile(
    r"^(?P<num>(?:\d+\.?)+)\s+"
    r"(?P<title>[A-Z][A-Z\s\-/]+(?::[A-Za-z\s\-/,]+)?)",
    re.MULTILINE,
)

RECOMMENDATION_PATTERN = re.compile(
    r"(?:Class\s+(?:of\s+)?(?:Recommendation|I|IIa|IIb|III)\b)"
    r"|(?:COR\s+(?:I|IIa|IIb|III))"
    r"|(?:Level\s+of\s+Evidence\s*:?\s*[A-Z](?:-[A-Z])?)"
    r"|(?:LOE\s*:?\s*[A-Z](?:-[A-Z])?)",
    re.IGNORECASE,
)


def _detect_section_headers(text: str) -> list[tuple[int, str, int]]:
    """Detect section headers with their line positions and depth levels."""
    headers: list[tuple[int, str, int]] = []
    for match in SECTION_HEADER_PATTERN.finditer(text):
        num_str = match.group("num")
        title = match.group("title").strip()
        depth = len(num_str.split("."))
        headers.append((match.start(), title, depth))
    return headers


def _detect_recommendation_boundaries(text: str) -> list[int]:
    """Detect positions where recommendation blocks start."""
    return [m.start() for m in RECOMMENDATION_PATTERN.finditer(text)]


def _extract_section_context(position: int, headers: list[tuple[int, str, int]]) -> dict[str, Any]:
    """Determine what section a position falls within based on detected headers."""
    section_path: list[str] = []
    for header_pos, title, _ in headers:
        if header_pos <= position:
            section_path.append(title)
    return {
        "section_path": " > ".join(section_path) if section_path else "unknown",
        "section": section_path[-1] if section_path else "unknown",
    }


def extract_metadata_from_source(source_path: str) -> dict[str, Any]:
    filename = Path(source_path).name
    year_match = re.search(r"(199\d|20[0-2]\d)", filename)
    year = int(year_match.group(0)) if year_match else None
    return {
        "source": filename,
        "guideline_year": year if year else "Unknown",
        "processed_type": "guideline_text",
    }


def clean_content(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class GuidelineChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or settings.CHUNK_SEPARATORS

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            strip_whitespace=True,
            length_function=len,
        )
        logger.debug(f"Initialized Chunker: Size={self.chunk_size}, Overlap={self.chunk_overlap}")

    def _enrich_with_structure(self, doc: Document) -> list[Document]:
        """Split a document page into section-anchored chunks with enriched metadata."""
        text = doc.page_content
        headers = _detect_section_headers(text)
        rec_boundaries = _detect_recommendation_boundaries(text)

        recs = self.splitter.split_documents([doc])
        enriched: list[Document] = []

        for chunk in recs:
            start_pos = text.find(chunk.page_content[:50])
            if start_pos == -1:
                start_pos = 0

            section_ctx = _extract_section_context(start_pos, headers)
            has_rec = any(abs(start_pos - rb) < 200 for rb in rec_boundaries)

            chunk.metadata["section"] = section_ctx["section"]
            chunk.metadata["section_path"] = section_ctx["section_path"]
            chunk.metadata["has_recommendation"] = has_rec

            enriched.append(chunk)

        return enriched

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            logger.warning("No documents provided to chunker.")
            return []

        logger.info(f"Chunking {len(documents)} raw pages...")

        try:
            for doc in documents:
                source = doc.metadata.get("source", "unknown_file")
                meta_update = extract_metadata_from_source(source)
                doc.metadata.update(meta_update)
                doc.page_content = clean_content(doc.page_content)

            chunks: list[Document] = []
            for doc in documents:
                enriched_chunks = self._enrich_with_structure(doc)
                chunks.extend(enriched_chunks)

            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i

            logger.info(
                f"Chunking Complete. "
                f"Input Pages: {len(documents)} -> Output Chunks: {len(chunks)}"
            )

            return chunks  # noqa: TRY300

        except Exception as e:
            error_msg = f"Failed to chunk documents: {e!s}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
