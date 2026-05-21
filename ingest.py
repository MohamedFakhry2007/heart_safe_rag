#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from heartsafe_rag.ingestion import process_batch, process_single_pdf
from heartsafe_rag.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="HeartSafe RAG Ingestion CLI")
    parser.add_argument("path", type=str, help="Path to PDF file or directory of PDFs")
    parser.add_argument("--output", "-o", type=str, default="faiss_index", help="Output directory")

    args = parser.parse_args()

    path = Path(args.path)
    output_dir = Path(args.output)

    try:
        if path.is_dir():
            process_batch(path, output_dir)
        else:
            process_single_pdf(path, output_dir)
        return 0
    except Exception as e:
        logger.critical(f"Ingestion Failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
