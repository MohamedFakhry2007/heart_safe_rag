#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from heartsafe_rag.ingestion import run_ingestion_pipeline
from heartsafe_rag.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="HeartSafe RAG Ingestion CLI")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF guideline file")
    parser.add_argument("--output", "-o", type=str, default="faiss_index", help="Output directory")

    args = parser.parse_args()

    try:
        pdf_path = Path(args.pdf_path)
        output_dir = Path(args.output)

        run_ingestion_pipeline(pdf_path, output_dir)
        return 0
    except Exception as e:
        logger.critical(f"Ingestion Failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
