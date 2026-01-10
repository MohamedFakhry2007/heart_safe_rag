# HeartSafe RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system for heart safety guidelines and information.

## Project Structure

- `data/`: Contains data files and guidelines
  - `guidelines/`: Storage for guideline documents
- `faiss_index/`: FAISS vector store for efficient similarity search
- `prompts/`: Contains prompt templates and configurations
- `eval/`: Evaluation scripts and test data
- `src/`: Source code for the RAG implementation

## Setup

1. Install dependencies:
   ```bash
   poetry install
   poetry run python ingest.py "data/guidelines/aha_guidelines_2022.pdf" --output-dir "faiss_index" --verbose
   ```

2. Add your guideline documents to `data/guidelines/`
3. Build the FAISS index (see documentation in `src/`)

## Usage

[Add usage instructions here]

## License

[Specify license]
