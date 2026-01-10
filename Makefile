.PHONY: install setup ingest run eval check

install:
	poetry install

setup: install
	@echo "Creating data directories..."
	mkdir -p data/guidelines faiss_index eval

format:
	poetry run ruff format .

check:
	poetry run ruff check .
	poetry run mypy .

ingest:
	@echo "Running Ingestion Pipeline..."
	poetry run python ingest.py

run:
	@echo "Starting Streamlit App..."
	poetry run streamlit run app.py

eval:
	@echo "Running Golden Dataset Evaluation..."
	poetry run python evaluation.py

test:
	poetry run pytest