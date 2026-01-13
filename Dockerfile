# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set work directory
WORKDIR /app

# Install system dependencies (curl for installing poetry, build-essential for compiling faiss/numpy if needed)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy only dependencies definition first (caching layer)
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "heartsafe_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]