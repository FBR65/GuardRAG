# GuardRAG - Secure Document RAG System
FROM python:3.11-slim

LABEL maintainer="GuardRAG Team"
LABEL description="RAG-System auf Basis von COLPALI mit integrierten Guardrails"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    poppler-utils \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY mcp_services/ ./mcp_services/
COPY mcp_fileconverter/ ./mcp_fileconverter/
COPY main.py ./
COPY .env.example ./.env

# Create necessary directories
RUN mkdir -p uploads logs

# Install Python dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["/bin/bash", "-c", "source .venv/bin/activate && python main.py"]
