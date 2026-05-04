# Docker image for the Financial Document Assistant
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y \
    # Tesseract OCR engine
    tesseract-ocr \
    tesseract-ocr-eng \
    # Additional language packs (add as needed)
    # tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-spa \
    # Image processing libraries
    libtesseract-dev \
    libleptonica-dev \
    # Build tools
    gcc \
    g++ \
    # Utilities
    curl \
    git \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Verify Tesseract installation
RUN tesseract --version

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p Source_files logs

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the API server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
