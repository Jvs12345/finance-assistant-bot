"""
Pytest configuration and shared fixtures for all tests.
"""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock
from typing import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.db.postgres import Base, get_postgres_client
from src.db.elasticsearch import get_elasticsearch_client
from src.config import settings


# ==================== Test Configuration ====================

@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    return {
        "pdf_storage_path": tempfile.mkdtemp(),
        "max_file_size_mb": 100,
        "log_level": "DEBUG",
    }


# ==================== FastAPI Test Client ====================

@pytest.fixture
def api_client() -> TestClient:
    """
    FastAPI test client for API endpoint testing.

    Returns:
        TestClient: FastAPI test client
    """
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict:
    """
    Authentication headers with valid API key.

    Returns:
        dict: Headers with Authorization bearer token
    """
    return {"Authorization": f"Bearer {settings.api_key}"}


# ==================== Database Fixtures ====================

@pytest.fixture(scope="session")
def test_db_engine():
    """
    Create a test database engine.

    Uses SQLite in-memory database for fast testing.

    Returns:
        Engine: SQLAlchemy engine
    """
    # Use SQLite for testing (faster than PostgreSQL)
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def test_db_session(test_db_engine):
    """
    Create a test database session.

    Each test gets a fresh session with automatic rollback.

    Args:
        test_db_engine: SQLAlchemy engine fixture

    Returns:
        Session: SQLAlchemy session
    """
    SessionLocal = sessionmaker(bind=test_db_engine)
    session = SessionLocal()

    yield session

    session.rollback()
    session.close()


@pytest.fixture
def mock_postgres_client():
    """
    Mock PostgreSQL client for unit testing.

    Returns:
        Mock: Mocked PostgreSQL client
    """
    client = Mock()

    # Mock common methods
    client.create_document = Mock(return_value=Mock(id="test-doc-id"))
    client.get_document = Mock(return_value=None)
    client.update_document_status = Mock(return_value=True)
    client.list_documents = Mock(return_value=[])
    client.count_documents = Mock(return_value=0)
    client.delete_document = Mock(return_value=True)

    return client


# ==================== Elasticsearch Fixtures ====================

@pytest.fixture
def mock_elasticsearch_client():
    """
    Mock Elasticsearch client for unit testing.

    Returns:
        Mock: Mocked Elasticsearch client
    """
    client = Mock()

    # Mock common methods
    client.ping = Mock(return_value=True)
    client.info = Mock(return_value={"version": {"number": "8.11.0"}})
    client.indices.create = Mock(return_value={"acknowledged": True})
    client.indices.delete = Mock(return_value={"acknowledged": True})
    client.indices.exists = Mock(return_value=True)
    client.index = Mock(return_value={"_id": "test-es-id", "result": "created"})
    client.bulk = Mock(return_value={"errors": False, "items": []})
    client.search = Mock(return_value={
        "took": 10,
        "hits": {
            "total": {"value": 0},
            "hits": []
        }
    })
    client.delete_by_query = Mock(return_value={"deleted": 0})

    return client


@pytest.fixture
def elasticsearch_test_client():
    """
    Real Elasticsearch client for integration testing.

    Only use when Elasticsearch is actually running.

    Returns:
        ElasticsearchClient: Real ES client (or None if unavailable)
    """
    try:
        es_client = get_elasticsearch_client()
        if es_client.ping():
            # Create test index
            test_index = f"test_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            yield es_client, test_index

            # Cleanup: delete test index
            try:
                es_client.indices.delete(index=test_index, ignore=[404])
            except:
                pass
        else:
            pytest.skip("Elasticsearch not available")
    except Exception as e:
        pytest.skip(f"Elasticsearch not available: {e}")


# ==================== Service Mocks ====================

@pytest.fixture
def mock_pdf_parser():
    """
    Mock PDF parser for testing.

    Returns:
        Mock: Mocked PDFParser
    """
    parser = Mock()
    parser.parse_pdf = Mock(return_value="# Test Document\n\nTest content")
    parser.parse_pdf_with_retry = Mock(return_value="# Test Document\n\nTest content")
    return parser


@pytest.fixture
def mock_summarizer():
    """
    Mock summarizer for testing.

    Returns:
        Mock: Mocked Summarizer
    """
    summarizer = Mock()
    summarizer.summarize_text = Mock(
        return_value="This is a test summary of the document."
    )
    summarizer.summarize_text_with_retry = Mock(
        return_value="This is a test summary of the document."
    )
    summarizer.batch_summarize = Mock(
        return_value=["Summary 1", "Summary 2"]
    )
    return summarizer


@pytest.fixture
def mock_document_processor():
    """
    Mock document processor for testing.

    Returns:
        Mock: Mocked DocumentProcessor
    """
    processor = Mock()
    processor.process_document = Mock(return_value={
        "document_id": "test-doc-id",
        "status": "ready",
        "total_pages": 5,
        "pages_indexed": 5,
        "summaries_generated": 5,
        "error_message": None
    })
    return processor


# ==================== LandingAI Mock ====================

@pytest.fixture
def mock_landingai_client():
    """
    Mock LandingAI client for testing.

    Simulates the LandingAI ADE SDK responses.

    Returns:
        Mock: Mocked LandingAI client
    """
    client = Mock()

    # Mock predict response
    mock_response = Mock()
    mock_response.text = "# Sample Document\n\nThis is parsed content from PDF."

    client.predict = Mock(return_value=mock_response)

    return client


# ==================== Anthropic Mock ====================

@pytest.fixture
def mock_anthropic_client():
    """
    Mock Anthropic Claude client for testing.

    Simulates Claude API responses for summarization.

    Returns:
        Mock: Mocked Anthropic client
    """
    client = Mock()

    # Mock message response
    mock_content = Mock()
    mock_content.text = "This is a concise summary of the technical document."

    mock_message = Mock()
    mock_message.content = [mock_content]
    mock_message.usage = Mock(input_tokens=1000, output_tokens=50)

    client.messages.create = Mock(return_value=mock_message)

    return client


# ==================== Sample Files ====================

@pytest.fixture
def sample_pdf_path() -> Generator[Path, None, None]:
    """
    Create a sample PDF file for testing.

    Yields:
        Path: Path to temporary PDF file
    """
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_markdown() -> str:
    """
    Sample markdown content for testing.

    Returns:
        str: Sample markdown text
    """
    return """# Technical Manual

## Page 1

### Machine Specifications

| Model | Power | Speed |
|-------|-------|-------|
| X-100 | 5kW   | 100/min |
| X-200 | 10kW  | 200/min |

Part Numbers: ABC-123, XYZ-789

---

## Page 2

### Maintenance Instructions

1. Check oil levels daily
2. Clean filters weekly
3. Inspect belts monthly

Contact: support@example.com
"""


@pytest.fixture
def sample_document_metadata() -> dict:
    """
    Sample document metadata for testing.

    Returns:
        dict: Document metadata
    """
    return {
        "document_id": "test-doc-123",
        "filename": "test_manual.pdf",
        "file_size": 102400,
        "file_path": "/data/pdfs/test-doc-123.pdf",
        "category": "maintenance",
        "machine_model": "X-100",
        "part_numbers": ["ABC-123", "XYZ-789"],
        "upload_date": datetime.utcnow(),
        "processing_status": "ready",
        "total_pages": 2,
        "indexed_at": datetime.utcnow(),
        "error_message": None
    }


# ==================== Pytest Markers ====================

def pytest_configure(config):
    """
    Configure custom pytest markers.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (requires external services)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (full workflow)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (>1 second)"
    )
    config.addinivalue_line(
        "markers", "performance: Performance/benchmark tests"
    )


# ==================== Cleanup Helpers ====================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """
    Automatically cleanup test files after each test.

    Runs after every test to ensure clean state.
    """
    yield

    # Cleanup any test files in /tmp
    test_pattern = "/tmp/test_*"
    import glob
    for file_path in glob.glob(test_pattern):
        try:
            Path(file_path).unlink()
        except:
            pass
