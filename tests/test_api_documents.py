"""
Tests for document management API endpoints.
"""

import io
import pytest
from pathlib import Path
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from src.main import app
from src.config import settings
from src.models.document import ProcessingStatus, DocumentCategory
from src.db.postgres import Document

client = TestClient(app)


@pytest.fixture
def mock_postgres_client():
    """Mock PostgreSQL client."""
    with patch("src.api.documents.get_postgres_client") as mock:
        pg_client = Mock()
        mock.return_value = pg_client
        yield pg_client


@pytest.fixture
def mock_document_processor():
    """Mock document processor."""
    with patch("src.api.documents.DocumentProcessor") as mock:
        processor = Mock()
        mock.return_value = processor
        yield processor


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing."""
    pdf_content = b"%PDF-1.4\n%Test PDF\n%%EOF"
    return io.BytesIO(pdf_content)


@pytest.fixture
def auth_headers():
    """Authentication headers."""
    return {"Authorization": f"Bearer {settings.api_key}"}


class TestUploadDocument:
    """Test document upload endpoint."""

    def test_upload_success(
        self, mock_postgres_client, sample_pdf_file, auth_headers
    ):
        """Test successful document upload."""
        # Mock database response
        mock_postgres_client.create_document.return_value = Mock()

        # Prepare upload
        files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
        data = {"category": "maintenance", "machine_model": "MODEL-123"}

        # Upload document
        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )

        # Assertions
        assert response.status_code == 202
        result = response.json()
        assert "document_id" in result
        assert result["filename"] == "test.pdf"
        assert result["status"] == "uploaded"
        assert "upload_date" in result

        # Verify database was called
        mock_postgres_client.create_document.assert_called_once()

    def test_upload_without_auth(self, sample_pdf_file):
        """Test upload without authentication."""
        files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
        data = {"category": "maintenance"}

        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
        )

        assert response.status_code == 403  # No credentials

    def test_upload_invalid_api_key(self, sample_pdf_file):
        """Test upload with invalid API key."""
        files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
        data = {"category": "maintenance"}
        headers = {"Authorization": "Bearer invalid_key"}

        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
            headers=headers,
        )

        assert response.status_code == 401

    def test_upload_non_pdf_file(self, auth_headers):
        """Test upload with non-PDF file."""
        files = {"file": ("test.txt", io.BytesIO(b"text"), "text/plain")}
        data = {"category": "maintenance"}

        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_invalid_category(self, sample_pdf_file, auth_headers):
        """Test upload with invalid category."""
        files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
        data = {"category": "invalid_category"}

        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "Invalid category" in response.json()["detail"]

    @patch("src.api.documents.save_uploaded_file")
    def test_upload_file_too_large(
        self, mock_save, mock_postgres_client, sample_pdf_file, auth_headers
    ):
        """Test upload with file exceeding size limit."""
        from fastapi import HTTPException

        # Mock save to raise size error
        mock_save.side_effect = HTTPException(status_code=413, detail="File too large")

        files = {"file": ("large.pdf", sample_pdf_file, "application/pdf")}
        data = {"category": "maintenance"}

        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )

        assert response.status_code == 413


class TestGetDocumentStatus:
    """Test get document status endpoint."""

    def test_get_status_success(self, mock_postgres_client, auth_headers):
        """Test getting document status successfully."""
        # Mock document
        mock_doc = Mock()
        mock_doc.id = "test-id"
        mock_doc.filename = "uuid-filename.pdf"
        mock_doc.original_filename = "test.pdf"
        mock_doc.processing_status = ProcessingStatus.READY
        mock_doc.upload_date = datetime.utcnow()
        mock_doc.indexed_at = datetime.utcnow()
        mock_doc.total_pages = 5
        mock_doc.error_message = None

        mock_postgres_client.get_document.return_value = mock_doc

        response = client.get(
            "/api/v1/documents/test-id",
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["document_id"] == "test-id"
        assert result["filename"] == "test.pdf"
        assert result["status"] == "ready"
        assert result["total_pages"] == 5

    def test_get_status_not_found(self, mock_postgres_client, auth_headers):
        """Test getting status for non-existent document."""
        mock_postgres_client.get_document.return_value = None

        response = client.get(
            "/api/v1/documents/nonexistent-id",
            headers=auth_headers,
        )

        assert response.status_code == 404

    def test_get_status_without_auth(self):
        """Test getting status without authentication."""
        response = client.get("/api/v1/documents/test-id")
        assert response.status_code == 403


class TestListDocuments:
    """Test list documents endpoint."""

    def test_list_success(self, mock_postgres_client, auth_headers):
        """Test listing documents successfully."""
        # Mock documents
        mock_doc = Mock()
        mock_doc.id = "doc-1"
        mock_doc.filename = "uuid-filename.pdf"
        mock_doc.original_filename = "test.pdf"
        mock_doc.file_size = 1024
        mock_doc.file_path = "/path/to/test.pdf"
        mock_doc.category = DocumentCategory.MAINTENANCE
        mock_doc.machine_model = "MODEL-123"
        mock_doc.processing_status = ProcessingStatus.READY
        mock_doc.upload_date = datetime.utcnow()
        mock_doc.indexed_at = datetime.utcnow()
        mock_doc.total_pages = 5
        mock_doc.error_message = None

        mock_postgres_client.list_documents.return_value = [mock_doc]
        mock_postgres_client.count_documents.return_value = 1

        response = client.get(
            "/api/v1/documents",
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["total"] == 1
        assert result["page"] == 1
        assert result["page_size"] == 10
        assert len(result["documents"]) == 1
        assert result["documents"][0]["document_id"] == "doc-1"

    def test_list_with_filters(self, mock_postgres_client, auth_headers):
        """Test listing documents with filters."""
        mock_postgres_client.list_documents.return_value = []
        mock_postgres_client.count_documents.return_value = 0

        response = client.get(
            "/api/v1/documents?status=ready&category=maintenance",
            headers=auth_headers,
        )

        assert response.status_code == 200

        # Verify filters were passed
        mock_postgres_client.list_documents.assert_called_once()
        call_kwargs = mock_postgres_client.list_documents.call_args[1]
        assert call_kwargs["status"] == ProcessingStatus.READY
        assert call_kwargs["category"] == DocumentCategory.MAINTENANCE

    def test_list_with_pagination(self, mock_postgres_client, auth_headers):
        """Test listing documents with pagination."""
        mock_postgres_client.list_documents.return_value = []
        mock_postgres_client.count_documents.return_value = 0

        response = client.get(
            "/api/v1/documents?page=2&page_size=20",
            headers=auth_headers,
        )

        assert response.status_code == 200

        # Verify pagination
        call_kwargs = mock_postgres_client.list_documents.call_args[1]
        assert call_kwargs["limit"] == 20
        assert call_kwargs["offset"] == 20  # (page 2 - 1) * 20

    def test_list_invalid_pagination(self, mock_postgres_client, auth_headers):
        """Test listing with invalid pagination parameters."""
        # Invalid page
        response = client.get(
            "/api/v1/documents?page=0",
            headers=auth_headers,
        )
        assert response.status_code == 400

        # Invalid page size
        response = client.get(
            "/api/v1/documents?page_size=200",
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_list_without_auth(self):
        """Test listing without authentication."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 403


class TestDeleteDocument:
    """Test delete document endpoint."""

    @patch("src.api.documents.get_elasticsearch_client")
    def test_delete_success(
        self, mock_es, mock_postgres_client, auth_headers, tmp_path
    ):
        """Test deleting document successfully."""
        # Create temp file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test")

        # Mock document
        mock_doc = Mock()
        mock_doc.id = "test-id"
        mock_doc.file_path = str(test_file)

        mock_postgres_client.get_document.return_value = mock_doc
        mock_postgres_client.delete_document.return_value = True

        # Mock ES client
        es_client = Mock()
        mock_es.return_value = es_client

        response = client.delete(
            "/api/v1/documents/test-id",
            headers=auth_headers,
        )

        assert response.status_code == 204

        # Verify deletions
        es_client.delete_by_query.assert_called_once()
        mock_postgres_client.delete_document.assert_called_once_with("test-id")
        assert not test_file.exists()

    def test_delete_not_found(self, mock_postgres_client, auth_headers):
        """Test deleting non-existent document."""
        mock_postgres_client.get_document.return_value = None

        response = client.delete(
            "/api/v1/documents/nonexistent",
            headers=auth_headers,
        )

        assert response.status_code == 404

    def test_delete_without_auth(self):
        """Test deleting without authentication."""
        response = client.delete("/api/v1/documents/test-id")
        assert response.status_code == 403


class TestDownloadDocument:
    """Test download document endpoint."""

    def test_download_success(self, mock_postgres_client, auth_headers, tmp_path):
        """Test downloading document successfully."""
        # Create temp PDF
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\ntest\n%%EOF")

        # Mock document
        mock_doc = Mock()
        mock_doc.id = "test-id"
        mock_doc.filename = "uuid-filename.pdf"
        mock_doc.original_filename = "test.pdf"
        mock_doc.file_path = str(test_file)

        mock_postgres_client.get_document.return_value = mock_doc

        response = client.get(
            "/api/v1/documents/test-id/download",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert b"PDF" in response.content

    def test_download_not_found(self, mock_postgres_client, auth_headers):
        """Test downloading non-existent document."""
        mock_postgres_client.get_document.return_value = None

        response = client.get(
            "/api/v1/documents/nonexistent/download",
            headers=auth_headers,
        )

        assert response.status_code == 404

    def test_download_file_missing(self, mock_postgres_client, auth_headers):
        """Test downloading when PDF file is missing."""
        # Mock document with non-existent file
        mock_doc = Mock()
        mock_doc.id = "test-id"
        mock_doc.file_path = "/nonexistent/path.pdf"

        mock_postgres_client.get_document.return_value = mock_doc

        response = client.get(
            "/api/v1/documents/test-id/download",
            headers=auth_headers,
        )

        assert response.status_code == 404
        assert "PDF file not found" in response.json()["detail"]

    def test_download_without_auth(self):
        """Test downloading without authentication."""
        response = client.get("/api/v1/documents/test-id/download")
        assert response.status_code == 403
