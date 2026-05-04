"""
Integration tests for large PDF processing with page limiting.

Tests the complete workflow with PDFs of various sizes to verify:
1. Large PDFs (>50 pages) are automatically truncated to 50 pages
2. Truncation notification appears in error_message
3. Processing completes successfully
4. Elasticsearch has correct number of indexed pages
5. Search works on processed content
6. Download returns original (not truncated) PDF
7. Medium/small PDFs process normally without truncation
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from src.main import app
from src.config import settings
from src.db.elasticsearch import get_elasticsearch_client
from src.db.postgres import get_postgres_client
from src.utils.pdf_utils import get_pdf_page_count


client = TestClient(app)


@pytest.mark.integration
@pytest.mark.slow
class TestLargePDFProcessing:
    """Test processing of large PDFs with automatic page limiting."""

    @pytest.fixture(scope="class")
    def test_files_dir(self):
        """Get path to test files directory."""
        return Path("/workspaces/doc-parser/test-files")

    @pytest.fixture(scope="class")
    def large_pdf_path(self, test_files_dir):
        """Get path to large PDF (164 pages)."""
        pdf_path = test_files_dir / "manual.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Large test PDF not found: {pdf_path}")
        return pdf_path

    @pytest.fixture(scope="class")
    def medium_pdf_path(self, test_files_dir):
        """Get path to medium PDF (20 pages)."""
        pdf_path = test_files_dir / "urs-1-20.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Medium test PDF not found: {pdf_path}")
        return pdf_path

    @pytest.fixture(scope="class")
    def small_pdf_path(self, test_files_dir):
        """Get path to small PDF (14 pages)."""
        pdf_path = test_files_dir / "MPAC-Group-Global-Supplier-Requirements.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Small test PDF not found: {pdf_path}")
        return pdf_path

    @pytest.fixture
    def es_client(self):
        """Get Elasticsearch client."""
        return get_elasticsearch_client()

    @pytest.fixture
    def db_client(self):
        """Get PostgreSQL client."""
        return get_postgres_client()

    @pytest.mark.skipif(
        not Path("/workspaces/doc-parser/test-files/manual.pdf").exists(),
        reason="Large test PDF not available"
    )
    def test_large_pdf_truncation_workflow(
        self,
        large_pdf_path,
        es_client,
        db_client
    ):
        """
        Test that 164-page PDF gets truncated to 50 pages.

        This is the critical test for large PDF handling:
        1. Upload 164-page PDF
        2. Wait for processing (mocked to avoid API costs)
        3. Verify exactly 50 pages indexed
        4. Verify truncation notice in error_message
        5. Verify search works
        6. Verify download returns original 164-page PDF
        """
        print(f"\n📄 Testing large PDF processing: {large_pdf_path.name}")

        # Verify file exists and page count
        original_page_count = get_pdf_page_count(large_pdf_path)
        print(f"   Original page count: {original_page_count}")
        assert original_page_count == 164, f"Expected 164 pages, got {original_page_count}"

        # Mock LandingAI and Claude to avoid API costs
        with patch("src.services.pdf_parser.PDFParser.parse_pdf") as mock_parse, \
             patch("src.services.summarizer.Summarizer.summarize_text") as mock_summarize:

            # Mock parsing - return markdown with proper LandingAI page markers
            def mock_parse_pages(file_path, model="dpt-2-latest"):
                """Mock PDF parsing to return 50 pages of content."""
                pages = []
                for i in range(50):  # Limited to 50 pages
                    page_content = f"""<tr><td>Page:</td><td>{i+1} of 50</td></tr>

# Page {i+1}

This is content for page {i+1} of the manual.

Some technical information here about motors and components.

## Section on page {i+1}

More detailed technical content for testing search functionality."""
                    pages.append(page_content)
                return "\\n\\n".join(pages)

            mock_parse.side_effect = mock_parse_pages
            mock_summarize.return_value = "Summary of technical information."

            # Upload the large PDF
            print("   Uploading PDF...")
            with open(large_pdf_path, "rb") as f:
                response = client.post(
                    "/api/v1/documents/upload",
                    files={"file": (large_pdf_path.name, f, "application/pdf")},
                    data={
                        "category": "maintenance",
                        "machine_model": "LARGE-TEST"
                    },
                    headers={"Authorization": f"Bearer {settings.api_key}"}
                )

            assert response.status_code == 202, f"Upload failed: {response.text}"
            upload_response = response.json()
            document_id = upload_response["document_id"]
            print(f"   Document ID: {document_id}")

            # Wait for processing to complete (background task)
            print("   Waiting for processing...")
            max_wait = 60  # 60 seconds max
            start_time = time.time()
            status = None

            while time.time() - start_time < max_wait:
                response = client.get(
                    f"/api/v1/documents/{document_id}",
                    headers={"Authorization": f"Bearer {settings.api_key}"}
                )

                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data["status"]

                    if status == "ready":
                        print(f"   ✅ Processing complete (status: {status})")
                        break
                    elif status == "failed":
                        print(f"   ❌ Processing failed: {status_data.get('error_message')}")
                        assert False, f"Processing failed: {status_data.get('error_message')}"

                time.sleep(2)

            assert status == "ready", f"Processing did not complete in {max_wait}s (status: {status})"

            # Get final document status
            response = client.get(
                f"/api/v1/documents/{document_id}",
                headers={"Authorization": f"Bearer {settings.api_key}"}
            )
            assert response.status_code == 200
            doc_status = response.json()

            # Verify truncation notice
            print(f"   Total pages processed: {doc_status['total_pages']}")
            print(f"   Error message: {doc_status['error_message']}")

            assert doc_status["total_pages"] == 50, f"Expected 50 pages, got {doc_status['total_pages']}"
            assert doc_status["error_message"] is not None, "Expected truncation notice"
            assert "164 pages" in doc_status["error_message"], "Should mention original page count"
            assert "50 pages" in doc_status["error_message"] or "first 50" in doc_status["error_message"]

            # Verify Elasticsearch has exactly 50 pages
            es_client.client.indices.refresh(index="documents")

            query = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                }
            }

            search_result = es_client.search(
                index_name="documents",
                query=query,
                size=100
            )

            indexed_pages = search_result["hits"]["total"]["value"]
            print(f"   Pages in Elasticsearch: {indexed_pages}")
            assert indexed_pages == 50, f"Expected 50 pages in ES, got {indexed_pages}"

            # Verify search works
            print("   Testing search...")
            search_response = client.post(
                "/api/v1/search",
                json={
                    "query": "technical information",
                    "page": 1,
                    "page_size": 10
                }
            )

            assert search_response.status_code == 200
            search_data = search_response.json()

            # Should find some results from our document
            matching_results = [
                r for r in search_data["results"]
                if r["document_id"] == document_id
            ]
            print(f"   Search results found: {len(matching_results)}")
            assert len(matching_results) > 0, "Search should find results in processed document"

            # Verify download returns original 164-page PDF
            print("   Testing download...")
            download_response = client.get(
                f"/api/v1/documents/{document_id}/download",
                headers={"Authorization": f"Bearer {settings.api_key}"}
            )

            assert download_response.status_code == 200
            assert download_response.headers["content-type"] == "application/pdf"

            # The downloaded file should still be the original 164-page PDF
            downloaded_size = len(download_response.content)
            original_size = large_pdf_path.stat().st_size
            print(f"   Downloaded size: {downloaded_size} bytes")
            print(f"   Original size: {original_size} bytes")
            assert downloaded_size == original_size, "Downloaded PDF should match original"

            # Cleanup
            print("   Cleaning up...")
            delete_response = client.delete(
                f"/api/v1/documents/{document_id}",
                headers={"Authorization": f"Bearer {settings.api_key}"}
            )
            assert delete_response.status_code == 204

            print("   ✅ Large PDF test PASSED!")

    @pytest.mark.skipif(
        not Path("/workspaces/doc-parser/test-files/urs-1-20.pdf").exists(),
        reason="Medium test PDF not available"
    )
    def test_medium_pdf_no_truncation(
        self,
        medium_pdf_path,
        es_client,
        db_client
    ):
        """
        Test that 20-page PDF processes without truncation.

        This verifies that PDFs under the 50-page limit:
        1. Process all pages
        2. Have no truncation notice
        3. All pages indexed in Elasticsearch
        """
        print(f"\n📄 Testing medium PDF processing: {medium_pdf_path.name}")

        # Verify file exists and page count
        original_page_count = get_pdf_page_count(medium_pdf_path)
        print(f"   Original page count: {original_page_count}")
        assert original_page_count == 20, f"Expected 20 pages, got {original_page_count}"

        # Mock LandingAI and Claude
        with patch("src.services.pdf_parser.PDFParser.parse_pdf") as mock_parse, \
             patch("src.services.summarizer.Summarizer.summarize_text") as mock_summarize:

            def mock_parse_pages(file_path, model="dpt-2-latest"):
                """Mock PDF parsing to return 20 pages of content."""
                pages = []
                for i in range(20):
                    page_content = f"""<tr><td>Page:</td><td>{i+1} of 20</td></tr>

# Page {i+1}

Content for page {i+1} with operational information."""
                    pages.append(page_content)
                return "\\n\\n".join(pages)

            mock_parse.side_effect = mock_parse_pages
            mock_summarize.return_value = "Page summary."

            # Upload the PDF
            print("   Uploading PDF...")
            with open(medium_pdf_path, "rb") as f:
                response = client.post(
                    "/api/v1/documents/upload",
                    files={"file": (medium_pdf_path.name, f, "application/pdf")},
                    data={
                        "category": "operations",
                        "machine_model": "MEDIUM-TEST"
                    },
                    headers={"Authorization": f"Bearer {settings.api_key}"}
                )

            assert response.status_code == 202
            document_id = response.json()["document_id"]
            print(f"   Document ID: {document_id}")

            # Wait for processing
            print("   Waiting for processing...")
            max_wait = 45
            start_time = time.time()
            status = None

            while time.time() - start_time < max_wait:
                response = client.get(
                    f"/api/v1/documents/{document_id}",
                    headers={"Authorization": f"Bearer {settings.api_key}"}
                )

                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data["status"]

                    if status in ["ready", "failed"]:
                        break

                time.sleep(2)

            assert status == "ready", f"Processing did not complete (status: {status})"

            # Get final status
            response = client.get(
                f"/api/v1/documents/{document_id}",
                headers={"Authorization": f"Bearer {settings.api_key}"}
            )
            doc_status = response.json()

            # Verify NO truncation
            print(f"   Total pages processed: {doc_status['total_pages']}")
            print(f"   Error message: {doc_status['error_message']}")

            assert doc_status["total_pages"] == 20, f"Expected 20 pages, got {doc_status['total_pages']}"

            # Error message should either be None or not contain truncation notice
            error_msg = doc_status["error_message"]
            if error_msg:
                assert "truncated" not in error_msg.lower(), "Should not mention truncation"
                assert "164 pages" not in error_msg, "Should not mention different page count"

            # Verify all 20 pages in Elasticsearch
            es_client.client.indices.refresh(index="documents")

            query = {"query": {"term": {"document_id": document_id}}}
            search_result = es_client.search(
                index_name="documents",
                query=query,
                size=50
            )

            indexed_pages = search_result["hits"]["total"]["value"]
            print(f"   Pages in Elasticsearch: {indexed_pages}")
            assert indexed_pages == 20, f"Expected 20 pages in ES, got {indexed_pages}"

            # Cleanup
            print("   Cleaning up...")
            client.delete(
                f"/api/v1/documents/{document_id}",
                headers={"Authorization": f"Bearer {settings.api_key}"}
            )

            print("   ✅ Medium PDF test PASSED!")

    @pytest.mark.skipif(
        not Path("/workspaces/doc-parser/test-files/MPAC-Group-Global-Supplier-Requirements.pdf").exists(),
        reason="Small test PDF not available"
    )
    def test_small_pdf_processing(
        self,
        small_pdf_path,
        es_client
    ):
        """
        Test that small PDF (14 pages) processes normally.

        Quick validation that small PDFs work as expected.
        """
        print(f"\n📄 Testing small PDF processing: {small_pdf_path.name}")

        original_page_count = get_pdf_page_count(small_pdf_path)
        print(f"   Original page count: {original_page_count}")

        with patch("src.services.pdf_parser.PDFParser.parse_pdf") as mock_parse, \
             patch("src.services.summarizer.Summarizer.summarize_text") as mock_summarize:

            def mock_parse_pages(file_path, model="dpt-2-latest"):
                pages = []
                for i in range(original_page_count):
                    page_content = f"""<tr><td>Page:</td><td>{i+1} of {original_page_count}</td></tr>

# Page {i+1}

Requirements on page {i+1}."""
                    pages.append(page_content)
                return "\\n\\n".join(pages)

            mock_parse.side_effect = mock_parse_pages
            mock_summarize.return_value = "Requirements summary."

            # Upload and process
            with open(small_pdf_path, "rb") as f:
                response = client.post(
                    "/api/v1/documents/upload",
                    files={"file": (small_pdf_path.name, f, "application/pdf")},
                    data={"category": "spare_parts"},
                    headers={"Authorization": f"Bearer {settings.api_key}"}
                )

            assert response.status_code == 202
            document_id = response.json()["document_id"]

            # Wait for processing
            for _ in range(30):  # 60 seconds max
                time.sleep(2)
                response = client.get(
                    f"/api/v1/documents/{document_id}",
                    headers={"Authorization": f"Bearer {settings.api_key}"}
                )

                if response.status_code == 200:
                    if response.json()["status"] == "ready":
                        break

            doc_status = response.json()
            assert doc_status["status"] == "ready"
            assert doc_status["total_pages"] == original_page_count
            assert doc_status["error_message"] is None or "truncated" not in doc_status["error_message"].lower()

            # Cleanup
            client.delete(
                f"/api/v1/documents/{document_id}",
                headers={"Authorization": f"Bearer {settings.api_key}"}
            )

            print(f"   ✅ Small PDF test PASSED ({original_page_count} pages)")
