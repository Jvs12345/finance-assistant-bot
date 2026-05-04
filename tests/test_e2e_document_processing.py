"""
End-to-end integration test for document processing workflow.
Tests the complete pipeline with a real PDF document.
"""

import pytest
from pathlib import Path
import uuid

from src.models.document import DocumentCategory, ProcessingStatus
from src.services.document_processor import get_document_processor
from src.db.elasticsearch import get_elasticsearch_client
from src.db.postgres import get_postgres_client


@pytest.mark.e2e
@pytest.mark.integration
class TestE2EDocumentProcessing:
    """End-to-end test for complete document processing workflow."""

    @pytest.fixture
    def pdf_file(self):
        """Get path to test PDF file."""
        pdf_path = Path("/workspaces/doc-parser/urs-1-20.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        return pdf_path

    @pytest.fixture
    def processor(self):
        """Get document processor instance."""
        return get_document_processor()

    @pytest.fixture
    def es_client(self):
        """Get Elasticsearch client."""
        return get_elasticsearch_client()

    @pytest.fixture
    def db_client(self):
        """Get PostgreSQL client."""
        return get_postgres_client()

    def test_full_document_processing_workflow(
        self,
        pdf_file,
        processor,
        es_client,
        db_client
    ):
        """
        Test complete workflow: PDF upload → parse → chunk → summarize → index.

        This test verifies:
        1. PDF parsing with LandingAI
        2. Markdown chunking by page
        3. Summary generation with Claude (optional - may skip if no API key)
        4. Indexing in Elasticsearch
        5. Metadata storage in PostgreSQL
        """
        # Generate unique document ID
        document_id = f"test-{uuid.uuid4().hex[:8]}"

        print(f"\n🧪 Testing document processing for {pdf_file.name}")
        print(f"📄 Document ID: {document_id}")

        try:
            # Create document record in database
            db_client.create_document(
                document_id=document_id,
                filename=pdf_file.name,
                original_filename=pdf_file.name,
                file_size=pdf_file.stat().st_size,
                file_path=str(pdf_file),
                category=DocumentCategory.MAINTENANCE
            )

            # Process document through pipeline
            # Skip summaries to avoid API costs in testing
            print("⚙️  Starting document processing pipeline...")
            result = processor.process_document(
                file_path=pdf_file,
                document_id=document_id,
                original_filename=pdf_file.name,
                category=DocumentCategory.MAINTENANCE,
                generate_summaries=False  # Skip to avoid API costs
            )

            # Verify processing results
            print(f"📊 Processing Results:")
            print(f"   - Status: {result['status']}")
            print(f"   - Total pages: {result['total_pages']}")
            print(f"   - Pages indexed: {result['pages_indexed']}")
            print(f"   - Summaries generated: {result['summaries_generated']}")

            assert result['status'] == ProcessingStatus.READY
            assert result['total_pages'] > 0, "Should have extracted pages"
            assert result['pages_indexed'] == result['total_pages'], "All pages should be indexed"

            # Update database record
            db_client.update_document_status(
                document_id=document_id,
                status=ProcessingStatus.READY,
                total_pages=result['total_pages']
            )

            # Verify pages in Elasticsearch
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
                size=1000
            )

            indexed_pages = search_result["hits"]["total"]["value"]
            print(f"✅ Verified {indexed_pages} pages in Elasticsearch")

            assert indexed_pages == result['total_pages'], "ES should have all pages"

            # Verify document structure
            first_page = search_result["hits"]["hits"][0]["_source"]
            print(f"\n📄 Sample Page Structure:")
            print(f"   - Document ID: {first_page['document_id']}")
            print(f"   - Filename: {first_page['filename']}")
            print(f"   - Page: {first_page['page']}")
            print(f"   - Category: {first_page['category']}")
            print(f"   - Content length: {len(first_page['content'])} chars")
            print(f"   - Has part numbers: {len(first_page.get('part_numbers', []))} items")

            assert first_page['document_id'] == document_id
            assert first_page['filename'] == pdf_file.name
            assert first_page['category'] == 'maintenance'
            assert len(first_page['content']) > 0
            assert 'page' in first_page
            assert 'upload_date' in first_page

            # Verify database record
            db_doc = db_client.get_document(document_id)
            assert db_doc is not None
            assert db_doc.processing_status == ProcessingStatus.READY
            assert db_doc.total_pages == result['total_pages']

            print(f"\n✅ Full workflow test PASSED!")
            print(f"   - {result['total_pages']} pages processed successfully")
            print(f"   - All data verified in Elasticsearch and PostgreSQL")

        finally:
            # Cleanup: Delete test document
            print(f"\n🧹 Cleaning up test data...")
            try:
                deleted = processor.delete_document(document_id)
                print(f"   - Deleted {deleted} pages from Elasticsearch")

                db_client.delete_document(document_id)
                print(f"   - Deleted database record")

            except Exception as e:
                print(f"⚠️  Cleanup error (non-fatal): {e}")

    def test_search_processed_document(
        self,
        pdf_file,
        processor,
        es_client
    ):
        """
        Test searching for content in a processed document.

        This verifies the indexed content is searchable.
        """
        document_id = f"test-search-{uuid.uuid4().hex[:8]}"

        try:
            print(f"\n🔍 Testing search functionality")

            # Process document (without summaries)
            result = processor.process_document(
                file_path=pdf_file,
                document_id=document_id,
                original_filename=pdf_file.name,
                category=DocumentCategory.MAINTENANCE,
                generate_summaries=False
            )

            # Refresh index to make documents searchable
            es_client.client.indices.refresh(index="documents")

            # Search for common technical terms
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "document_id": document_id
                                }
                            }
                        ]
                    }
                }
            }

            search_result = es_client.search(
                index_name="documents",
                query=search_query,
                size=10
            )

            hits = search_result["hits"]["total"]["value"]
            print(f"   - Found {hits} pages for document")

            assert hits > 0, "Should find indexed pages"

            # Try fuzzy search
            fuzzy_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "document_id": document_id
                                }
                            },
                            {
                                "match": {
                                    "content": {
                                        "query": "specification",
                                        "fuzziness": "AUTO"
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            fuzzy_result = es_client.search(
                index_name="documents",
                query=fuzzy_query,
                size=5
            )

            fuzzy_hits = fuzzy_result["hits"]["total"]["value"]
            print(f"   - Fuzzy search for 'specification': {fuzzy_hits} results")

            print(f"✅ Search test PASSED!")

        finally:
            # Cleanup
            try:
                processor.delete_document(document_id)
            except:
                pass
