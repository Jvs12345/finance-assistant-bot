"""
Tests for Elasticsearch client and index management.
"""

import pytest
from datetime import datetime

from src.db.elasticsearch import get_elasticsearch_client, ElasticsearchClient
from src.db.index_schemas import create_documents_index


@pytest.fixture
def es_client():
    """Get Elasticsearch client for testing."""
    client = get_elasticsearch_client()
    yield client
    # Cleanup after tests
    try:
        client.delete_index("test_documents")
    except:
        pass


class TestElasticsearchConnection:
    """Test Elasticsearch connection and health checks."""

    @pytest.mark.integration
    def test_client_initialization(self, es_client):
        """Test that Elasticsearch client initializes successfully."""
        assert es_client is not None
        assert isinstance(es_client, ElasticsearchClient)

    @pytest.mark.integration
    def test_ping(self, es_client):
        """Test that Elasticsearch is reachable."""
        result = es_client.ping()
        assert result is True

    @pytest.mark.integration
    def test_health_check(self, es_client):
        """Test Elasticsearch cluster health check."""
        health = es_client.health_check()
        assert health is not None
        assert "status" in health
        assert health["status"] in ["green", "yellow", "red"]
        assert "number_of_nodes" in health
        assert health["number_of_nodes"] >= 1


class TestIndexManagement:
    """Test Elasticsearch index creation and deletion."""

    @pytest.mark.integration
    def test_create_index(self, es_client):
        """Test creating a new index."""
        index_name = "test_documents"

        # Clean up if exists
        if es_client.client.indices.exists(index=index_name):
            es_client.delete_index(index_name)

        # Create index
        mappings = {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"}
            }
        }
        settings = {"number_of_shards": 1, "number_of_replicas": 0}

        created = es_client.create_index(
            index_name=index_name,
            mappings=mappings,
            settings=settings
        )

        assert created is True
        assert es_client.client.indices.exists(index=index_name)

        # Try creating again (should return False)
        created_again = es_client.create_index(
            index_name=index_name,
            mappings=mappings,
            settings=settings
        )
        assert created_again is False

        # Cleanup
        es_client.delete_index(index_name)

    @pytest.mark.integration
    def test_delete_index(self, es_client):
        """Test deleting an index."""
        index_name = "test_documents"

        # Create index first
        mappings = {"properties": {"title": {"type": "text"}}}
        es_client.create_index(index_name=index_name, mappings=mappings)

        # Delete index
        deleted = es_client.delete_index(index_name)
        assert deleted is True
        assert not es_client.client.indices.exists(index=index_name)

        # Try deleting again (should return False)
        deleted_again = es_client.delete_index(index_name)
        assert deleted_again is False

    @pytest.mark.integration
    def test_documents_index_creation(self, es_client):
        """Test creating the documents index with proper schema."""
        # Delete if exists
        try:
            es_client.delete_index("documents")
        except:
            pass

        # Create documents index
        created = create_documents_index(es_client)
        assert created is True

        # Verify index exists
        assert es_client.client.indices.exists(index="documents")

        # Verify mappings
        mapping = es_client.client.indices.get_mapping(index="documents")
        properties = mapping["documents"]["mappings"]["properties"]

        # Check key fields exist
        assert "document_id" in properties
        assert "filename" in properties
        assert "content" in properties
        assert "summary" in properties
        assert "page" in properties
        assert "category" in properties
        assert "upload_date" in properties

        # Check field types
        assert properties["document_id"]["type"] == "keyword"
        assert properties["content"]["type"] == "text"
        assert properties["page"]["type"] == "integer"


class TestDocumentOperations:
    """Test document indexing and retrieval."""

    @pytest.mark.integration
    def test_index_document(self, es_client):
        """Test indexing a single document."""
        index_name = "test_documents"

        # Create index
        mappings = {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "timestamp": {"type": "date"}
            }
        }
        es_client.create_index(index_name=index_name, mappings=mappings)

        # Index a document
        doc = {
            "title": "Test Document",
            "content": "This is a test document",
            "timestamp": datetime.utcnow().isoformat()
        }

        doc_id = es_client.index_document(
            index_name=index_name,
            document=doc,
            doc_id="test_123"
        )

        assert doc_id == "test_123"

        # Refresh index to make document searchable
        es_client.client.indices.refresh(index=index_name)

        # Verify document was indexed
        result = es_client.client.get(index=index_name, id=doc_id)
        assert result["_id"] == doc_id
        assert result["_source"]["title"] == "Test Document"

        # Cleanup
        es_client.delete_index(index_name)

    @pytest.mark.integration
    def test_bulk_index(self, es_client):
        """Test bulk indexing multiple documents."""
        index_name = "test_documents"

        # Create index
        mappings = {"properties": {"content": {"type": "text"}}}
        es_client.create_index(index_name=index_name, mappings=mappings)

        # Bulk index documents
        docs = [
            {"content": f"Document {i}"}
            for i in range(5)
        ]

        success, errors = es_client.bulk_index(
            index_name=index_name,
            documents=docs
        )

        assert success == 5
        assert len(errors) == 0

        # Refresh and verify
        es_client.client.indices.refresh(index=index_name)
        count = es_client.client.count(index=index_name)
        assert count["count"] == 5

        # Cleanup
        es_client.delete_index(index_name)

    @pytest.mark.integration
    def test_search_documents(self, es_client):
        """Test searching documents."""
        index_name = "test_documents"

        # Create index and add documents
        mappings = {"properties": {"content": {"type": "text"}}}
        es_client.create_index(index_name=index_name, mappings=mappings)

        docs = [
            {"content": "Python programming language"},
            {"content": "JavaScript programming language"},
            {"content": "Document about databases"}
        ]
        es_client.bulk_index(index_name=index_name, documents=docs)
        es_client.client.indices.refresh(index=index_name)

        # Search for "programming"
        query = {
            "query": {
                "match": {
                    "content": "programming"
                }
            }
        }

        results = es_client.search(index_name=index_name, query=query)

        assert results["hits"]["total"]["value"] == 2
        assert len(results["hits"]["hits"]) == 2

        # Verify search results contain "programming"
        for hit in results["hits"]["hits"]:
            assert "programming" in hit["_source"]["content"].lower()

        # Cleanup
        es_client.delete_index(index_name)
