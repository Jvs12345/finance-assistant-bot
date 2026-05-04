"""
Tests for search service.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.search_service import SearchService, get_search_service
from src.models.search import SearchRequest, SearchFilters
from src.models.document import DocumentCategory


@pytest.mark.unit
class TestSearchService:
    """Unit tests for SearchService."""

    @pytest.fixture
    def mock_es_client(self):
        """Create mock Elasticsearch client."""
        mock_client = Mock()
        mock_client.client = Mock()
        return mock_client

    @pytest.fixture
    def search_service(self, mock_es_client):
        """Create search service with mocked ES client."""
        with patch("src.services.search_service.get_elasticsearch_client", return_value=mock_es_client):
            service = SearchService()
            return service

    def test_build_basic_query(self, search_service):
        """Test building a basic search query without filters."""
        request = SearchRequest(query="test query")

        query = search_service._build_query(request)

        # Check query structure
        assert "query" in query
        assert "bool" in query["query"]
        assert "must" in query["query"]["bool"]

        # Check multi-match query
        multi_match = query["query"]["bool"]["must"][0]["multi_match"]
        assert multi_match["query"] == "test query"
        assert "part_numbers^3" in multi_match["fields"]
        assert "content^2" in multi_match["fields"]
        assert multi_match["fuzziness"] == "AUTO"

        # Check sorting
        assert query["sort"] == ["_score", {"upload_date": {"order": "desc"}}]

    def test_build_query_with_highlights(self, search_service):
        """Test query includes highlighting configuration."""
        request = SearchRequest(query="test", include_highlights=True, include_content=True)

        query = search_service._build_query(request)

        assert "highlight" in query
        assert "content" in query["highlight"]["fields"]
        assert "summary" in query["highlight"]["fields"]
        # When include_content=True, content uses full-field highlighting (no fragments)
        assert query["highlight"]["fields"]["content"]["number_of_fragments"] == 0
        assert query["highlight"]["fields"]["summary"]["fragment_size"] == 150

    def test_build_query_with_snippet_highlights(self, search_service):
        """Test query uses fragment highlighting when content not included."""
        request = SearchRequest(query="test", include_highlights=True, include_content=False)

        query = search_service._build_query(request)

        assert "highlight" in query
        assert "content" in query["highlight"]["fields"]
        # When include_content=False, content uses fragment highlighting for snippets
        assert query["highlight"]["fields"]["content"]["fragment_size"] == 150
        assert query["highlight"]["fields"]["content"]["number_of_fragments"] == 1

    def test_build_query_without_fuzzy(self, search_service):
        """Test query without fuzzy matching."""
        request = SearchRequest(query="test", enable_fuzzy=False)

        query = search_service._build_query(request)

        multi_match = query["query"]["bool"]["must"][0]["multi_match"]
        assert "fuzziness" not in multi_match

    def test_build_filters_category(self, search_service):
        """Test building category filter."""
        filters = SearchFilters(category=DocumentCategory.MAINTENANCE)

        filter_clauses = search_service._build_filters(filters)

        assert len(filter_clauses) == 1
        assert filter_clauses[0] == {"term": {"category": "maintenance"}}

    def test_build_filters_machine_model(self, search_service):
        """Test building machine model filter."""
        filters = SearchFilters(machine_model="Model-XYZ")

        filter_clauses = search_service._build_filters(filters)

        assert len(filter_clauses) == 1
        assert filter_clauses[0] == {"term": {"machine_model": "Model-XYZ"}}

    def test_build_filters_date_range(self, search_service):
        """Test building date range filter."""
        date_from = datetime(2024, 1, 1)
        date_to = datetime(2024, 12, 31)
        filters = SearchFilters(date_from=date_from, date_to=date_to)

        filter_clauses = search_service._build_filters(filters)

        assert len(filter_clauses) == 1
        date_filter = filter_clauses[0]["range"]["upload_date"]
        assert date_filter["gte"] == date_from.isoformat()
        assert date_filter["lte"] == date_to.isoformat()

    def test_build_filters_part_numbers(self, search_service):
        """Test building part numbers filter."""
        filters = SearchFilters(part_numbers=["12345", "67890"])

        filter_clauses = search_service._build_filters(filters)

        assert len(filter_clauses) == 1
        assert filter_clauses[0] == {"terms": {"part_numbers": ["12345", "67890"]}}

    def test_build_filters_multiple(self, search_service):
        """Test building multiple filters together."""
        filters = SearchFilters(
            category=DocumentCategory.OPERATIONS,
            machine_model="Model-ABC",
            part_numbers=["12345"]
        )

        filter_clauses = search_service._build_filters(filters)

        assert len(filter_clauses) == 3

    def test_parse_response_basic(self, search_service):
        """Test parsing basic Elasticsearch response."""
        es_response = {
            "took": 42,
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_score": 5.5,
                        "_source": {
                            "document_id": "doc1",
                            "filename": "test.pdf",
                            "page": 1,
                            "category": "maintenance",
                            "summary": "Test summary",
                            "machine_model": "Model-X",
                            "part_numbers": ["123"],
                            "upload_date": "2024-01-01T00:00:00"
                        }
                    },
                    {
                        "_score": 3.2,
                        "_source": {
                            "document_id": "doc2",
                            "filename": "test2.pdf",
                            "page": 5,
                            "category": "operations",
                            "upload_date": "2024-01-02T00:00:00"
                        }
                    }
                ]
            }
        }

        response = search_service._parse_response(
            es_response=es_response,
            query="test query",
            page=1,
            page_size=10,
            include_highlights=False
        )

        assert response.query == "test query"
        assert response.total == 2
        assert response.took == 42
        assert len(response.results) == 2

        # Check first result
        assert response.results[0].document_id == "doc1"
        assert response.results[0].score == 5.5
        assert response.results[0].summary == "Test summary"

    def test_parse_response_with_highlights(self, search_service):
        """Test parsing response with highlighted snippets."""
        es_response = {
            "took": 10,
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_score": 4.0,
                        "_source": {
                            "document_id": "doc1",
                            "filename": "test.pdf",
                            "page": 1,
                            "category": "maintenance"
                        },
                        "highlight": {
                            "content": ["This is a <mark>highlighted</mark> snippet"]
                        }
                    }
                ]
            }
        }

        response = search_service._parse_response(
            es_response=es_response,
            query="test",
            page=1,
            page_size=10,
            include_highlights=True
        )

        assert response.results[0].snippet == "This is a <mark>highlighted</mark> snippet"

    def test_parse_response_prefers_summary_highlight(self, search_service):
        """Test that summary highlights are preferred over content."""
        es_response = {
            "took": 10,
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_score": 4.0,
                        "_source": {
                            "document_id": "doc1",
                            "filename": "test.pdf",
                            "page": 1,
                            "category": "maintenance"
                        },
                        "highlight": {
                            "summary": ["Summary <mark>highlight</mark>"],
                            "content": ["Content highlight"]
                        }
                    }
                ]
            }
        }

        response = search_service._parse_response(
            es_response=es_response,
            query="test",
            page=1,
            page_size=10,
            include_highlights=True
        )

        assert response.results[0].snippet == "Summary <mark>highlight</mark>"

    def test_search_pagination(self, search_service, mock_es_client):
        """Test search with pagination."""
        # Mock Elasticsearch response
        mock_es_client.client.search.return_value = {
            "took": 5,
            "hits": {
                "total": {"value": 50},
                "hits": []
            }
        }

        request = SearchRequest(query="test", page=3, page_size=10)
        response = search_service.search(request)

        # Check pagination calculation
        call_args = mock_es_client.client.search.call_args
        assert call_args.kwargs["from_"] == 20  # (page 3 - 1) * 10
        assert call_args.kwargs["size"] == 10

        # Check response pagination
        assert response.total_pages == 5
        assert response.has_next is True
        assert response.has_previous is True

    def test_get_search_service_singleton(self):
        """Test that get_search_service returns singleton instance."""
        service1 = get_search_service()
        service2 = get_search_service()

        assert service1 is service2


@pytest.mark.integration
class TestSearchServiceIntegration:
    """Integration tests with real Elasticsearch."""

    @pytest.fixture
    def search_service(self):
        """Create search service with real ES connection."""
        return SearchService()

    def test_search_empty_index(self, search_service):
        """Test search on empty/non-matching query."""
        request = SearchRequest(query="nonexistent_unique_query_12345")

        response = search_service.search(request)

        assert response.total >= 0
        assert response.took > 0
        assert isinstance(response.results, list)
