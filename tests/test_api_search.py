"""
Tests for search API endpoints.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from src.main import app
from src.models.search import SearchResponse, SearchResult


@pytest.mark.unit
class TestSearchAPI:
    """Unit tests for search API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_search_response(self):
        """Create mock search response."""
        return SearchResponse(
            query="test query",
            total=2,
            page=1,
            page_size=10,
            took=42,
            results=[
                SearchResult(
                    document_id="doc1",
                    filename="test.pdf",
                    page=1,
                    category="maintenance",
                    score=5.5,
                    snippet="This is a <mark>test</mark> snippet",
                    summary="Test summary",
                    machine_model="Model-X",
                    part_numbers=["123"],
                    upload_date=datetime(2024, 1, 1)
                ),
                SearchResult(
                    document_id="doc2",
                    filename="test2.pdf",
                    page=3,
                    category="operations",
                    score=3.2
                )
            ]
        )

    def test_search_basic_query(self, client, mock_search_response):
        """Test basic search with query only."""
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.return_value = mock_search_response

            response = client.post(
                "/api/v1/search",
                json={"query": "test query"}
            )

            assert response.status_code == 200
            data = response.json()

            assert data["query"] == "test query"
            assert data["total"] == 2
            assert data["took"] == 42
            assert len(data["results"]) == 2

            # Check first result
            assert data["results"][0]["document_id"] == "doc1"
            assert data["results"][0]["score"] == 5.5
            assert data["results"][0]["snippet"] == "This is a <mark>test</mark> snippet"

    def test_search_with_pagination(self, client, mock_search_response):
        """Test search with pagination parameters."""
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.return_value = mock_search_response

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test",
                    "page": 2,
                    "page_size": 20
                }
            )

            assert response.status_code == 200

            # Verify service was called with correct parameters
            call_args = mock_service.return_value.search.call_args[0][0]
            assert call_args.page == 2
            assert call_args.page_size == 20

    def test_search_with_filters(self, client, mock_search_response):
        """Test search with all filter types."""
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.return_value = mock_search_response

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test",
                    "filters": {
                        "category": "maintenance",
                        "machine_model": "Model-X",
                        "date_from": "2024-01-01T00:00:00",
                        "date_to": "2024-12-31T23:59:59",
                        "part_numbers": ["123", "456"]
                    }
                }
            )

            assert response.status_code == 200

            # Verify filters were passed correctly
            call_args = mock_service.return_value.search.call_args[0][0]
            assert call_args.filters.category.value == "maintenance"
            assert call_args.filters.machine_model == "Model-X"
            assert call_args.filters.part_numbers == ["123", "456"]

    def test_search_without_fuzzy(self, client, mock_search_response):
        """Test search with fuzzy matching disabled."""
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.return_value = mock_search_response

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test",
                    "enable_fuzzy": False
                }
            )

            assert response.status_code == 200

            call_args = mock_service.return_value.search.call_args[0][0]
            assert call_args.enable_fuzzy is False

    def test_search_without_highlights(self, client, mock_search_response):
        """Test search with highlights disabled."""
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.return_value = mock_search_response

            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test",
                    "include_highlights": False
                }
            )

            assert response.status_code == 200

            call_args = mock_service.return_value.search.call_args[0][0]
            assert call_args.include_highlights is False

    def test_search_empty_query(self, client):
        """Test search with empty query returns 422."""
        response = client.post(
            "/api/v1/search",
            json={"query": ""}
        )

        assert response.status_code == 422

    def test_search_whitespace_query(self, client):
        """Test search with whitespace-only query returns 422."""
        response = client.post(
            "/api/v1/search",
            json={"query": "   "}
        )

        assert response.status_code == 422

    def test_search_missing_query(self, client):
        """Test search without query parameter returns 422."""
        response = client.post(
            "/api/v1/search",
            json={}
        )

        assert response.status_code == 422

    def test_search_query_too_long(self, client):
        """Test search with query exceeding max length returns 422."""
        long_query = "a" * 501  # Max is 500

        response = client.post(
            "/api/v1/search",
            json={"query": long_query}
        )

        assert response.status_code == 422

    def test_search_invalid_page(self, client):
        """Test search with invalid page number returns 422."""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "page": 0  # Must be >= 1
            }
        )

        assert response.status_code == 422

    def test_search_invalid_page_size(self, client):
        """Test search with invalid page size returns 422."""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "page_size": 101  # Max is 100
            }
        )

        assert response.status_code == 422

    def test_search_invalid_date_range(self, client):
        """Test search with invalid date range returns 422."""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "filters": {
                    "date_from": "2024-12-31T00:00:00",
                    "date_to": "2024-01-01T00:00:00"  # Before date_from
                }
            }
        )

        assert response.status_code == 422

    def test_search_service_error(self, client):
        """Test search returns 500 when service fails."""
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.side_effect = Exception("ES connection failed")

            response = client.post(
                "/api/v1/search",
                json={"query": "test"}
            )

            assert response.status_code == 500
            assert "Search failed" in response.json()["detail"]

    def test_search_value_error(self, client):
        """Test search returns 400 for value errors."""
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.side_effect = ValueError("Invalid query")

            response = client.post(
                "/api/v1/search",
                json={"query": "test"}
            )

            assert response.status_code == 400
            assert "Invalid query" in response.json()["detail"]

    def test_search_response_pagination_metadata(self, client, mock_search_response):
        """Test search response includes pagination metadata."""
        # Adjust mock to have more pages
        mock_search_response.total = 100
        mock_search_response.page = 2
        mock_search_response.page_size = 10

        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.search.return_value = mock_search_response

            response = client.post(
                "/api/v1/search",
                json={"query": "test", "page": 2}
            )

            assert response.status_code == 200
            data = response.json()

            # Note: total_pages, has_next, has_previous are computed properties
            # They may not be in JSON response unless explicitly included
            assert data["total"] == 100
            assert data["page"] == 2
            assert data["page_size"] == 10


@pytest.mark.integration
class TestSearchAPIIntegration:
    """Integration tests for search API with real backend."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_search_integration(self, client):
        """Test search endpoint with real Elasticsearch."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test query that probably does not exist"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert "total" in data
        assert "results" in data
        assert isinstance(data["results"], list)
