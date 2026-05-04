"""
Compatibility module that re-exports the primary Elasticsearch client.

This project now uses a single Elasticsearch implementation:
`src.db.elasticsearch_client`.
"""

from src.db.elasticsearch_client import ElasticsearchClient, get_elasticsearch_client

__all__ = ["ElasticsearchClient", "get_elasticsearch_client"]
