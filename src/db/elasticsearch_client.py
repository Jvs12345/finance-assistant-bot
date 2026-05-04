"""Elasticsearch client."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError, ConnectionError as ESConnectionError

from src.utils.logging import get_logger
from src.config import settings
from src.services.embedding_service import EmbeddingService, EmbeddingProvider

logger = get_logger(__name__)


class ElasticsearchClient:
    """Handles indexing and retrieval in Elasticsearch."""

    def __init__(self):
        """Initialize Elasticsearch client."""
        self.es = Elasticsearch(
            [settings.elasticsearch_url],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        self.index_name = settings.elasticsearch_index
        
        self.embedding_service = EmbeddingService(
            provider=EmbeddingProvider.LOCAL,
            model="all-MiniLM-L6-v2"
        )

        self._ensure_connection()
        self._create_index_if_not_exists()

    def _ensure_connection(self):
        """Connect to Elasticsearch with retries."""
        import time
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if self.es.ping():
                    logger.info(f"Connected to Elasticsearch at {settings.elasticsearch_url}")
                    return
                
                alt_url = "http://localhost:39200"
                alt_es = Elasticsearch(
                    [alt_url],
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                
                if alt_es.ping():
                    logger.info(f"Successfully connected to Elasticsearch at {alt_url}")
                    self.es = alt_es
                    return
                
                raise ESConnectionError("Ping failed on both ports")

            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)

        logger.error("Could not connect to Elasticsearch after multiple attempts.")
        raise ESConnectionError(f"Cannot connect to Elasticsearch at {settings.elasticsearch_url}")

    def _create_index_if_not_exists(self):
        """Create index with proper mappings if it doesn't exist."""
        if self.es.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists")
            return

        mapping = {
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "filename": {"type": "text"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "summary": {"type": "text"},
                    "excerpt": {"type": "text"},
                    "category": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "source_filename": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "jurisdiction": {"type": "keyword"},
                    "tax_year": {"type": "integer"},
                    "client_name": {"type": "keyword"},
                    "entity_type": {"type": "keyword"},
                    "source_name": {"type": "keyword"},
                    "section_reference": {"type": "keyword"},
                    "upload_date": {"type": "date"},
                    "page_number": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": False},
                    "indexed_at": {"type": "date"},
                    "file_size": {"type": "long"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            }
        }

        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Created index '{self.index_name}'")

    def index_document(self, document: Dict[str, Any]) -> bool:
        """
        Index a single document.

        Args:
            document: Document data with summary

        Returns:
            bool: Success status
        """
        try:
            doc_id = document.get("document_id")

            es_doc = {
                "document_id": doc_id,
                "filename": document.get("filename", "Unknown"),
                "title": document.get("title", document.get("filename", "Unknown")),
                "content": document.get("content", ""),
                "summary": document.get("summary", ""),
                "excerpt": document.get("excerpt", ""),
                "category": document.get("category", "other"),
                "file_type": document.get("file_type", "unknown"),
                "source_filename": document.get("source_filename") or document.get("filename", "Unknown"),
                "chunk_id": document.get("chunk_id") or document.get("id"),
                "jurisdiction": document.get("jurisdiction"),
                "tax_year": document.get("tax_year"),
                "client_name": document.get("client_name"),
                "entity_type": document.get("entity_type"),
                "source_name": document.get("source_name") or document.get("filename", "Unknown"),
                "section_reference": document.get("section_reference"),
                "upload_date": document.get("upload_date"),
                "page_number": document.get("page_number") or document.get("page"),
                "chunk_index": document.get("chunk_index"),
                "total_chunks": document.get("total_chunks"),
                "metadata": document.get("metadata", {}),
                "indexed_at": datetime.utcnow().isoformat(),
                "file_size": document.get("metadata", {}).get("file_size", document.get("file_size", 0)),
                "embedding": document.get("embedding"),
            }

            self.es.index(
                index=self.index_name,
                id=doc_id,
                document=es_doc
            )

            logger.info(f"Indexed document: {es_doc['filename']} (ID: {doc_id})")
            return True

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False

    def bulk_index(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index multiple documents in bulk.

        Args:
            documents: List of documents to index

        Returns:
            dict: Statistics (success_count, error_count)
        """
        actions = []

        for doc in documents:
            doc_id = (
                doc.get("chunk_id")
                or doc.get("id")
                or f"{doc.get('document_id', 'doc')}-p{doc.get('page_number') or doc.get('page') or 0}-c{doc.get('chunk_index') or 1}"
            )

            action = {
                "_index": self.index_name,
                "_id": doc_id,
                "_source": {
                    "document_id": doc.get("document_id", doc_id),
                    "filename": doc.get("filename", "Unknown"),
                    "title": doc.get("title", doc.get("filename", "Unknown")),
                    "content": doc.get("content", ""),
                    "summary": doc.get("summary", ""),
                    "excerpt": doc.get("excerpt", ""),
                    "category": doc.get("category", "other"),
                    "file_type": doc.get("file_type", "unknown"),
                    "source_filename": doc.get("source_filename") or doc.get("filename", "Unknown"),
                    "chunk_id": doc.get("chunk_id") or doc.get("id") or doc_id,
                    "jurisdiction": doc.get("jurisdiction"),
                    "tax_year": doc.get("tax_year"),
                    "client_name": doc.get("client_name"),
                    "entity_type": doc.get("entity_type"),
                    "source_name": doc.get("source_name") or doc.get("filename", "Unknown"),
                    "section_reference": doc.get("section_reference"),
                    "upload_date": doc.get("upload_date"),
                    "page_number": doc.get("page_number") or doc.get("page"),
                    "chunk_index": doc.get("chunk_index"),
                    "total_chunks": doc.get("total_chunks"),
                    "metadata": doc.get("metadata", {}),
                    "indexed_at": datetime.utcnow().isoformat(),
                    "file_size": doc.get("metadata", {}).get("file_size", doc.get("file_size", 0)),
                    "embedding": doc.get("embedding"),
                }
            }
            actions.append(action)

        try:
            success, errors = helpers.bulk(
                self.es,
                actions,
                raise_on_error=False,
                raise_on_exception=False
            )

            error_count = len(errors) if isinstance(errors, list) else 0
            success_count = success

            logger.info(f"Bulk indexed {success_count} documents ({error_count} errors)")

            self.es.indices.refresh(index=self.index_name)

            return {
                "success_count": success_count,
                "error_count": error_count
            }

        except Exception as e:
            logger.error(f"Bulk indexing error: {e}")
            return {
                "success_count": 0,
                "error_count": len(documents)
            }

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        enable_fuzzy: bool = False,
        category: Optional[str] = None,
        file_type: Optional[str] = None,
        system_context: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        tax_year: Optional[int] = None,
        entity_type: Optional[str] = None,
        client_name: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents using Elasticsearch.

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Number of results to skip
            enable_fuzzy: Enable fuzzy matching
            category: Filter by category
            file_type: Filter by file type
            system_context: Legacy context boost value
            jurisdiction: Preferred legal/tax jurisdiction
            tax_year: Preferred tax year
            entity_type: Preferred entity type
            client_name: Preferred client name
            document_type: Preferred financial document type

        Returns:
            List of search results with summaries
        """
        try:
            must_queries = []

            if enable_fuzzy:
                must_queries.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "summary^1.5", "content"],
                        "fuzziness": "AUTO",
                        "prefix_length": 2
                    }
                })
            else:
                must_queries.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "summary^1.5", "content"],
                        "type": "best_fields"
                    }
                })

            filter_queries = []
            if category:
                filter_queries.append({"term": {"category": category}})
            if file_type:
                filter_queries.append({"term": {"file_type": file_type}})
            if document_type:
                filter_queries.append({"term": {"category": document_type}})
            if jurisdiction:
                filter_queries.append({"term": {"jurisdiction": jurisdiction}})
            if tax_year is not None:
                filter_queries.append({"term": {"tax_year": int(tax_year)}})
                
            should_queries = []
            if entity_type:
                should_queries.append({
                    "term": {"entity_type": {"value": entity_type, "boost": 2.5}}
                })
            if client_name:
                should_queries.append({
                    "term": {"client_name": {"value": client_name, "boost": 2.5}}
                })
            if system_context:
                should_queries.append({
                    "multi_match": {
                        "query": system_context,
                        "fields": ["content", "title", "summary", "category"],
                        "boost": 2.0
                    }
                })

            es_query = {
                "bool": {
                    "must": must_queries,
                    "filter": filter_queries,
                    "should": should_queries
                }
            }

            query_embedding = None
            try:
                query_embedding = self.embedding_service.get_embedding(query)
            except Exception as e:
                logger.warning(f"Failed to generate query embedding: {e}")

            if query_embedding:
                knn_query = {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": limit,
                    "num_candidates": 100,
                    "boost": 0.9
                }

                response = self.es.search(
                    index=self.index_name,
                    knn=knn_query,
                    query=es_query,
                    size=limit,
                    from_=offset,
                    _source=True
                )
            else:
                response = self.es.search(
                    index=self.index_name,
                    query=es_query,
                    size=limit,
                    from_=offset,
                    _source=True
                )

            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']

                snippet = source.get("excerpt", "")
                if not snippet:
                     snippet = source.get("summary", "")
                if not snippet and source.get("content"):
                     snippet = source.get("content")[:200] + "..."

                result = {
                    "document_id": source["document_id"],
                    "filename": source["filename"],
                    "title": source.get("title", source["filename"]),
                    "summary": source.get("summary") or "",
                    "excerpt": source.get("excerpt", ""),
                    "snippet": snippet,
                    "content": source.get("content", ""),
                    "category": source["category"],
                    "file_type": source["file_type"],
                    "jurisdiction": source.get("jurisdiction"),
                    "tax_year": source.get("tax_year"),
                    "entity_type": source.get("entity_type"),
                    "client_name": source.get("client_name"),
                    "source_name": source.get("source_name"),
                    "section_reference": source.get("section_reference"),
                    "source_filename": source.get("source_filename"),
                    "chunk_id": source.get("chunk_id"),
                    "page_number": source.get("page_number"),
                    "chunk_index": source.get("chunk_index"),
                    "total_chunks": source.get("total_chunks"),
                    "score": hit['_score'],
                    "metadata": source.get("metadata", {}),
                    "indexed_at": source.get("indexed_at")
                }
                results.append(result)

            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the index.

        Args:
            document_id: Document ID to delete

        Returns:
            bool: Success status
        """
        try:
            self.es.delete(index=self.index_name, id=document_id)
            logger.info(f"Deleted document: {document_id}")
            return True

        except NotFoundError:
            logger.warning(f"Document not found: {document_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    def get_document_count(self) -> int:
        """
        Get total number of indexed documents.

        Returns:
            int: Document count
        """
        try:
            response = self.es.count(index=self.index_name)
            return response['count']
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    def clear_index(self) -> bool:
        """
        Clear all documents from the index.

        Returns:
            bool: Success status
        """
        try:
            self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}}
            )
            self.es.indices.refresh(index=self.index_name)
            logger.info("Cleared all documents from index")
            return True

        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False


# Singleton instance
_client: Optional[ElasticsearchClient] = None


def get_elasticsearch_client() -> ElasticsearchClient:
    """
    Get the global Elasticsearch client instance.

    Returns:
        ElasticsearchClient: The client
    """
    global _client
    if _client is None:
        _client = ElasticsearchClient()
    return _client
