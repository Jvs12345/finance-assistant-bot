"""
Semantic search index using Elasticsearch with dense vector embeddings.

This module provides semantic search over document summaries using
kNN vector similarity, with optional BM25 hybrid search.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError, ConnectionError as ESConnectionError

from src.utils.logging import get_logger
from src.config import settings
from src.services.embedding_service import get_embedding_service

logger = get_logger(__name__)


# Configuration from environment
SEARCH_MODE = os.getenv("SEARCH_MODE", "semantic")  # semantic, bm25, or hybrid
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))  # Alpha for hybrid
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))  # Embedding dimension


class SemanticSearchClient:
    """
    Elasticsearch client with semantic search capabilities.

    Features:
    - Pure semantic search using kNN over summary embeddings
    - Optional BM25 keyword search
    - Hybrid search combining semantic + BM25
    - Document-level retrieval (not chunks/snippets)
    """

    def __init__(self, index_name: Optional[str] = None):
        """
        Initialize semantic search client.

        Args:
            index_name: Index name (default: documents_semantic)
        """
        self.es = Elasticsearch(
            [settings.elasticsearch_url],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )

        self.index_name = index_name or os.getenv(
            "SEMANTIC_INDEX_NAME", "documents_semantic"
        )

        self.embedding_service = get_embedding_service()
        self.search_mode = SEARCH_MODE
        self.semantic_weight = SEMANTIC_WEIGHT

        # Ensure connection and index
        self._ensure_connection()
        self._create_index_if_not_exists()

    def _ensure_connection(self):
        """Ensure Elasticsearch is accessible."""
        try:
            if not self.es.ping():
                raise ESConnectionError("Cannot connect to Elasticsearch")
            logger.info(f"Connected to Elasticsearch at {settings.elasticsearch_url}")
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            raise

    def _create_index_if_not_exists(self):
        """
        Create semantic search index with vector field if it doesn't exist.

        Index schema:
        - id: Document identifier
        - title: Document title (text + keyword)
        - summary_short: Brief 1-3 sentence summary (text)
        - summary_long: Detailed 1-2 paragraph summary (text)
        - full_text: Complete document content (text)
        - summary_embedding: Dense vector for semantic search
        - metadata: Document metadata (product, version, locale, etc.)
        - category: Document category (keyword)
        - file_type: File type (keyword)
        - indexed_at: Timestamp
        """
        if self.es.indices.exists(index=self.index_name):
            logger.info(f"Semantic index '{self.index_name}' already exists")
            return

        # Get embedding dimension from service
        embedding_dim = self.embedding_service.dimension

        mapping = {
            "mappings": {
                "properties": {
                    # Core fields
                    "document_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },

                    # Summary fields (for search and display)
                    "summary_short": {"type": "text"},
                    "summary_long": {"type": "text"},
                    "full_text": {"type": "text"},

                    # Vector embedding for semantic search
                    "summary_embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dim,
                        "index": True,
                        "similarity": "cosine"  # cosine, dot_product, or l2_norm
                    },

                    # Metadata and filters
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "keyword"},
                            "version": {"type": "keyword"},
                            "locale": {"type": "keyword"},
                            "tags": {"type": "keyword"}
                        }
                    },
                    "category": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "filename": {"type": "keyword"},

                    # Timestamps
                    "indexed_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
                # kNN is enabled by default in ES 8.x when using dense_vector with index: true
            }
        }

        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(
            f"Created semantic index '{self.index_name}' with {embedding_dim}-dim vectors"
        )

    def index_document(
        self,
        document_id: str,
        title: str,
        summary_short: str,
        summary_long: str,
        full_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        file_type: Optional[str] = None,
        filename: Optional[str] = None
    ) -> bool:
        """
        Index a single document with semantic embedding.

        Args:
            document_id: Unique document identifier
            title: Document title
            summary_short: Brief 1-3 sentence summary
            summary_long: Detailed 1-2 paragraph summary
            full_text: Complete document text
            metadata: Additional metadata (product, version, locale, tags, etc.)
            category: Document category
            file_type: File type
            filename: Original filename

        Returns:
            bool: Success status
        """
        try:
            # Generate embedding from title + summary_long
            embedding_text = f"{title}\n\n{summary_long}"
            embedding = self.embedding_service.get_embedding(embedding_text)

            # Prepare document
            doc = {
                "document_id": document_id,
                "title": title,
                "summary_short": summary_short,
                "summary_long": summary_long,
                "full_text": full_text,
                "summary_embedding": embedding,
                "metadata": metadata or {},
                "category": category or "other",
                "file_type": file_type or "unknown",
                "filename": filename or "Unknown",
                "indexed_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Index document
            self.es.index(index=self.index_name, id=document_id, document=doc)

            logger.info(f"Indexed document: {title} (ID: {document_id})")
            return True

        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {e}")
            return False

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure semantic search using kNN over summary embeddings.

        Args:
            query: User query text
            limit: Maximum results to return
            filters: Metadata filters (category, product, version, locale, etc.)

        Returns:
            List of search results with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.get_embedding(query)

            # Build filter clauses
            filter_clauses = []
            if filters:
                if "category" in filters:
                    filter_clauses.append({"term": {"category": filters["category"]}})
                if "file_type" in filters:
                    filter_clauses.append({"term": {"file_type": filters["file_type"]}})
                if "product" in filters:
                    filter_clauses.append({"term": {"metadata.product": filters["product"]}})
                if "version" in filters:
                    filter_clauses.append({"term": {"metadata.version": filters["version"]}})
                if "locale" in filters:
                    filter_clauses.append({"term": {"metadata.locale": filters["locale"]}})

            # kNN search query
            knn_query = {
                "field": "summary_embedding",
                "query_vector": query_embedding,
                "k": limit,
                "num_candidates": min(limit * 10, 1000)  # Search broader, return top k
            }

            # Add filters if present
            if filter_clauses:
                knn_query["filter"] = {"bool": {"must": filter_clauses}}

            # Execute kNN search
            response = self.es.search(
                index=self.index_name,
                knn=knn_query,
                size=limit,
                _source=True
            )

            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']

                result = {
                    "document_id": source["document_id"],
                    "title": source["title"],
                    "summary_short": source.get("summary_short", ""),
                    "summary_long": source.get("summary_long", ""),
                    "full_text": source.get("full_text", ""),
                    "filename": source.get("filename", "Unknown"),
                    "category": source.get("category", "other"),
                    "file_type": source.get("file_type", "unknown"),
                    "metadata": source.get("metadata", {}),
                    "score": hit['_score'],
                    "indexed_at": source.get("indexed_at"),
                    "search_type": "semantic"
                }
                results.append(result)

            logger.info(
                f"Semantic search for '{query}' returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def bm25_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        BM25 keyword search over title and summaries.

        Args:
            query: User query text
            limit: Maximum results to return
            filters: Metadata filters

        Returns:
            List of search results with BM25 scores
        """
        try:
            # Build query
            must_queries = [{
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "summary_long^2", "summary_short^1.5", "full_text"],
                    "type": "best_fields"
                }
            }]

            # Build filter clauses
            filter_clauses = []
            if filters:
                if "category" in filters:
                    filter_clauses.append({"term": {"category": filters["category"]}})
                if "file_type" in filters:
                    filter_clauses.append({"term": {"file_type": filters["file_type"]}})
                if "product" in filters:
                    filter_clauses.append({"term": {"metadata.product": filters["product"]}})

            # Complete query
            es_query = {
                "bool": {
                    "must": must_queries,
                    "filter": filter_clauses
                }
            }

            # Execute search
            response = self.es.search(
                index=self.index_name,
                query=es_query,
                size=limit,
                _source=True
            )

            # Format results
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']

                result = {
                    "document_id": source["document_id"],
                    "title": source["title"],
                    "summary_short": source.get("summary_short", ""),
                    "summary_long": source.get("summary_long", ""),
                    "full_text": source.get("full_text", ""),
                    "filename": source.get("filename", "Unknown"),
                    "category": source.get("category", "other"),
                    "file_type": source.get("file_type", "unknown"),
                    "metadata": source.get("metadata", {}),
                    "score": hit['_score'],
                    "indexed_at": source.get("indexed_at"),
                    "search_type": "bm25"
                }
                results.append(result)

            logger.info(f"BM25 search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and BM25 results.

        Args:
            query: User query text
            limit: Maximum results to return
            filters: Metadata filters
            alpha: Weight for semantic score (default from SEMANTIC_WEIGHT env)
                   final_score = alpha * semantic + (1 - alpha) * bm25

        Returns:
            List of merged and re-ranked results
        """
        if alpha is None:
            alpha = self.semantic_weight

        # Get both result sets
        semantic_results = self.semantic_search(query, limit=limit * 2, filters=filters)
        bm25_results = self.bm25_search(query, limit=limit * 2, filters=filters)

        # Normalize scores for each result set
        def normalize_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
            """Normalize scores to 0-1 range."""
            if not results:
                return {}

            scores = [r["score"] for r in results]
            min_score = min(scores)
            max_score = max(scores)

            if max_score == min_score:
                return {r["document_id"]: 1.0 for r in results}

            return {
                r["document_id"]: (r["score"] - min_score) / (max_score - min_score)
                for r in results
            }

        semantic_scores = normalize_scores(semantic_results)
        bm25_scores = normalize_scores(bm25_results)

        # Merge results by document ID
        merged: Dict[str, Dict[str, Any]] = {}

        for result in semantic_results:
            doc_id = result["document_id"]
            merged[doc_id] = result
            merged[doc_id]["semantic_score"] = semantic_scores[doc_id]
            merged[doc_id]["bm25_score"] = 0.0
            merged[doc_id]["search_type"] = "hybrid"

        for result in bm25_results:
            doc_id = result["document_id"]
            if doc_id in merged:
                merged[doc_id]["bm25_score"] = bm25_scores[doc_id]
            else:
                merged[doc_id] = result
                merged[doc_id]["semantic_score"] = 0.0
                merged[doc_id]["bm25_score"] = bm25_scores[doc_id]
                merged[doc_id]["search_type"] = "hybrid"

        # Calculate hybrid scores
        for doc_id, result in merged.items():
            semantic = result.get("semantic_score", 0.0)
            bm25 = result.get("bm25_score", 0.0)
            result["hybrid_score"] = alpha * semantic + (1 - alpha) * bm25
            result["score"] = result["hybrid_score"]  # Use hybrid as primary score

        # Sort by hybrid score and return top results
        ranked_results = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )[:limit]

        logger.info(
            f"Hybrid search for '{query}' returned {len(ranked_results)} results "
            f"(alpha={alpha:.2f})"
        )
        return ranked_results

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Unified search interface supporting semantic, BM25, or hybrid modes.

        Args:
            query: User query text
            limit: Maximum results to return
            filters: Metadata filters
            mode: Search mode override ('semantic', 'bm25', 'hybrid')
                  Defaults to SEARCH_MODE environment variable

        Returns:
            List of search results
        """
        search_mode = mode or self.search_mode

        if search_mode == "semantic":
            return self.semantic_search(query, limit, filters)
        elif search_mode == "bm25":
            return self.bm25_search(query, limit, filters)
        elif search_mode == "hybrid":
            return self.hybrid_search(query, limit, filters)
        else:
            logger.warning(f"Unknown search mode '{search_mode}', using semantic")
            return self.semantic_search(query, limit, filters)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the index."""
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
        """Get total number of indexed documents."""
        try:
            response = self.es.count(index=self.index_name)
            return response['count']
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0


# Singleton instance
_semantic_client: Optional[SemanticSearchClient] = None


def get_semantic_search_client() -> SemanticSearchClient:
    """
    Get the global semantic search client instance.

    Returns:
        SemanticSearchClient: The client
    """
    global _semantic_client
    if _semantic_client is None:
        _semantic_client = SemanticSearchClient()
    return _semantic_client
