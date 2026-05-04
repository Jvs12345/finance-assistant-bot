"""
Search service using Elasticsearch for document search.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from src.db.elasticsearch_client import get_elasticsearch_client
from src.db.postgres import get_postgres_client
from src.models.search import SearchRequest, SearchResponse, SearchResult, SearchFilters
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeedbackCache:
    """Simple in-memory cache for feedback boost scores with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize feedback cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default 5 minutes)
        """
        self.cache: Dict[str, tuple[float, float]] = {}  # key -> (boost_score, expiry_time)
        self.ttl_seconds = ttl_seconds

    def get(self, document_id: str, page: int) -> Optional[float]:
        """
        Get boost score from cache if not expired.

        Args:
            document_id: Document identifier
            page: Page number

        Returns:
            float: Boost score or None if not cached or expired
        """
        key = f"{document_id}:{page}"
        if key in self.cache:
            boost_score, expiry_time = self.cache[key]
            if time.time() < expiry_time:
                return boost_score
            else:
                # Expired, remove from cache
                del self.cache[key]
        return None

    def set(self, document_id: str, page: int, boost_score: float) -> None:
        """
        Store boost score in cache with TTL.

        Args:
            document_id: Document identifier
            page: Page number
            boost_score: Boost multiplier to cache
        """
        key = f"{document_id}:{page}"
        expiry_time = time.time() + self.ttl_seconds
        self.cache[key] = (boost_score, expiry_time)

    def invalidate(self, document_id: str, page: int) -> None:
        """
        Invalidate cache entry for a specific document page.

        Args:
            document_id: Document identifier
            page: Page number
        """
        key = f"{document_id}:{page}"
        if key in self.cache:
            del self.cache[key]


class SearchService:
    """Service for searching documents using Elasticsearch."""

    def __init__(self):
        """Initialize search service with Elasticsearch client."""
        self.search_client = get_elasticsearch_client()
        self.pg_client = get_postgres_client()
        self.feedback_cache = FeedbackCache(ttl_seconds=300)  # 5-minute cache

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute a search request.

        Args:
            request: Search request with query and parameters

        Returns:
            SearchResponse: Search results with metadata
        """
        start_time = time.time()
        logger.debug(f"Starting search with query: {request.query}")
        try:
            # Calculate pagination
            offset = (request.page - 1) * request.page_size

            # Execute search
            logger.debug(f"Executing search with offset={offset}, limit={request.page_size}")
            results = self.search_client.search(
                query=request.query,
                limit=request.page_size,
                offset=offset,
                enable_fuzzy=request.enable_fuzzy,
                category=request.filters.category.value if request.filters and request.filters.category else None,
                jurisdiction=request.filters.jurisdiction if request.filters else None,
                tax_year=request.filters.tax_year if request.filters else None,
                entity_type=request.filters.entity_type if request.filters else None,
                client_name=request.filters.client_name if request.filters else None,
            )
            logger.debug(f"Search returned {len(results)} results")

            # Convert to response model
            search_results = []
            for hit in results:
                try:
                    # Apply feedback boosting to score
                    document_id = hit["document_id"]
                    page_num = hit.get("page_number", 1) or 1 # Safety default
                    base_score = hit["score"]
                    logger.debug(f"Processing hit: doc={document_id}, page={page_num}, score={base_score}")
                except KeyError as e:
                    logger.error(f"Missing field in search result: {e}")
                    logger.debug(f"Available fields: {hit.keys()}")
                    continue
                
                feedback_boost = self.get_feedback_boost(document_id, page_num)
                boosted_score = base_score * feedback_boost

                result = SearchResult(
                    document_id=document_id,
                    page=page_num,
                    filename=hit["filename"],
                    score=float(boosted_score),  # Ensure score is float
                    snippet=hit.get("snippet", ""),  # Provide default empty string
                    content=hit.get("content"),  # Full plain content
                    highlighted_content=hit.get("highlighted_content"),  # Full content with highlights
                    upload_date=hit.get("indexed_at") # Use nice get
                )
                logger.debug(f"Created search result: {result}")
                search_results.append(result)

            # Sort by boosted score
            search_results.sort(key=lambda r: r.score, reverse=True)

            # Calculate response metadata
            took = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            total = len(results)
            total_pages = (total + request.page_size - 1) // request.page_size

            response = SearchResponse(
                query=request.query,
                total=total,
                page=request.page,
                page_size=request.page_size,
                took=took,
                results=search_results
            )

            logger.info(
                f"Search completed: {response.total} results in {response.took}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _build_query(self, request: SearchRequest) -> Dict[str, Any]:
        """
        Build Elasticsearch query from search request.

        Args:
            request: Search request parameters

        Returns:
            dict: Elasticsearch query DSL
        """
        # Build multi-match query with field boosting
        query_clause = {
            "multi_match": {
                "query": request.query,
                "fields": [
                    "jurisdiction^3",
                    "tax_year^3",
                    "entity_type^2.5",
                    "client_name^2.2",
                    "content^2",
                    "summary^1.5",
                    "filename^1.2"
                ],
                "type": "best_fields",
                "operator": "or"
            }
        }

        # Add fuzzy matching if enabled
        if request.enable_fuzzy:
            query_clause["multi_match"]["fuzziness"] = "AUTO"

        # Build bool query with filters
        bool_query = {
            "must": [query_clause]
        }

        # Add filters
        if request.filters:
            filter_clauses = self._build_filters(request.filters)
            if filter_clauses:
                bool_query["filter"] = filter_clauses

        # Complete query structure
        es_query = {
            "query": {
                "bool": bool_query
            },
            "sort": [
                "_score",  # Sort by relevance score
                {"upload_date": {"order": "desc"}}  # Then by upload date
            ]
        }

        # Add highlighting if requested
        if request.include_highlights:
            highlight_fields = {
                "summary": {
                    "fragment_size": 150,
                    "number_of_fragments": 1,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            }

            # Add content highlighting
            if request.include_content:
                # Full-field highlighting for full content view
                highlight_fields["content"] = {
                    "number_of_fragments": 0,  # Return entire field highlighted
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            else:
                # Fragment highlighting for snippet view
                highlight_fields["content"] = {
                    "fragment_size": 150,
                    "number_of_fragments": 1,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }

            es_query["highlight"] = {
                "fields": highlight_fields,
                "order": "score"
            }

        return es_query

    def _build_filters(self, filters: SearchFilters) -> List[Dict[str, Any]]:
        """
        Build Elasticsearch filter clauses from search filters.

        Args:
            filters: Search filters

        Returns:
            list: List of filter clauses
        """
        filter_clauses = []

        # Category filter
        if filters.category:
            filter_clauses.append({
                "term": {"category": filters.category.value}
            })

        # Jurisdiction filter
        if filters.jurisdiction:
            filter_clauses.append({
                "term": {"jurisdiction": filters.jurisdiction}
            })

        # Tax year filter
        if filters.tax_year is not None:
            filter_clauses.append({
                "term": {"tax_year": filters.tax_year}
            })

        # Client/entity filters
        if filters.client_name:
            filter_clauses.append({
                "term": {"client_name": filters.client_name}
            })
        if filters.entity_type:
            filter_clauses.append({
                "term": {"entity_type": filters.entity_type}
            })

        # Date range filter
        if filters.date_from or filters.date_to:
            date_range = {}
            if filters.date_from:
                date_range["gte"] = filters.date_from.isoformat()
            if filters.date_to:
                date_range["lte"] = filters.date_to.isoformat()

            filter_clauses.append({
                "range": {"upload_date": date_range}
            })

        # Part numbers filter
        if filters.part_numbers:
            filter_clauses.append({
                "terms": {"part_numbers": filters.part_numbers}
            })

        return filter_clauses

    def get_feedback_boost(self, document_id: str, page: int) -> float:
        """
        Get feedback boost score for a document page.

        Args:
            document_id: Document identifier
            page: Page number

        Returns:
            float: Boost multiplier (1.0 = neutral, >1.0 = boost, <1.0 = penalty)
        """
        # Check cache first
        cached_boost = self.feedback_cache.get(document_id, page)
        if cached_boost is not None:
            return cached_boost

        # Fetch from database
        try:
            stats = self.pg_client.get_feedback_stats(document_id, page)
            positive_count = stats["positive_count"]
            negative_count = stats["negative_count"]

            # Calculate boost: 1.0 + (net_votes * 0.1)
            # Each net positive vote adds 10% boost
            # Each net negative vote subtracts 10% boost
            net_votes = positive_count - negative_count
            boost = 1.0 + (net_votes * 0.1)

            # Clamp to reasonable range (0.1 to 3.0)
            boost = max(0.1, min(3.0, boost))

            # Cache the result
            self.feedback_cache.set(document_id, page, boost)

            return boost

        except Exception as e:
            logger.warning(f"Failed to get feedback boost for {document_id}:{page}: {e}")
            return 1.0  # Neutral boost on error

    def invalidate_feedback_cache(self, document_id: str, page: int) -> None:
        """
        Invalidate feedback cache for a document page.
        Call this when new feedback is submitted.

        Args:
            document_id: Document identifier
            page: Page number
        """
        self.feedback_cache.invalidate(document_id, page)

    def get_page_content(self, document_id: str, page: int) -> Dict[str, Any]:
        """
        Get the full content of a specific document page.

        Args:
            document_id: Document identifier
            page: Page number

        Returns:
            dict with 'content', 'snippet', 'highlighted_content', 'indexed_at'
        """
        try:
            # Search for this specific page
            results = self.search_client.search(
                query=f"document_id:{document_id} AND page_number:{page}",
                limit=1,
                offset=0
            )

            if results:
                return results[0]
            else:
                return {
                    "content": "",
                    "snippet": "",
                    "highlighted_content": "",
                    "indexed_at": None
                }
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return {
                "content": "",
                "snippet": "",
                "highlighted_content": "",
                "indexed_at": None
            }

    def _parse_response(
        self,
        es_response: Dict[str, Any],
        query: str,
        page: int,
        page_size: int,
        include_highlights: bool,
        include_content: bool = True
    ) -> SearchResponse:
        """
        Parse Elasticsearch response into SearchResponse model.
        Applies feedback boosting to scores and re-sorts results.

        Args:
            es_response: Raw Elasticsearch response
            query: Original search query
            page: Current page number
            page_size: Results per page
            include_highlights: Whether to include highlighted snippets
            include_content: Whether to include full page content

        Returns:
            SearchResponse: Parsed search response
        """
        total = es_response["hits"]["total"]["value"]
        took = es_response["took"]

        results = []
        for hit in es_response["hits"]["hits"]:
            source = hit["_source"]
            base_score = hit["_score"]

            # Apply feedback boosting to score
            document_id = source["document_id"]
            page_num = source["page"]
            feedback_boost = self.get_feedback_boost(document_id, page_num)
            boosted_score = base_score * feedback_boost

            # Log if boost is significant
            if feedback_boost != 1.0:
                logger.debug(
                    f"Feedback boost applied: {document_id}:{page_num} "
                    f"base={base_score:.2f} boost={feedback_boost:.2f} final={boosted_score:.2f}"
                )

            # Extract highlight snippet and full highlighted content
            snippet = None
            highlighted_content = None

            if include_highlights and "highlight" in hit:
                # Extract highlighted content (full-field when include_content=True)
                if "content" in hit["highlight"] and include_content:
                    # Full-field highlighting - use for highlighted_content
                    highlighted_content = hit["highlight"]["content"][0]

                # Extract snippet for preview (prefer summary, fall back to content fragment)
                if "summary" in hit["highlight"]:
                    snippet = hit["highlight"]["summary"][0]
                elif "content" in hit["highlight"] and not include_content:
                    # Content fragment (when include_content=False) - use for snippet
                    snippet = hit["highlight"]["content"][0]

            # Create search result with boosted score
            result = SearchResult(
                document_id=source["document_id"],
                filename=source["filename"],
                page=source["page"],
                score=boosted_score,  # Use boosted score
                snippet=snippet,
                content=source.get("content") if include_content else None,
                highlighted_content=highlighted_content,
                upload_date=source.get("upload_date")
            )
            results.append(result)

        # Re-sort results by boosted score (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        return SearchResponse(
            query=query,
            total=total,
            page=page,
            page_size=page_size,
            took=took,
            results=results
        )


# Global search service instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """
    Get the global search service instance (singleton pattern).

    Returns:
        SearchService: The search service instance
    """
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
