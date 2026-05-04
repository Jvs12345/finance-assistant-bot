"""
API endpoints for semantic search over financial documents.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.db.elasticsearch_client import get_elasticsearch_client
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/semantic", tags=["Semantic Search"])


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    enable_fuzzy: bool = Field(default=False, description="Enable fuzzy matching")
    category: Optional[str] = Field(None, description="Filter by category")
    file_type: Optional[str] = Field(None, description="Filter by file type")


class SemanticSearchResult(BaseModel):
    """Single search result with summary."""

    document_id: str
    filename: str
    title: str
    summary: str
    category: str
    file_type: str
    score: float
    metadata: dict
    indexed_at: Optional[str]


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search."""

    results: List[SemanticSearchResult]
    total: int
    query: str
    took_ms: float


@router.post("/search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """Run semantic search over indexed documents."""
    import time

    start_time = time.time()

    try:
        search_client = get_elasticsearch_client()

        results = search_client.search(
            query=request.query,
            limit=request.limit,
            enable_fuzzy=request.enable_fuzzy,
            category=request.category,
            file_type=request.file_type
        )

        took_ms = (time.time() - start_time) * 1000

        return SemanticSearchResponse(
            results=[SemanticSearchResult(**r) for r in results],
            total=len(results),
            query=request.query,
            took_ms=took_ms
        )

    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_search_stats():
    """
    Get search index statistics.

    Returns:
        Statistics about indexed documents
    """
    try:
        search_client = get_elasticsearch_client()
        doc_count = search_client.get_document_count()

        return {
            "total_documents": doc_count,
            "index_name": search_client.index_name,
            "status": "ready"
        }

    except Exception as e:
        logger.error(f"Error getting search stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index")
async def clear_search_index():
    """
    Clear all documents from search index.

    WARNING: This will delete all indexed documents!

    Returns:
        Success status
    """
    try:
        search_client = get_elasticsearch_client()
        success = search_client.clear_index()

        if success:
            return {
                "message": "Search index cleared successfully",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear index")

    except Exception as e:
        logger.error(f"Error clearing search index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
