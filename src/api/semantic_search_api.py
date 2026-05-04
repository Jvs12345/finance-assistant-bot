"""
Semantic search API endpoints.
Provides REST API for semantic PDF search.
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from src.services.semantic_search_engine import SemanticSearchEngine, SearchResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/semantic", tags=["Semantic Search"])


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of results")
    generate_answer: bool = Field(default=True, description="Generate direct answer")


class SearchResultResponse(BaseModel):
    """Single search result."""
    doc_id: str
    title: str
    snippet: str
    score: float
    source: str
    page_num: Optional[int] = None
    section_type: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    """Semantic search response."""
    query: str
    direct_answer: Optional[str] = None
    results: list[SearchResultResponse]
    total_results: int
    search_time_ms: float


# Global search engine instance
_search_engine: Optional[SemanticSearchEngine] = None


def get_search_engine() -> SemanticSearchEngine:
    """Get or create search engine instance."""
    global _search_engine

    if _search_engine is None:
        index_path = Path("./data/semantic_index")

        if not (index_path / "index.json").exists():
            raise HTTPException(
                status_code=503,
                detail="Search index not available. Please run INDEX_PDFS.bat first."
            )

        _search_engine = SemanticSearchEngine(
            index_path=index_path,
            use_embeddings=False,
            use_llm_rerank=False
        )
        logger.info("Semantic search engine initialized")

    return _search_engine


@router.post("/search", response_model=SemanticSearchResponse)
async def search(request: SearchRequest):
    """
    Perform semantic search over indexed PDFs.

    Returns Google-like search results with optional direct answer.
    """
    try:
        engine = get_search_engine()

        # Perform search
        response = engine.search(
            query=request.query,
            top_k=request.top_k,
            generate_answer=request.generate_answer
        )

        # Convert to API response
        results = [
            SearchResultResponse(
                doc_id=r.doc_id,
                title=r.title,
                snippet=r.snippet,
                score=r.score,
                source=r.source,
                page_num=r.page_num,
                section_type=r.section_type
            )
            for r in response.results
        ]

        return SemanticSearchResponse(
            query=response.query,
            direct_answer=response.direct_answer,
            results=results,
            total_results=response.total_results,
            search_time_ms=response.search_time_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search", response_model=SemanticSearchResponse)
async def search_get(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=8, ge=1, le=50, description="Number of results"),
    generate_answer: bool = Query(default=True, description="Generate direct answer")
):
    """
    Perform semantic search (GET method for simple queries).
    """
    request = SearchRequest(
        query=q,
        top_k=top_k,
        generate_answer=generate_answer
    )
    return await search(request)


@router.get("/status")
async def status():
    """
    Get semantic search system status.
    """
    index_path = Path("./data/semantic_index")
    index_exists = (index_path / "index.json").exists()

    status_info = {
        "available": index_exists,
        "index_path": str(index_path)
    }

    if index_exists:
        try:
            from src.services.semantic_indexer import SemanticIndexer
            indexer = SemanticIndexer(index_dir=index_path)
            summary = indexer.get_summary()

            status_info.update({
                "total_documents": summary.get("total_documents", 0),
                "total_sections": summary.get("total_sections", 0),
                "last_updated": summary.get("last_updated"),
            })
        except Exception as e:
            logger.warning(f"Failed to load index summary: {e}")

    return status_info


@router.get("/summary")
async def get_summary():
    """
    Get index summary and statistics.
    """
    index_path = Path("./data/semantic_index")

    if not (index_path / "summary.json").exists():
        raise HTTPException(
            status_code=404,
            detail="No index summary found. Please run INDEX_PDFS.bat first."
        )

    try:
        from src.services.semantic_indexer import SemanticIndexer
        indexer = SemanticIndexer(index_dir=index_path)
        summary = indexer.get_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to load summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load summary: {str(e)}")


@router.get("/issues")
async def get_issues():
    """
    Get logged issues from indexing process.
    """
    index_path = Path("./data/semantic_index")

    if not (index_path / "issues.json").exists():
        return {"issues": []}

    try:
        from src.services.semantic_indexer import SemanticIndexer
        indexer = SemanticIndexer(index_dir=index_path)
        issues = indexer.get_issues()
        return {"issues": issues, "total": len(issues)}
    except Exception as e:
        logger.error(f"Failed to load issues: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load issues: {str(e)}")
