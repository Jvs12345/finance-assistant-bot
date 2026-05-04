"""
Search API endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional

from src.models.search import SearchRequest, SearchResponse, SearchFilters
from src.services.search_service import get_search_service
from src.services.ai_service import get_ai_service
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    ai_model: str = Query(default="llama3.2", description="Ollama model to use for AI answers")
) -> SearchResponse:
    """Search documents and return an answer plus source snippets."""
    try:
        logger.info(f"AI Search request: query='{request.query}' with model '{ai_model}'")

        ai_service = get_ai_service(model=ai_model)
        ai_result = ai_service.ask(
            question=request.query,
            max_context_docs=request.page_size
        )

        search_response = SearchResponse(
            query=request.query,
            total=ai_result.get("search_results_count", 0),
            page=request.page,
            page_size=request.page_size,
            took=0,
            results=[],
            ai_answer=ai_result["answer"],
            ai_model=ai_result["model"]
        )

        from src.models.search import SearchResult

        for source in ai_result.get("sources", []):
            if not source.get("document_id"):
                continue
            result = SearchResult(
                document_id=source["document_id"],
                filename=source["filename"],
                page=source["page"],
                score=source.get("score", 0.0),
                snippet=source.get(
                    "snippet",
                    f"Page {source.get('page')} from {source.get('filename')}"
                ) or f"Page {source.get('page')} from {source.get('filename')}",
                content=source.get("content"),
                highlighted_content=source.get("highlighted_content"),
                upload_date=source.get("indexed_at")
            )
            search_response.results.append(result)

        search_response.total = len(search_response.results)

        logger.info(
            f"AI Search completed: {search_response.total} results, "
            f"AI answer: {len(ai_result['answer'])} chars"
        )

        return search_response

    except ValueError as e:
        logger.warning(f"Invalid search request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"AI Search failed: {e}", exc_info=True)
        try:
            search_service = get_search_service()
            response = search_service.search(request)
            response.ai_answer = f"AI temporarily unavailable: {str(e)}"
            return response
        except:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search failed. Please try again later."
            )
