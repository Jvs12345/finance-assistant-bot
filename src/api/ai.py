"""Financial Q&A endpoints (legacy compatibility route)."""

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.services.ai_service import get_ai_service
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["AI"])


class AskRequest(BaseModel):
    """Request payload."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question to ask the AI"
    )
    model: Optional[str] = Field(
        default="llama3.2",
        description="Ollama model to use (e.g., 'llama3.2', 'mistral', 'phi3')"
    )
    max_context_docs: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of documents to use as context"
    )
    jurisdiction: Optional[str] = Field(default=None, description="Jurisdiction hint for retrieval")
    tax_year: Optional[int] = Field(default=None, description="Tax year hint for retrieval")
    entity_type: Optional[str] = Field(default=None, description="Entity type hint for retrieval")
    client_name: Optional[str] = Field(default=None, description="Client name hint for retrieval")
    document_type: Optional[str] = Field(default=None, description="Document type hint for retrieval")


class Source(BaseModel):
    """Source document reference."""
    filename: str
    page: Optional[int] = None
    score: float
    document_id: str


class AskResponse(BaseModel):
    """Response payload."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="AI-generated answer")
    sources: List[Source] = Field(..., description="Source documents used")
    model: str = Field(..., description="Model used to generate answer")
    found_documents: bool = Field(..., description="Whether relevant documents were found")
    search_results_count: Optional[int] = Field(None, description="Number of documents retrieved")


@router.post("/ask", response_model=AskResponse)
async def ask_ai(request: AskRequest) -> AskResponse:
    """Ask a question and return an answer with sources."""
    try:
        logger.info(f"AI ask: '{request.question}' using model '{request.model}'")

        ai_service = get_ai_service(model=request.model)
        result = ai_service.ask(
            question=request.question,
            max_context_docs=request.max_context_docs,
            jurisdiction=request.jurisdiction,
            tax_year=request.tax_year,
            entity_type=request.entity_type,
            client_name=request.client_name,
            document_type=request.document_type,
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result["answer"]
            )

        return AskResponse(
            question=request.question,
            answer=result["answer"],
            sources=[Source(**source) for source in result["sources"]],
            model=result["model"],
            found_documents=result["found_documents"],
            search_results_count=result.get("search_results_count")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask AI failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}"
        )


@router.get("/models")
async def list_models():
    """List local Ollama models."""
    try:
        ai_service = get_ai_service()
        models = ai_service.list_available_models()

        return {
            "models": models,
            "count": len(models),
            "message": "Install models with: ollama pull <model-name>" if not models else None
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {
            "models": [],
            "count": 0,
            "error": str(e),
            "message": "Make sure Ollama is running"
        }
