"""API endpoints for financial document Q&A."""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.llama_service import get_llama_service
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/llama", tags=["Financial Q&A"])


class LlamaQuestionRequest(BaseModel):
    """Request model for Llama question."""

    question: str = Field(..., description="User's question", min_length=3)
    max_context_docs: int = Field(default=settings.retrieval_top_k, ge=1, le=20, description="Max documents for retrieval")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Response creativity (0=focused, 1=creative)")
    model: str = Field(default=settings.ollama_model, description="Ollama model to use")
    system_context: Optional[str] = Field(default=None, description="Legacy context field (deprecated)")
    jurisdiction: Optional[str] = Field(default=None, description="Jurisdiction to prioritize (e.g., Netherlands, EU)")
    tax_year: Optional[int] = Field(default=None, description="Tax year to prioritize")
    document_type: Optional[str] = Field(default=None, description="Preferred document type (tax_law, regulation, etc.)")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation history [{'role': 'user', 'content': '...'}, ...]")


class SourceDocument(BaseModel):
    """Source document used for answer."""

    document_id: str
    filename: str
    title: str
    score: float
    category: str
    document_type_detail: Optional[str] = None
    page: Optional[int] = None
    snippet: Optional[str] = None
    jurisdiction: Optional[str] = None
    tax_year: Optional[int] = None
    entity_type: Optional[str] = None
    client_name: Optional[str] = None
    section_reference: Optional[str] = None
    corpus_type: Optional[str] = None


class LlamaAnswerResponse(BaseModel):
    """Response model for Llama answer."""

    answer: str
    sources: List[SourceDocument]
    model: str
    found_documents: bool
    num_documents_used: Optional[int] = None
    filters_used: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


@router.post("/ask", response_model=LlamaAnswerResponse)
async def ask_llama(request: LlamaQuestionRequest):
    """Main Q&A endpoint."""
    try:
        llama_service = get_llama_service(model=request.model)

        result = llama_service.ask(
            question=request.question,
            max_context_docs=request.max_context_docs,
            temperature=request.temperature,
            system_context=request.system_context,
            jurisdiction=request.jurisdiction,
            tax_year=request.tax_year,
            document_type=request.document_type,
            history=request.history
        )

        return LlamaAnswerResponse(**result)

    except Exception as e:
        logger.error(f"Error in Llama Q&A: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List local Ollama models."""
    try:
        llama_service = get_llama_service()
        models = llama_service.list_available_models()

        return {
            "models": models,
            "default": settings.ollama_model,
            "recommended": ["llama3.2", "llama3.1", "mistral"]
        }

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def check_status():
    """Check Ollama availability."""
    try:
        llama_service = get_llama_service()
        models = llama_service.list_available_models()

        if models:
            return {
                "status": "available",
                "ollama_running": True,
                "models_installed": len(models),
                "models": models
            }
        else:
            return {
                "status": "not_available",
                "ollama_running": False,
                "message": "Ollama is not running. Please start it with: ollama serve"
            }

    except Exception as e:
        return {
            "status": "error",
            "ollama_running": False,
            "error": str(e),
            "message": "Install Ollama from: https://ollama.com"
        }
