"""
Feedback API endpoints.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from typing import Dict

from src.models.feedback import FeedbackRequest, FeedbackResponse, FeedbackStats
from src.db.postgres import get_postgres_client
from src.services.search_service import get_search_service
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/feedback",
    tags=["feedback"]
)


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback for a search result",
    description="Submit positive or negative feedback for a search result. Anonymous feedback is supported."
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit feedback for a search result.

    Args:
        request: Feedback submission data

    Returns:
        FeedbackResponse: Confirmation with feedback details

    Raises:
        HTTPException: If document not found or database error
    """
    logger.info(f"Feedback submission: {request.rating.value} for document {request.document_id} page {request.page}")

    pg_client = get_postgres_client()

    # Verify document exists
    document = pg_client.get_document(request.document_id)
    if not document:
        logger.warning(f"Feedback rejected: document not found {request.document_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {request.document_id}"
        )

    # Create feedback record
    feedback_id = str(uuid.uuid4())

    try:
        feedback = pg_client.create_feedback(
            feedback_id=feedback_id,
            query=request.query,
            document_id=request.document_id,
            page=request.page,
            rating=request.rating.value,
            session_id=request.session_id
        )

        logger.info(f"Feedback created: {feedback_id}")

        # Invalidate search service cache for this document page
        search_service = get_search_service()
        search_service.invalidate_feedback_cache(request.document_id, request.page)

        return FeedbackResponse(
            feedback_id=feedback.id,
            query=feedback.query,
            document_id=feedback.document_id,
            page=feedback.page,
            rating=feedback.rating,
            timestamp=feedback.timestamp,
            message="Feedback submitted successfully"
        )

    except Exception as e:
        logger.error(f"Failed to create feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get(
    "/stats/{document_id}/{page}",
    response_model=FeedbackStats,
    summary="Get feedback statistics for a document page",
    description="Retrieve aggregated feedback statistics (admin endpoint)"
)
async def get_feedback_stats(document_id: str, page: int) -> FeedbackStats:
    """
    Get feedback statistics for a specific document page.

    Args:
        document_id: Document identifier
        page: Page number

    Returns:
        FeedbackStats: Aggregated feedback statistics

    Raises:
        HTTPException: If document not found or database error
    """
    logger.info(f"Fetching feedback stats for document {document_id} page {page}")

    pg_client = get_postgres_client()

    # Verify document exists
    document = pg_client.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )

    try:
        stats = pg_client.get_feedback_stats(document_id, page)

        # Calculate boost score
        positive_count = stats["positive_count"]
        negative_count = stats["negative_count"]
        net_votes = positive_count - negative_count
        boost_score = 1.0 + (net_votes * 0.1)

        # Clamp boost score to reasonable range (0.1 to 3.0)
        boost_score = max(0.1, min(3.0, boost_score))

        return FeedbackStats(
            document_id=document_id,
            page=page,
            positive_count=positive_count,
            negative_count=negative_count,
            total_count=stats["total_count"],
            boost_score=boost_score
        )

    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve feedback statistics: {str(e)}"
        )
