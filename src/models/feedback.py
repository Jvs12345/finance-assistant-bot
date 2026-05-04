"""
Feedback request and response models.
"""

from typing import Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class FeedbackRating(str, Enum):
    """Feedback rating values."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query that generated the result"
    )
    document_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Document identifier"
    )
    page: int = Field(
        ...,
        ge=0,
        description="Page number that was rated"
    )
    rating: FeedbackRating = Field(
        ...,
        description="Rating value (positive or negative)"
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=36,
        description="Optional session identifier for anonymous tracking"
    )

    @validator("query")
    def validate_query(cls, v):
        """Validate and clean query string."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace")
        return v


class FeedbackResponse(BaseModel):
    """Response model after submitting feedback."""

    feedback_id: str = Field(..., description="Unique feedback identifier")
    query: str = Field(..., description="Search query")
    document_id: str = Field(..., description="Document identifier")
    page: int = Field(..., description="Page number")
    rating: str = Field(..., description="Rating value")
    timestamp: datetime = Field(..., description="Feedback submission timestamp")
    message: str = Field(..., description="Success message")


class FeedbackStats(BaseModel):
    """Feedback statistics for a document page."""

    document_id: str = Field(..., description="Document identifier")
    page: int = Field(..., description="Page number")
    positive_count: int = Field(..., description="Number of positive ratings")
    negative_count: int = Field(..., description="Number of negative ratings")
    total_count: int = Field(..., description="Total number of ratings")
    boost_score: float = Field(..., description="Calculated boost multiplier")
