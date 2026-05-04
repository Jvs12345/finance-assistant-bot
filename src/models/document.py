"""
Pydantic models for financial document management.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Document processing status enumeration."""
    UPLOADED = "uploaded"
    PARSING = "parsing"
    SUMMARIZING = "summarizing"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"
    FAILED = "failed"


class DocumentCategory(str, Enum):
    """Financial document category enumeration."""
    TAX_LAW = "tax_law"
    REGULATION = "regulation"
    ANNUAL_REPORT = "annual_report"
    INVOICE = "invoice"
    LEDGER = "ledger"
    BANK_STATEMENT = "bank_statement"
    TAX_RETURN = "tax_return"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"


class EntityType(str, Enum):
    """Entity type classification for tax/accounting context."""
    INDIVIDUAL = "individual"
    SOLE_TRADER = "sole_trader"
    BV = "bv"
    NV = "nv"
    FOUNDATION = "foundation"
    PARTNERSHIP = "partnership"
    OTHER = "other"


class FinancialMetadata(BaseModel):
    """Metadata used for retrieval and response context."""
    document_type: Optional[DocumentCategory] = None
    jurisdiction: Optional[str] = None
    tax_year: Optional[int] = None
    client_name: Optional[str] = None
    entity_type: Optional[EntityType] = None
    source_name: Optional[str] = None
    section_reference: Optional[str] = None

    @validator("tax_year")
    def validate_tax_year(cls, v):
        """Allow realistic year bounds only."""
        if v is None:
            return v
        if v < 1900 or v > 2100:
            raise ValueError("tax_year must be between 1900 and 2100")
        return v

    class Config:
        use_enum_values = True


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    category: DocumentCategory
    machine_model: Optional[str] = None
    part_numbers: Optional[List[str]] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    tax_year: Optional[int] = None
    client_name: Optional[str] = None
    entity_type: Optional[EntityType] = None
    source_name: Optional[str] = None
    section_reference: Optional[str] = None

    @validator("category", pre=True)
    def validate_category(cls, v):
        """Validate and normalize category."""
        if isinstance(v, str):
            # Convert to lowercase and handle underscores
            v = v.lower().replace(" ", "_")
            if v not in [cat.value for cat in DocumentCategory]:
                raise ValueError(
                    f"Category must be one of: {[cat.value for cat in DocumentCategory]}"
                )
        return v

    @validator("tax_year")
    def validate_upload_tax_year(cls, v):
        """Validate tax year in upload payload."""
        if v is None:
            return v
        if v < 1900 or v > 2100:
            raise ValueError("tax_year must be between 1900 and 2100")
        return v

    class Config:
        use_enum_values = True


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    status: ProcessingStatus
    upload_date: datetime
    number_of_pages: Optional[int] = None
    number_of_chunks: Optional[int] = None
    metadata: Optional[FinancialMetadata] = None
    indexed_at: Optional[datetime] = None
    message: str = "Document uploaded and indexed successfully"

    class Config:
        use_enum_values = True


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    document_id: str
    filename: str
    file_size: int  # Bytes
    file_path: str
    category: str
    machine_model: Optional[str] = None
    part_numbers: List[str] = Field(default_factory=list)
    financial_metadata: FinancialMetadata = Field(default_factory=FinancialMetadata)
    upload_date: datetime
    processing_status: ProcessingStatus
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    total_pages: Optional[int] = None

    class Config:
        use_enum_values = True
        from_attributes = True  # For SQLAlchemy models


class DocumentStatusResponse(BaseModel):
    """Response model for document status check."""
    document_id: str
    filename: str
    status: ProcessingStatus
    upload_date: datetime
    indexed_at: Optional[datetime] = None
    total_pages: Optional[int] = None
    error_message: Optional[str] = None

    class Config:
        use_enum_values = True


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    total: int
    page: int
    page_size: int
    documents: List[DocumentMetadata]


class DocumentPage(BaseModel):
    """Model for a document page chunk."""
    document_id: str
    filename: str
    page: int
    content: str
    summary: Optional[str] = None
    category: DocumentCategory
    machine_model: Optional[str] = None
    part_numbers: List[str] = Field(default_factory=list)
    financial_metadata: FinancialMetadata = Field(default_factory=FinancialMetadata)
    upload_date: datetime
    indexed_at: Optional[datetime] = None
    file_size: int
    file_path: str
    processing_status: ProcessingStatus = ProcessingStatus.READY

    class Config:
        use_enum_values = True


class ProcessingProgress(BaseModel):
    """Model for tracking document processing progress."""
    document_id: str
    current_stage: ProcessingStatus
    total_pages: Optional[int] = None
    pages_processed: int = 0
    started_at: datetime
    error_message: Optional[str] = None

    class Config:
        use_enum_values = True
