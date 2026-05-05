"""
Document management API endpoints.
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
    Depends,
    Query,
)
from fastapi.responses import FileResponse

from src.models.document import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentListResponse,
    DocumentCategory,
    FinancialMetadata,
    EntityType,
    ProcessingStatus,
    DocumentMetadata,
)
from src.db.postgres import get_postgres_client
from src.db.elasticsearch_client import get_elasticsearch_client
from src.services.document_indexing_service import DocumentIndexingService
from src.config import settings
from src.utils.auth import verify_api_key
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def validate_pdf_file(file: UploadFile) -> None:
    """Validate that upload is a PDF."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed",
        )

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {file.content_type}. Expected application/pdf",
        )


async def save_uploaded_file(
    file: UploadFile, document_id: str, storage_path: Path
) -> tuple[Path, int]:
    """Save uploaded file and return its path and size."""
    storage_path.mkdir(parents=True, exist_ok=True)

    file_extension = Path(file.filename).suffix
    file_path = storage_path / f"{document_id}{file_extension}"

    file_size = 0
    max_size = settings.max_file_size_mb * 1024 * 1024

    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                file_size += len(chunk)

                if file_size > max_size:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB",
                    )

                f.write(chunk)

        logger.info(f"Saved file: {file_path} ({file_size} bytes)")
        return file_path, file_size

    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(...),
    machine_model: Optional[str] = Form(None),
    jurisdiction: Optional[str] = Form(None),
    tax_year: Optional[int] = Form(None),
    client_name: Optional[str] = Form(None),
    entity_type: Optional[str] = Form(None),
    source_name: Optional[str] = Form(None),
    section_reference: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key),
) -> DocumentUploadResponse:
    """Upload and index a PDF file."""
    validate_pdf_file(file)

    try:
        doc_category = DocumentCategory(category.lower().replace(" ", "_"))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category. Must be one of: {[c.value for c in DocumentCategory]}",
        )

    normalized_entity_type = None
    if entity_type:
        try:
            normalized_entity_type = EntityType(
                entity_type.lower().replace(" ", "_")
            ).value
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid entity_type. Must be one of: {[e.value for e in EntityType]}",
            )

    document_id = str(uuid.uuid4())

    storage_path = Path(settings.pdf_storage_path)
    file_path, file_size = await save_uploaded_file(file, document_id, storage_path)

    pg_client = get_postgres_client()

    try:
        pg_client.create_document(
            document_id=document_id,
            filename=file_path.name,  # Storage filename (UUID.pdf)
            original_filename=file.filename,  # Original uploaded filename
            file_path=str(file_path),
            file_size=file_size,
            category=doc_category,
            machine_model=machine_model,
            jurisdiction=jurisdiction,
            tax_year=tax_year,
            client_name=client_name,
            entity_type=normalized_entity_type,
            source_name=source_name,
            section_reference=section_reference,
        )
    except Exception as e:
        file_path.unlink(missing_ok=True)
        logger.error(f"Failed to create document record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document record: {str(e)}",
        )

    indexing_service = DocumentIndexingService()
    es_client = get_elasticsearch_client()

    metadata_payload = {
        "document_type": doc_category.value,
        "jurisdiction": jurisdiction,
        "tax_year": tax_year,
        "client_name": client_name,
        "entity_type": normalized_entity_type,
        "source_name": source_name or file.filename,
        "section_reference": section_reference,
    }

    try:
        pg_client.update_document_status(document_id, ProcessingStatus.PARSING)
        try:
            prepared = indexing_service.prepare_pdf_document(
                file_path=file_path,
                document_id=document_id,
                source_filename=file.filename,
                category=doc_category.value,
                corpus_type="uploaded",
                metadata=metadata_payload,
            )
        except TypeError:
            prepared = indexing_service.prepare_pdf_document(
                file_path=file_path,
                document_id=document_id,
                source_filename=file.filename,
                category=doc_category.value,
                metadata=metadata_payload,
            )

        pg_client.update_document_status(
            document_id=document_id,
            status=ProcessingStatus.INDEXING,
            total_pages=prepared["total_pages"],
        )

        indexing_service.add_embeddings(prepared["documents"])
        index_result = indexing_service.index_documents(es_client, prepared["documents"])

        if index_result["error_count"] > 0:
            raise RuntimeError(
                f"Indexing errors encountered: {index_result['error_count']}"
            )

        indexed_at = datetime.utcnow()
        pg_client.update_document_status(
            document_id=document_id,
            status=ProcessingStatus.READY,
            total_pages=prepared["total_pages"],
            indexed_at=indexed_at,
            error_message=None,
        )

        logger.info(
            f"Document {document_id} indexed successfully: "
            f"{prepared['total_pages']} pages, {prepared['total_chunks']} chunks"
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status=ProcessingStatus.READY,
            upload_date=datetime.utcnow(),
            number_of_pages=prepared["total_pages"],
            number_of_chunks=prepared["total_chunks"],
            metadata=FinancialMetadata(
                document_type=doc_category.value,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                client_name=client_name,
                entity_type=normalized_entity_type,
                source_name=source_name or file.filename,
                section_reference=section_reference,
            ),
            indexed_at=indexed_at,
            message="Document uploaded and indexed successfully",
        )
    except Exception as e:
        logger.error(f"Document indexing failed for {document_id}: {e}")
        pg_client.update_document_status(
            document_id=document_id,
            status=ProcessingStatus.FAILED,
            error_message=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str, api_key: str = Depends(verify_api_key)
) -> DocumentStatusResponse:
    """
    Get document metadata and processing status.

    Args:
        document_id: Document ID
        api_key: API key for authentication

    Returns:
        DocumentStatusResponse: Document status information

    Raises:
        HTTPException: If document not found
    """
    pg_client = get_postgres_client()
    doc = pg_client.get_document(document_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    return DocumentStatusResponse(
        document_id=doc.id,
        filename=doc.original_filename,  # Use original filename for display
        status=doc.processing_status,
        upload_date=doc.upload_date,
        indexed_at=doc.indexed_at,
        total_pages=doc.total_pages,
        error_message=doc.error_message,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    doc_status: Optional[str] = Query(None, alias="status"),
    category: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    api_key: str = Depends(verify_api_key),
) -> DocumentListResponse:
    """
    List documents with optional filters and pagination.

    Args:
        doc_status: Optional status filter
        category: Optional category filter
        page: Page number (1-indexed)
        page_size: Items per page (max 100)
        api_key: API key for authentication

    Returns:
        DocumentListResponse: Paginated list of documents

    Raises:
        HTTPException: If validation fails
    """
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Page must be >= 1"
        )

    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be between 1 and 100",
        )

    status_filter = None
    if doc_status:
        try:
            status_filter = ProcessingStatus(doc_status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.value for s in ProcessingStatus]}",
            )

    category_filter = None
    if category:
        try:
            category_filter = DocumentCategory(category)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category. Must be one of: {[c.value for c in DocumentCategory]}",
            )

    pg_client = get_postgres_client()
    offset = (page - 1) * page_size

    docs = pg_client.list_documents(
        status=status_filter, category=category_filter, limit=page_size, offset=offset
    )

    total = pg_client.count_documents(status=status_filter, category=category_filter)

    document_list = []
    for doc in docs:
        document_list.append(
            DocumentMetadata(
                document_id=doc.id,
                filename=doc.original_filename,  # Use original filename for display
                file_size=doc.file_size,
                file_path=doc.file_path,
                category=doc.category or DocumentCategory.OTHER.value,
                machine_model=doc.machine_model,
                part_numbers=[],
                financial_metadata=FinancialMetadata(
                    document_type=doc.category or DocumentCategory.OTHER.value,
                    jurisdiction=doc.jurisdiction,
                    tax_year=doc.tax_year,
                    client_name=doc.client_name,
                    entity_type=doc.entity_type,
                    source_name=doc.source_name,
                    section_reference=doc.section_reference,
                ),
                upload_date=doc.upload_date,
                processing_status=doc.processing_status,
                indexed_at=doc.indexed_at,
                error_message=doc.error_message,
                total_pages=doc.total_pages,
            )
        )

    return DocumentListResponse(
        total=total, page=page, page_size=page_size, documents=document_list
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str, api_key: str = Depends(verify_api_key)
) -> None:
    """
    Delete a document and all associated data.

    Args:
        document_id: Document ID
        api_key: API key for authentication

    Raises:
        HTTPException: If document not found or deletion fails
    """
    pg_client = get_postgres_client()
    es_client = get_elasticsearch_client()

    doc = pg_client.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    try:
        should_filters = [{"term": {"document_id": document_id}}]
        if doc.original_filename:
            should_filters.append({"term": {"source_filename": doc.original_filename}})
            should_filters.append({"term": {"filename": doc.original_filename}})

        es_client.es.delete_by_query(
            index=es_client.index_name,
            body={
                "query": {
                    "bool": {
                        "should": should_filters,
                        "minimum_should_match": 1,
                    }
                }
            },
            conflicts="proceed",
            refresh=True,
        )
        logger.info(f"Deleted Elasticsearch chunks for document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to delete from Elasticsearch: {e}")

    try:
        file_path = Path(doc.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to delete file: {e}")

    pg_client.delete_document(document_id)

    logger.info(f"Document {document_id} deleted successfully")


@router.get("/{document_id}/download")
async def download_document(
    document_id: str, api_key: str = Depends(verify_api_key)
) -> FileResponse:
    """
    Download the original PDF document.

    Args:
        document_id: Document ID
        api_key: API key for authentication

    Returns:
        FileResponse: PDF file download

    Raises:
        HTTPException: If document not found or file missing
    """
    pg_client = get_postgres_client()

    doc = pg_client.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found for document {document_id}",
        )

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=doc.filename,
    )


@router.get("/{document_id}/view")
async def view_document(
    document_id: str
) -> FileResponse:
    """View PDF in browser (used by the UI source links)."""
    pg_client = get_postgres_client()

    doc = pg_client.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found for document {document_id}",
        )

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=doc.filename,
        content_disposition_type="inline"
    )
