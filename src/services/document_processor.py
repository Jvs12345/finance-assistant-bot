"""
Document processing pipeline orchestrator.
Coordinates PDF parsing, chunking, summarization, and indexing.
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.models.document import ProcessingStatus, DocumentCategory, DocumentPage
from src.services.pdf_parser import get_pdf_parser
from src.services.markdown_chunker import get_markdown_chunker
from src.services.summarizer import get_summarizer
from src.db.elasticsearch import get_elasticsearch_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Orchestrates the document processing pipeline."""

    def __init__(self):
        """Initialize document processor with service dependencies."""
        self.pdf_parser = get_pdf_parser()
        self.chunker = get_markdown_chunker()
        self.summarizer = get_summarizer()
        self.es_client = get_elasticsearch_client()

    def process_document(
        self,
        file_path: Path,
        document_id: str,
        original_filename: str,
        category: DocumentCategory,
        machine_model: Optional[str] = None,
        financial_metadata: Optional[Dict[str, Any]] = None,
        generate_summaries: bool = True
    ) -> Dict[str, Any]:
        """
        Process a PDF document through the complete pipeline.

        Pipeline stages:
        1. Parse PDF to markdown
        2. Chunk markdown by page
        3. Generate summaries for each page (optional)
        4. Index pages in Elasticsearch

        Args:
            file_path: Path to PDF file
            document_id: Unique document identifier
            original_filename: Original uploaded filename (for display)
            category: Document category
            machine_model: Optional legacy machine model
            financial_metadata: Optional financial/tax metadata for retrieval filtering
            generate_summaries: Whether to generate summaries (default: True)

        Returns:
            dict: Processing results with stats

        Raises:
            Exception: If processing fails
        """
        logger.info(f"Starting document processing pipeline for {document_id}")

        result = {
            "document_id": document_id,
            "filename": original_filename,
            "status": ProcessingStatus.UPLOADED,
            "total_pages": 0,
            "pages_indexed": 0,
            "summaries_generated": 0,
            "error_message": None,
            "started_at": datetime.utcnow(),
            "completed_at": None
        }

        try:
            financial_metadata = financial_metadata or {}

            # Stage 1: Parse PDF to markdown
            logger.info(f"[{document_id}] Stage 1: Parsing PDF")
            result["status"] = ProcessingStatus.PARSING
            markdown_content = self.pdf_parser.parse_pdf_with_retry(file_path)

            # Stage 2: Chunk markdown by page
            logger.info(f"[{document_id}] Stage 2: Chunking markdown")
            page_chunks = self.chunker.chunk_by_page(markdown_content)
            result["total_pages"] = len(page_chunks)

            logger.info(
                f"[{document_id}] Extracted {len(page_chunks)} pages from PDF"
            )

            # Stage 3: Generate summaries (optional)
            summaries = []
            if generate_summaries:
                logger.info(f"[{document_id}] Stage 3: Generating summaries")
                result["status"] = ProcessingStatus.SUMMARIZING

                for chunk in page_chunks:
                    try:
                        summary = self.summarizer.summarize_text_with_retry(
                            chunk["content"]
                        )
                        summaries.append(summary)
                        result["summaries_generated"] += 1

                    except Exception as e:
                        logger.warning(
                            f"[{document_id}] Failed to summarize page {chunk['page']}: {e}"
                        )
                        # Add empty summary on failure
                        summaries.append("")

            else:
                logger.info(f"[{document_id}] Skipping summary generation")
                summaries = [""] * len(page_chunks)

            # Stage 4: Index pages in Elasticsearch
            logger.info(f"[{document_id}] Stage 4: Indexing in Elasticsearch")
            result["status"] = ProcessingStatus.INDEXING

            upload_date = datetime.utcnow()
            file_size = file_path.stat().st_size

            # Prepare documents for bulk indexing
            documents = []
            for i, chunk in enumerate(page_chunks):
                # Extract metadata from chunk
                metadata = self.chunker.extract_metadata(chunk["content"])
                # Merge extraction metadata with user-provided financial metadata.
                merged_metadata = {
                    **metadata,
                    "document_type": category.value if isinstance(category, DocumentCategory) else category,
                    "jurisdiction": financial_metadata.get("jurisdiction"),
                    "tax_year": financial_metadata.get("tax_year"),
                    "client_name": financial_metadata.get("client_name"),
                    "entity_type": financial_metadata.get("entity_type"),
                    "source_name": financial_metadata.get("source_name") or original_filename,
                    "section_reference": financial_metadata.get("section_reference"),
                }

                doc = {
                    "document_id": document_id,
                    "filename": original_filename,  # Use original filename for display
                    "page": chunk["page"],
                    "content": chunk["content"],
                    "summary": summaries[i] if summaries[i] else None,
                    "category": category.value if isinstance(category, DocumentCategory) else category,
                    "machine_model": machine_model,
                    "part_numbers": metadata.get("part_numbers", []),
                    "jurisdiction": financial_metadata.get("jurisdiction"),
                    "tax_year": financial_metadata.get("tax_year"),
                    "client_name": financial_metadata.get("client_name"),
                    "entity_type": financial_metadata.get("entity_type"),
                    "source_name": financial_metadata.get("source_name") or original_filename,
                    "section_reference": financial_metadata.get("section_reference"),
                    "upload_date": upload_date.isoformat(),
                    "indexed_at": datetime.utcnow().isoformat(),
                    "file_size": file_size,
                    "file_path": str(file_path),
                    "metadata": merged_metadata,
                    "processing_status": ProcessingStatus.READY.value
                }

                documents.append(doc)

            # Bulk index all pages
            success, errors = self.es_client.bulk_index(
                index_name="documents",
                documents=documents
            )

            result["pages_indexed"] = success

            if errors:
                logger.warning(
                    f"[{document_id}] {len(errors)} pages failed to index"
                )

            # Mark as complete
            result["status"] = ProcessingStatus.READY
            result["completed_at"] = datetime.utcnow()

            logger.info(
                f"[{document_id}] Processing complete: "
                f"{result['pages_indexed']}/{result['total_pages']} pages indexed, "
                f"{result['summaries_generated']} summaries generated"
            )

            return result

        except Exception as e:
            logger.error(f"[{document_id}] Processing failed: {e}")
            result["status"] = ProcessingStatus.FAILED
            result["error_message"] = str(e)
            result["completed_at"] = datetime.utcnow()
            raise

    def reprocess_document(
        self,
        document_id: str,
        regenerate_summaries: bool = False
    ) -> Dict[str, Any]:
        """
        Reprocess an existing document (e.g., to regenerate summaries).

        Args:
            document_id: Document ID to reprocess
            regenerate_summaries: Whether to regenerate summaries

        Returns:
            dict: Processing results

        Raises:
            ValueError: If document not found
        """
        # Query existing document pages
        query = {
            "query": {
                "term": {
                    "document_id": document_id
                }
            },
            "size": 1000
        }

        results = self.es_client.search(index_name="documents", query=query)
        pages = results["hits"]["hits"]

        if not pages:
            raise ValueError(f"Document {document_id} not found")

        logger.info(
            f"Reprocessing document {document_id} ({len(pages)} pages)"
        )

        # If regenerating summaries, process each page
        if regenerate_summaries:
            for page in pages:
                try:
                    content = page["_source"]["content"]
                    summary = self.summarizer.summarize_text_with_retry(content)

                    # Update document with new summary
                    self.es_client.client.update(
                        index="documents",
                        id=page["_id"],
                        body={"doc": {"summary": summary}}
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to regenerate summary for page {page['_source']['page']}: {e}"
                    )

        return {
            "document_id": document_id,
            "pages_processed": len(pages),
            "summaries_regenerated": regenerate_summaries
        }

    def delete_document(self, document_id: str) -> int:
        """
        Delete all pages of a document from Elasticsearch.

        Args:
            document_id: Document ID to delete

        Returns:
            int: Number of pages deleted
        """
        logger.info(f"Deleting document {document_id} from Elasticsearch")

        # Delete all pages with matching document_id
        query = {
            "query": {
                "term": {
                    "document_id": document_id
                }
            }
        }

        response = self.es_client.client.delete_by_query(
            index="documents",
            body=query
        )

        deleted = response["deleted"]
        logger.info(f"Deleted {deleted} pages for document {document_id}")

        return deleted


# Global document processor instance
_document_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """
    Get the global document processor instance (singleton pattern).

    Returns:
        DocumentProcessor: The document processor instance
    """
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
