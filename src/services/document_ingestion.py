"""
Document ingestion service for semantic search.

This service handles the complete pipeline:
1. Extract text from document
2. Generate summary_short and summary_long
3. Compute embedding from summary
4. Index into semantic search index
"""

from pathlib import Path
from typing import Optional, Dict, Any
import uuid

from src.services.multi_format_processor import get_multi_format_processor
from src.db.semantic_index import get_semantic_search_client
from src.utils.logging import get_logger
from src.config import settings, is_configured_secret
import anthropic

logger = get_logger(__name__)


class SemanticDocumentIngestion:
    """
    Document ingestion pipeline for semantic search.

    Converts raw documents into semantically searchable records:
    - Extracts full text
    - Generates short and long summaries
    - Indexes with embeddings
    """

    def __init__(self):
        """Initialize ingestion service."""
        self.processor = get_multi_format_processor()
        self.search_client = get_semantic_search_client()
        self.anthropic_client = None
        if is_configured_secret(settings.anthropic_api_key):
            self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate_summaries(
        self,
        full_text: str,
        title: str
    ) -> tuple[str, str]:
        """
        Generate both short and long summaries from document text.

        Args:
            full_text: Complete document text
            title: Document title

        Returns:
            tuple: (summary_short, summary_long)
                   summary_short: 1-3 sentences
                   summary_long: 1-2 paragraphs
        """
        # Check if AI is available
        if not is_configured_secret(settings.anthropic_api_key) or self.anthropic_client is None:
            # Fallback: extract summaries from text
            logger.warning(f"No AI key available, using text excerpts for {title}")

            summary_short = full_text[:200] + "..." if len(full_text) > 200 else full_text
            summary_long = full_text[:1000] + "..." if len(full_text) > 1000 else full_text

            return summary_short, summary_long

        try:
            # Truncate very long content
            max_content = 50000
            if len(full_text) > max_content:
                full_text = full_text[:max_content] + "... [truncated]"

            prompt = f"""You are analyzing a financial, tax, or accounting document titled "{title}".

Generate TWO summaries:

1. SHORT SUMMARY (1-3 sentences):
   - Extremely concise overview
   - Capture the core purpose
   - Use clear, searchable language

2. LONG SUMMARY (1-2 paragraphs):
   - Comprehensive overview
   - Include key topics, concepts, and regulatory/accounting details
   - Mention important identifiers, jurisdictions, tax years, and entity types when present
   - Use terminology that would help someone find this document
   - Be detailed enough to answer questions about the content

Document content:
{full_text}

Provide your response in this exact format:

SHORT:
<your short summary here>

LONG:
<your long summary here>"""

            message = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Parse response
            summary_short = ""
            summary_long = ""

            if "SHORT:" in response_text and "LONG:" in response_text:
                parts = response_text.split("LONG:")
                summary_short = parts[0].replace("SHORT:", "").strip()
                summary_long = parts[1].strip()
            else:
                # Fallback: use first paragraph as short, rest as long
                paragraphs = response_text.split("\n\n")
                summary_short = paragraphs[0] if paragraphs else response_text[:200]
                summary_long = response_text

            logger.info(
                f"Generated summaries for '{title}': "
                f"short={len(summary_short)} chars, long={len(summary_long)} chars"
            )

            return summary_short, summary_long

        except Exception as e:
            logger.error(f"Summary generation failed for '{title}': {e}")

            # Fallback to simple excerpts
            summary_short = full_text[:200] + "..." if len(full_text) > 200 else full_text
            summary_long = full_text[:1000] + "..." if len(full_text) > 1000 else full_text

            return summary_short, summary_long

    def ingest_document(
        self,
        file_path: Path,
        document_id: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the semantic search index.

        Pipeline:
        1. Extract text from file
        2. Generate summaries (short + long)
        3. Compute embedding
        4. Index document

        Args:
            file_path: Path to document file
            document_id: Optional document ID (generated if None)
            category: Document category
            metadata: Additional metadata (product, version, locale, tags, etc.)

        Returns:
            dict: Ingestion result with status and document info
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        logger.info(f"Ingesting document: {file_path.name} (ID: {document_id})")

        result = {
            "document_id": document_id,
            "filename": file_path.name,
            "status": "processing",
            "error": None
        }

        try:
            # Step 1: Extract text
            logger.info(f"Extracting text from {file_path.name}")
            full_text = self.processor.extract_text_from_file(file_path)

            if not full_text or full_text.startswith("Error"):
                raise ValueError(f"Failed to extract text: {full_text}")

            # Use filename as title if not in metadata
            title = metadata.get("title", file_path.stem)

            # Step 2: Generate summaries
            logger.info(f"Generating summaries for {file_path.name}")
            summary_short, summary_long = self.generate_summaries(full_text, title)

            # Step 3: Index with embedding
            logger.info(f"Indexing {file_path.name} with embeddings")
            success = self.search_client.index_document(
                document_id=document_id,
                title=title,
                summary_short=summary_short,
                summary_long=summary_long,
                full_text=full_text,
                metadata=metadata,
                category=category,
                file_type=self.processor.detect_file_type(file_path),
                filename=file_path.name
            )

            if not success:
                raise RuntimeError("Failed to index document")

            result["status"] = "success"
            result["title"] = title
            result["summary_short"] = summary_short
            result["summary_long"] = summary_long
            result["text_length"] = len(full_text)

            logger.info(f"Successfully ingested {file_path.name}")
            return result

        except Exception as e:
            logger.error(f"Error ingesting document {file_path.name}: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            return result

    def bulk_ingest(
        self,
        file_paths: list[Path],
        metadata_map: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents in batch.

        Args:
            file_paths: List of file paths to ingest
            metadata_map: Optional dict mapping filename to metadata

        Returns:
            dict: Batch ingestion statistics
        """
        metadata_map = metadata_map or {}

        results = {
            "total": len(file_paths),
            "success": 0,
            "failed": 0,
            "errors": []
        }

        for file_path in file_paths:
            metadata = metadata_map.get(file_path.name, {})

            result = self.ingest_document(
                file_path=file_path,
                metadata=metadata
            )

            if result["status"] == "success":
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "filename": file_path.name,
                    "error": result.get("error", "Unknown error")
                })

        logger.info(
            f"Bulk ingestion complete: {results['success']}/{results['total']} succeeded"
        )

        return results


# Singleton instance
_ingestion_service: Optional[SemanticDocumentIngestion] = None


def get_ingestion_service() -> SemanticDocumentIngestion:
    """
    Get the global document ingestion service.

    Returns:
        SemanticDocumentIngestion: The ingestion service
    """
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = SemanticDocumentIngestion()
    return _ingestion_service
