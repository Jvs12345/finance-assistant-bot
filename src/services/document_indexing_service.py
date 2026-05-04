"""
Shared PDF indexing service used by both batch indexing and API uploads.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from elasticsearch import helpers

from src.services.embedding_service import EmbeddingService, EmbeddingProvider
from src.services.multi_format_processor import MultiFormatProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentIndexingService:
    """
    Build and index PDF chunks in a format compatible with the main retrieval path.
    """

    def __init__(self, chunk_size: int = 10000, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.processor = MultiFormatProcessor()
        self.embedding_service = EmbeddingService(
            provider=EmbeddingProvider.LOCAL,
            model="all-MiniLM-L6-v2",
        )

    def prepare_pdf_document(
        self,
        file_path: Path,
        document_id: str,
        source_filename: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract pages and build chunk documents for Elasticsearch.
        """
        metadata = metadata or {}
        pages = self.processor.extract_pages_from_pdf(file_path)
        if not pages:
            raise ValueError(f"No readable pages extracted from {file_path.name}")

        inferred_tax_year = metadata.get("tax_year") or self._infer_tax_year(source_filename, pages)
        inferred_jurisdiction = metadata.get("jurisdiction") or self._infer_jurisdiction(source_filename, pages)
        resolved_category = self._resolve_document_type(category, source_filename, pages, metadata)

        metadata["tax_year"] = inferred_tax_year
        metadata["jurisdiction"] = inferred_jurisdiction
        metadata["document_type"] = resolved_category

        upload_date = datetime.utcnow().isoformat()
        file_size = file_path.stat().st_size
        documents: List[Dict[str, Any]] = []

        for page_entry in pages:
            page_number = int(page_entry["page"])
            page_content = page_entry["content"]

            chunks = self._chunk_content(page_content)
            for chunk_index, chunk_text in enumerate(chunks, 1):
                chunk_id = f"{document_id}-p{page_number}-c{chunk_index}"
                merged_metadata = {
                    **metadata,
                    "source_filename": source_filename,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                }
                doc = {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "title": f"{Path(source_filename).stem} - Page {page_number}",
                    "filename": source_filename,
                    "source_filename": source_filename,
                    "content": chunk_text,
                    "excerpt": chunk_text[:500],
                    "summary": None,
                    "category": resolved_category,
                    "file_type": "pdf",
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "upload_date": upload_date,
                    "indexed_at": datetime.utcnow().isoformat(),
                    "jurisdiction": metadata.get("jurisdiction"),
                    "tax_year": metadata.get("tax_year"),
                    "entity_type": metadata.get("entity_type"),
                    "client_name": metadata.get("client_name"),
                    "source_name": metadata.get("source_name") or source_filename,
                    "section_reference": metadata.get("section_reference"),
                    "metadata": merged_metadata,
                }
                documents.append(doc)

        return {
            "document_id": document_id,
            "source_filename": source_filename,
            "total_pages": len(pages),
            "total_chunks": len(documents),
            "documents": documents,
            "metadata": metadata,
        }

    def _resolve_document_type(
        self,
        category: str,
        source_filename: str,
        pages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:
        provided = (metadata.get("document_type") or category or "other").strip().lower()
        if provided and provided != "other":
            return provided

        text_blob = f"{source_filename} " + " ".join(
            p.get("content", "")[:1500] for p in pages[:3]
        )
        normalized = text_blob.lower()
        mapping = [
            ("tax_law", ["tax law", "tax code", "belastingwet", "internal revenue code"]),
            ("regulation", ["regulation", "directive", "compliance rule", "ifrs", "gaap"]),
            ("annual_report", ["annual report", "form 10-k", "jaarverslag"]),
            ("invoice", ["invoice", "factuur", "bill to"]),
            ("ledger", ["ledger", "general ledger", "trial balance", "grootboek"]),
            ("bank_statement", ["bank statement", "account statement", "iban"]),
            ("tax_return", ["tax return", "income tax return", "vat return", "belastingaangifte"]),
            ("correspondence", ["dear", "regards", "sincerely", "letter", "email"]),
        ]
        for doc_type, keywords in mapping:
            if any(k in normalized for k in keywords):
                return doc_type
        return "other"

    def _infer_tax_year(self, source_filename: str, pages: List[Dict[str, Any]]) -> Optional[int]:
        candidates: List[int] = []
        filename_years = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", source_filename)
        candidates.extend(int(y) for y in filename_years)
        for page in pages[:3]:
            years = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", page.get("content", ""))
            candidates.extend(int(y) for y in years)
        valid = [y for y in candidates if 1900 <= y <= 2100]
        if not valid:
            return None
        # Prefer the most recent explicit year.
        return max(valid)

    def _infer_jurisdiction(self, source_filename: str, pages: List[Dict[str, Any]]) -> Optional[str]:
        text_blob = f"{source_filename} " + " ".join(
            p.get("content", "")[:2000] for p in pages[:3]
        )
        t = text_blob.lower()
        mapping = [
            ("Netherlands", ["netherlands", "dutch", "nederland", "belastingdienst"]),
            ("EU", ["european union", "eu directive", "eu regulation"]),
            ("Germany", ["germany", "deutschland", "bundesfinanzministerium"]),
            ("UK", ["united kingdom", "uk", "hmrc", "companies house"]),
            ("United States", ["united states", "u.s.", "usa", "irs", "sec", "form 10-k"]),
        ]
        for jurisdiction, keywords in mapping:
            if any(k in t for k in keywords):
                return jurisdiction
        return None

    def add_embeddings(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> None:
        """
        Generate embeddings in-place for the provided chunk documents.
        """
        if not documents:
            return

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts: List[str] = []
            for doc in batch:
                text = doc.get("content", "")
                if len(text) > 1000:
                    text = f"{doc.get('title', '')}: {text[:1000]}"
                texts.append(text)

            embeddings = self.embedding_service.get_embeddings_batch(texts)
            for j, doc in enumerate(batch):
                doc["embedding"] = embeddings[j]

    def index_documents(
        self,
        es_client: Any,
        documents: List[Dict[str, Any]],
        chunk_size: int = 500,
        refresh: bool = True,
    ) -> Dict[str, Any]:
        """
        Bulk index prepared chunk documents into Elasticsearch.
        """
        if not documents:
            return {"success_count": 0, "error_count": 0, "errors": []}

        actions = []
        for doc in documents:
            source_doc = dict(doc)
            source_doc.pop("id", None)
            actions.append(
                {
                    "_index": es_client.index_name,
                    "_id": doc.get("chunk_id") or doc.get("id"),
                    "_source": source_doc,
                }
            )

        success, errors = helpers.bulk(
            es_client.es,
            actions,
            chunk_size=chunk_size,
            request_timeout=120,
            raise_on_error=False,
            raise_on_exception=False,
            max_retries=3,
            initial_backoff=2,
            max_backoff=60,
        )

        if refresh:
            es_client.es.indices.refresh(index=es_client.index_name)

        error_count = len(errors) if isinstance(errors, list) else 0
        if error_count:
            logger.warning(f"Bulk indexing completed with {error_count} errors")

        return {
            "success_count": success,
            "error_count": error_count,
            "errors": errors if error_count else [],
        }

    def _chunk_content(self, content: str) -> List[str]:
        """
        Split text into overlapping chunks.
        """
        if len(content) <= self.chunk_size:
            return [content]

        chunks: List[str] = []
        start = 0
        while start < len(content):
            end = start + self.chunk_size
            chunks.append(content[start:end])
            if end >= len(content):
                break
            start = max(0, end - self.overlap)
        return chunks
