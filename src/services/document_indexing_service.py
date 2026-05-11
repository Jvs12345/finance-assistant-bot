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
from src.services.xaf_conversion_service import get_xaf_conversion_service
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentIndexingService:
    """
    Build and index document chunks in a format compatible with the main retrieval path.
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
        corpus_type: str = "uploaded",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract pages and build chunk documents for Elasticsearch.
        """
        metadata = metadata or {}
        resolved_corpus_type = corpus_type if corpus_type in {"existing", "uploaded"} else "uploaded"
        pages = self.processor.extract_pages_from_pdf(file_path)
        if not pages:
            raise ValueError(f"No readable pages extracted from {file_path.name}")

        inferred_tax_year = metadata.get("tax_year") or self._infer_tax_year(source_filename, pages)
        inferred_jurisdiction = metadata.get("jurisdiction") or self._infer_jurisdiction(source_filename, pages)
        resolved_category, detailed_type = self._resolve_document_type(category, source_filename, pages, metadata)

        metadata["tax_year"] = inferred_tax_year
        metadata["jurisdiction"] = inferred_jurisdiction
        metadata["document_type"] = resolved_category
        metadata["document_type_detail"] = detailed_type
        metadata["corpus_type"] = resolved_corpus_type

        upload_date = datetime.utcnow().isoformat()
        file_size = file_path.stat().st_size
        view_filename = metadata.get("view_filename") or source_filename
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
                    "view_filename": view_filename,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                }
                doc = {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "title": f"{Path(source_filename).stem} - Page {page_number}",
                    "filename": source_filename,
                    "view_filename": view_filename,
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
                    "corpus_type": resolved_corpus_type,
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

    def prepare_document(
        self,
        file_path: Path,
        document_id: str,
        source_filename: str,
        category: str,
        corpus_type: str = "uploaded",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare any supported document type for Elasticsearch indexing.
        """
        detected_type = self.processor.detect_file_type(file_path)
        if detected_type == "pdf":
            return self.prepare_pdf_document(
                file_path=file_path,
                document_id=document_id,
                source_filename=source_filename,
                category=category,
                corpus_type=corpus_type,
                metadata=metadata,
            )

        metadata = metadata or {}
        resolved_corpus_type = corpus_type if corpus_type in {"existing", "uploaded"} else "uploaded"

        if detected_type == "xaf":
            view_filename = source_filename
            converted_path = get_xaf_conversion_service().convert_to_text_file(file_path)
            metadata["xaf_converted_path"] = str(converted_path)
            preview_pdf_path = get_xaf_conversion_service().convert_to_pdf_file(file_path)
            if preview_pdf_path:
                metadata["xaf_preview_pdf"] = str(preview_pdf_path)
                view_filename = preview_pdf_path.name
            metadata["view_filename"] = view_filename
            raw_text = converted_path.read_text(encoding="utf-8", errors="ignore")
        else:
            view_filename = source_filename
            raw_text = self.processor.extract_text_from_file(file_path)

        if (
            not raw_text
            or not raw_text.strip()
            or raw_text.startswith("Error reading")
            or raw_text.startswith("Unsupported file type")
            or raw_text.startswith("Binary file (skipped)")
        ):
            raise ValueError(f"No readable content extracted from {file_path.name}")

        synthetic_pages = [{"content": raw_text}]
        inferred_tax_year = metadata.get("tax_year") or self._infer_tax_year(source_filename, synthetic_pages)
        inferred_jurisdiction = metadata.get("jurisdiction") or self._infer_jurisdiction(source_filename, synthetic_pages)
        resolved_category, detailed_type = self._resolve_document_type(category, source_filename, synthetic_pages, metadata)

        metadata["tax_year"] = inferred_tax_year
        metadata["jurisdiction"] = inferred_jurisdiction
        metadata["document_type"] = resolved_category
        metadata["document_type_detail"] = detailed_type
        metadata["corpus_type"] = resolved_corpus_type

        upload_date = datetime.utcnow().isoformat()
        file_size = file_path.stat().st_size
        chunks = self._chunk_content(raw_text)
        documents: List[Dict[str, Any]] = []

        for chunk_index, chunk_text in enumerate(chunks, 1):
            chunk_id = f"{document_id}-c{chunk_index}"
            merged_metadata = {
                **metadata,
                "source_filename": source_filename,
                "view_filename": view_filename,
                "chunk_id": chunk_id,
            }
            documents.append(
                {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "title": f"{Path(source_filename).stem} - Part {chunk_index}",
                    "filename": source_filename,
                    "view_filename": view_filename,
                    "source_filename": source_filename,
                    "content": chunk_text,
                    "excerpt": chunk_text[:500],
                    "summary": None,
                    "category": resolved_category,
                    "file_type": detected_type,
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "page_number": None,
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
                    "corpus_type": resolved_corpus_type,
                    "metadata": merged_metadata,
                }
            )

        return {
            "document_id": document_id,
            "source_filename": source_filename,
            "total_pages": 1,
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
    ) -> tuple[str, str]:
        provided = (metadata.get("document_type") or category or "other").strip().lower()
        if provided and provided != "other":
            return provided, provided

        detailed_type = self._infer_document_type_detail(source_filename, pages)
        category_map = {
            "profit_loss": "other",
            "balance_sheet": "other",
            "vat_overview": "tax_return",
            "client_notes": "correspondence",
            "contract": "correspondence",
            "insurance_document": "other",
            "annual_accounts_checklist": "regulation",
            "accounting_guidance": "regulation",
            "vat_guidance": "regulation",
            "insurance_guidance": "regulation",
            "annual_report": "annual_report",
            "invoice": "invoice",
            "bank_statement": "bank_statement",
            "tax_return": "tax_return",
            "ledger": "ledger",
            "tax_law": "tax_law",
            "regulation": "regulation",
            "correspondence": "correspondence",
            "other": "other",
        }
        return category_map.get(detailed_type, "other"), detailed_type

    def _infer_document_type_detail(self, source_filename: str, pages: List[Dict[str, Any]]) -> str:
        text_blob = f"{source_filename} " + " ".join(
            p.get("content", "")[:2000] for p in pages[:3]
        )
        normalized = text_blob.lower()
        mapping = [
            ("profit_loss", ["winst-en-verliesrekening", "resultatenrekening", "profit and loss", "p&l", "statement of operations"]),
            ("balance_sheet", ["balans", "balance sheet", "statement of financial position"]),
            ("vat_overview", ["btw-overzicht", "btw aangifte", "omzetbelasting", "vat overview", "vat return", "omzet hoog tarief", "belastbare omzet"]),
            ("client_notes", ["klantnotities", "klantnotitie", "client notes", "client memo"]),
            ("contract", ["contract", "overeenkomst", "service agreement", "payment terms"]),
            ("insurance_document", ["polis", "polisoverzicht", "verzekering", "dekking", "insured amount"]),
            ("annual_accounts_checklist", ["jaarrekening checklist", "annual accounts checklist"]),
            ("vat_guidance", ["btw controle", "btw checklist", "omzetbelasting controle"]),
            ("insurance_guidance", ["verzekeringsrisico", "risico checklist", "insurance risk checklist"]),
            ("tax_law", ["tax law", "tax code", "belastingwet", "internal revenue code"]),
            ("regulation", ["regulation", "directive", "compliance rule", "ifrs", "gaap"]),
            ("annual_report", ["annual report", "form 10-k", "jaarverslag"]),
            ("bank_statement", ["bankafschrift", "bank statement", "account statement", "iban"]),
            ("invoice", ["invoice", "factuur", "bill to"]),
            ("ledger", ["ledger", "general ledger", "trial balance", "grootboek"]),
            ("tax_return", ["tax return", "income tax return", "belastingaangifte"]),
            ("correspondence", ["dear", "regards", "sincerely", "letter", "email"]),
        ]
        for doc_type, keywords in mapping:
            if any(k in normalized for k in keywords):
                return doc_type
        return "other"

    def _infer_tax_year(self, source_filename: str, pages: List[Dict[str, Any]]) -> Optional[int]:
        # 1) Filename hint (strong for files like AuditFile2025...)
        filename_years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", source_filename)]

        # 2) Use only the first pages/chunks for inference stability.
        sample_text = " ".join(page.get("content", "")[:8000] for page in pages[:3])

        # 3) XAF-specific strong signal.
        fiscal_year_matches = [
            int(m.group(1))
            for m in re.finditer(r"\bfiscal(?:\s*year|Year)\s*:\s*(20\d{2})\b", sample_text, flags=re.IGNORECASE)
        ]
        if fiscal_year_matches:
            return max(fiscal_year_matches)

        # 4) Date-pattern years (YYYY-MM-DD) are more reliable than raw numbers.
        date_years = [int(y) for y in re.findall(r"\b(20\d{2})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b", sample_text)]

        # 5) Fallback: general year tokens in a safe range.
        text_years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", sample_text)]

        candidates: List[int] = []
        candidates.extend(y for y in filename_years if 1990 <= y <= 2099)
        candidates.extend(y for y in date_years if 1990 <= y <= 2099)
        candidates.extend(y for y in text_years if 1990 <= y <= 2099)
        if not candidates:
            return None

        # Prefer the most frequent year; tie-break on recency.
        counts: Dict[int, int] = {}
        for y in candidates:
            counts[y] = counts.get(y, 0) + 1
        best_year = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]
        return best_year

    def _infer_jurisdiction(self, source_filename: str, pages: List[Dict[str, Any]]) -> Optional[str]:
        text_blob = f"{source_filename} " + " ".join(
            p.get("content", "")[:2000] for p in pages[:3]
        )
        t = text_blob.lower()
        dutch_terms = [
            "btw", "omzetbelasting", "jaarrekening", "balans", "winst-en-verliesrekening",
            "klantnotities", "polis", "verzekering", "belastingdienst", "mkb", "vof", "bv", "nv",
            "netherlands", "nederland", "dutch",
        ]
        us_terms = [
            "sec", "form 10-k", "10-k", "10-q", "irs", "delaware", "united states",
            "u.s.", "usa", "nasdaq", "nyse", "gaap",
        ]
        eu_terms = ["european union", "eu directive", "eu regulation"]
        germany_terms = ["germany", "deutschland", "bundesfinanzministerium"]
        uk_terms = ["united kingdom", " hmrc", "companies house"]

        dutch_score = sum(1 for kw in dutch_terms if kw in t)
        us_score = sum(1 for kw in us_terms if kw in t)
        if dutch_score > 0 and dutch_score >= us_score:
            return "Netherlands"
        if us_score >= 2 and us_score > dutch_score:
            return "United States"
        if any(k in t for k in eu_terms):
            return "EU"
        if any(k in t for k in germany_terms):
            return "Germany"
        if any(k in t for k in uk_terms):
            return "UK"
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
