"""Llama/Ollama service for financial document Q&A."""

from typing import Dict, Any, List, Optional, Tuple
import re
import json
from time import perf_counter

from src.db.elasticsearch_client import get_elasticsearch_client
from src.services.ollama_client import chat as ollama_chat, list_models as ollama_list_models, OllamaError
from src.services.calculation_sandbox import CalculationSandbox, CalculationSandboxError
from src.services.formula_registry import get_formula_registry, FormulaDefinition
from src.services.financial_value_extractor import FinancialValueExtractor
from src.services.advisor_intents import detect_advisor_intent
from src.services.verification_service import AnswerVerificationService
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
SERVICE_VERSION = 5


class LlamaService:
    """Q&A service using Elasticsearch retrieval and Ollama."""

    CONTEXT_KEYWORDS = {
        "jurisdiction": ["tax rate", "vat", "deductible", "deduction", "threshold", "deadline", "exemption", "filing"],
        "tax_year": ["tax year", "fiscal year", "deadline", "threshold", "rate", "allowance", "deductible"],
        "entity_type": ["deductible", "corporate", "income tax", "filing", "return", "allowance"],
    }
    BASE_PROMPT_RULES = (
        "Rules you must follow:\n"
        "1. Ground every statement in the retrieved documents.\n"
        "2. Cite sources using [Document X].\n"
        "3. Do not invent tax rates, thresholds, deadlines, exemptions, legal interpretations, or numbers.\n"
        "4. If documents are insufficient, say: \"I could not find this in the provided documents.\"\n"
        "5. Distinguish factual extraction from interpretation.\n"
        "6. If multiple documents conflict, describe the conflict and cite both.\n"
        "7. Before answering, verify sources match company, document, period, jurisdiction, and tax year in the question.\n"
        "8. For numeric calculations, include a calculation payload only when values are explicitly present in the retrieved text:\n"
        "```calc\n"
        "{\"expression\":\"(revenue_2026 - revenue_2025) / revenue_2025 * 100\",\"variables\":{\"revenue_2026\":109896,\"revenue_2025\":90234},\"label\":\"Revenue growth\",\"unit\":\"%\"}\n"
        "```\n"
        "If data is missing, do not include this block.\n"
    )
    WORKFLOW_PROMPT_RULES = (
        "Workflow rules:\n"
        "1. Use only the retrieved document context.\n"
        "2. Show source references for claims.\n"
        "3. Keep evidence separate from interpretation.\n"
        "4. If information is missing, say that clearly.\n"
        "5. Do not provide final certified accounting, tax, legal, or insurance advice.\n"
        "6. Include practical next steps.\n"
        "7. If the user asks in Dutch, answer in natural professional Dutch.\n"
    )
    INTENT_RETRIEVAL_TERMS = {
        "missing_info_check": [
            "jaarrekening", "annual accounts", "balance sheet", "balans", "profit and loss",
            "winst-en-verliesrekening", "btw", "btw-overzicht", "bank", "invoice", "factuur",
            "depreciation", "afschrijving", "loan", "lening", "assets", "activa",
            "liabilities", "schulden", "notes", "klantnotities", "contract",
        ],
        "inconsistency_check": [
            "winst-en-verliesrekening", "profit and loss", "omzet", "revenue", "kosten",
            "expenses", "btw-overzicht", "vat", "klantnotities", "invoices", "facturen",
            "sales", "purchases",
        ],
        "advisory_points": [
            "omzet", "revenue", "margin", "marge", "kosten", "cash flow", "liquiditeit",
            "tax", "vat", "btw", "risk", "risico", "contract", "growth", "groei", "decline",
            "daling", "customer", "klant", "supplier", "leverancier",
        ],
        "insurance_risk_check": [
            "insurance", "verzekering", "verzekeringsrisico", "assets", "activa", "equipment",
            "inventaris", "voorraad", "inventory", "contract", "liability", "aansprakelijkheid",
            "cyber", "transport", "bedrijfsactiviteiten", "business activity",
        ],
        "client_file_summary": [
            "financial statements", "winst-en-verliesrekening", "balans", "tax", "vat", "btw",
            "klantnotities", "contract", "insurance", "verzekering", "cash flow", "liquiditeit",
        ],
    }
    WORKFLOW_INTENTS = {
        "missing_info_check",
        "inconsistency_check",
        "advisory_points",
        "insurance_risk_check",
        "client_file_summary",
    }
    TICKER_STOPWORDS = {
        "MKB", "SME", "BV", "NV", "ZZP", "BTW", "VAT", "KPI", "AI", "API",
        "P", "L", "IB", "VPB", "EU", "NL", "PDF", "OCR", "IFRS", "GAAP",
    }
    CLIENT_DOC_HINTS = [
        "winst", "verlies", "balans", "btw", "klantnotities", "contract", "polis",
        "bank", "factuur", "invoice", "voorraad", "liquiditeit",
    ]
    REFERENCE_DOC_HINTS = ["referentie", "checklist", "belastingdienst", "guidance", "framework"]
    MISSING_INFO_CHECK_ITEMS = [
        {
            "key": "bank_support",
            "label": "Bankafschriften",
            "terms": ["bankafschrift", "bank statement", "bank statements", "banktransactie", "bank transaction"],
            "why": "Bankafschriften zijn nodig om kas- en bankmutaties en eindsaldi te controleren.",
            "next_step": "Vraag de ontbrekende bankafschriften of een transactiedetail per periode op.",
        },
        {
            "key": "invoice_support",
            "label": "Factuuronderbouwing",
            "terms": ["factuur", "invoice", "sales invoice", "purchase invoice", "crediteuren", "debiteuren"],
            "why": "Facturen zijn nodig om omzet, kosten en btw-aansluitingen te onderbouwen.",
            "next_step": "Controleer of inkoop- en verkoopfacturen compleet en herleidbaar zijn.",
        },
        {
            "key": "inventory_value",
            "label": "Voorraadwaarde",
            "terms": ["voorraad", "inventory", "stock", "inventaris"],
            "why": "De voorraadwaardering beïnvloedt zowel het resultaat als de balans.",
            "next_step": "Vraag een voorraadopstelling met waarderingsmethode en peildatum op.",
        },
        {
            "key": "depreciation_assets",
            "label": "Afschrijvingen en activa-details",
            "terms": ["afschrijving", "depreciation", "fixed assets", "materiele vaste activa", "intangible assets"],
            "why": "Activa- en afschrijvingsspecificaties zijn nodig om het resultaat en de balanswaarden te onderbouwen.",
            "next_step": "Controleer activa-specificaties, aanschafwaarden, en afschrijvingsschema.",
        },
        {
            "key": "loans_liabilities",
            "label": "Leningen en schulden",
            "terms": ["lening", "loan", "liability", "schuld", "schulden", "krediet"],
            "why": "Lening- en schuldenspecificaties zijn nodig om verplichtingen, aflossingen en rentelasten te controleren.",
            "next_step": "Vraag leningsovereenkomsten, aflossingsschema, en renteoverzicht op.",
        },
        {
            "key": "vat_reconciliation",
            "label": "BTW-aansluiting",
            "terms": ["btw", "vat", "btw-overzicht", "omzetbelasting", "vat return"],
            "why": "Een btw-aansluiting is nodig om omzet, aangiften en grootboekmutaties op elkaar aan te laten sluiten.",
            "next_step": "Vergelijk btw-overzicht met omzet en grootboekmutaties per periode.",
        },
        {
            "key": "contract_details",
            "label": "Contractdetails",
            "terms": ["contract", "overeenkomst", "service agreement", "payment terms", "betaaltermijn"],
            "why": "Contractvoorwaarden helpen om omzetmomenten, verplichtingen en risico’s te onderbouwen.",
            "next_step": "Controleer contracten op looptijd, prijsafspraken, en verplichtingen.",
        },
        {
            "key": "insurance_coverage",
            "label": "Verzekeringsinformatie",
            "terms": ["verzekering", "polis", "coverage", "insured amount", "aansprakelijkheid"],
            "why": "Verzekeringsinformatie is relevant voor risicobeoordeling en continuïteit.",
            "next_step": "Vraag polisvoorwaarden en dekking per risicocategorie op.",
        },
    ]
    WORKFLOW_HINT_TERMS = {
        "missing_info_check": [
            "ontbreekt", "ontbrekend", "niet aangeleverd", "onduidelijk", "nog te ontvangen",
            "geen specificatie", "factuur ontbreekt", "bankafschrift ontbreekt",
            "voorraadwaarde onduidelijk", "contract niet volledig", "polis ontbreekt",
            "afschrijving", "lening", "schuld", "activa",
        ],
        "inconsistency_check": [
            "omzet", "btw", "btw-overzicht", "winst-en-verliesrekening", "verschil",
            "aansluiting", "komt niet overeen", "klantnotities", "facturen", "verkoop", "inkoop",
        ],
        "advisory_points": [
            "omzet", "marge", "kosten", "liquiditeit", "btw", "risico", "voorraad",
            "contract", "klant", "leverancier", "groei", "daling", "ontbreekt", "verzekering",
        ],
        "insurance_risk_check": [
            "verzekering", "polis", "dekking", "voorraad", "inventaris", "activa",
            "bedrijfsmiddelen", "contract", "aansprakelijkheid", "cyber", "klantdata",
            "transport", "onderverzekering",
        ],
    }

    def __init__(self, model: str = "llama3.2"):
        if settings.demo_mode and settings.demo_ollama_model:
            self.model = settings.demo_ollama_model
        else:
            self.model = model or settings.ollama_model
        self.es_client = get_elasticsearch_client()
        self.formula_registry = get_formula_registry()
        self.value_extractor = FinancialValueExtractor(self.formula_registry)
        self.verification_service = AnswerVerificationService()
        self.retrieval_top_k = max(1, int(settings.retrieval_top_k))
        self.final_context_chunks = max(1, int(settings.final_context_chunks))
        self.max_context_chars = max(200, int(settings.max_context_chars))
        self.max_chars_per_chunk = max(300, int(settings.max_chars_per_chunk))
        self.enable_latency_logs = bool(settings.enable_latency_logs)
        self.ollama_num_predict = max(128, int(settings.ollama_num_predict))
        self._service_version = SERVICE_VERSION

    def _latency_logs_enabled(self) -> bool:
        return bool(getattr(self, "enable_latency_logs", True))

    def ask(
        self,
        question: str,
        max_context_docs: int = 5,
        temperature: float = 0.3,
        corpus_type: Optional[str] = None,
        system_context: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        tax_year: Optional[int] = None,
        entity_type: Optional[str] = None,
        client_name: Optional[str] = None,
        document_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Ask a question and return an answer with sources."""
        try:
            logger.info(f"Financial question: {question}")
            request_started = perf_counter()
            detected_intent = self._detect_advisor_intent(question)
            logger.info(f"Detected advisor intent: {detected_intent}")
            timing = {
                "retrieval_ms": 0.0,
                "consistency_ms": 0.0,
                "formula_ms": 0.0,
                "prompt_ms": 0.0,
                "ollama_ms": 0.0,
                "post_ms": 0.0,
                "llm_ms": 0.0,
            }

            def finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
                answer_text = payload.get("answer")
                if isinstance(answer_text, str) and payload.get("found_documents", False):
                    verification = self.verification_service.verify(
                        question=question,
                        answer=answer_text,
                        sources=payload.get("sources", []) or [],
                        search_results=search_results,
                    )
                    if verification.status == "fail":
                        if detected_intent == "inconsistency_check":
                            repaired_answer = self._build_inconsistency_answer_verified(search_results)
                            repaired_verification = self.verification_service.verify(
                                question=question,
                                answer=repaired_answer,
                                sources=payload.get("sources", []) or [],
                                search_results=search_results,
                            )
                            if repaired_verification.status == "pass":
                                payload["answer"] = repaired_answer
                                payload["warnings"] = (payload.get("warnings") or []) + [
                                    "Inconsistentie-antwoord automatisch hersteld na bron- en rekencontrole."
                                ]
                            else:
                                payload["answer"] = repaired_answer
                                payload["warnings"] = (payload.get("warnings") or []) + [
                                    "Inconsistentie-antwoord herbouwd met strengere labels/perioden; handmatige controle blijft nodig."
                                ] + repaired_verification.issues
                        else:
                            payload["answer"] = verification.safe_answer
                            payload["warnings"] = (payload.get("warnings") or []) + verification.issues
                total_ms = (perf_counter() - request_started) * 1000
                if self._latency_logs_enabled():
                    logger.info(
                        "[latency] formula=%.3fs prompt=%.3fs ollama=%.3fs total=%.3fs",
                        timing["formula_ms"] / 1000.0,
                        timing["prompt_ms"] / 1000.0,
                        timing["ollama_ms"] / 1000.0,
                        total_ms / 1000.0,
                    )
                return payload

            filters_used = {
                "corpus_type": corpus_type,
                "document_type": document_type,
                "jurisdiction": jurisdiction,
                "tax_year": tax_year,
                "entity_type": entity_type,
                "client_name": client_name,
                "intent": detected_intent,
            }
            retrieval_query = self._build_intent_retrieval_query(question, detected_intent)
            retrieval_limit = max(1, min(max_context_docs or self.retrieval_top_k, 20))

            retrieval_started = perf_counter()
            search_results = self._retrieve_with_intent_strategy(
                question=retrieval_query,
                detected_intent=detected_intent,
                retrieval_limit=retrieval_limit,
                system_context=system_context,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                corpus_type=corpus_type,
            )
            timing["retrieval_ms"] = (perf_counter() - retrieval_started) * 1000

            consistency_started = perf_counter()
            search_results, consistency_error, consistency_notes = self._enforce_source_consistency(
                question=question,
                search_results=search_results,
                intent=detected_intent,
            )
            timing["consistency_ms"] = (perf_counter() - consistency_started) * 1000

            if not search_results:
                return finalize({
                    "answer": self._no_results_response(question),
                    "sources": [],
                    "model": self.model,
                    "found_documents": False,
                    "filters_used": filters_used,
                })

            if consistency_error:
                return finalize({
                    "answer": consistency_error,
                    "sources": self._format_sources(search_results, list(range(min(3, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                    "warnings": consistency_notes,
                })

            if detected_intent not in self.WORKFLOW_INTENTS and self._is_low_confidence(question, search_results):
                return finalize({
                    "answer": self._low_confidence_response(),
                    "sources": self._format_sources(search_results, list(range(min(2, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                })

            if detected_intent not in self.WORKFLOW_INTENTS:
                context_requirements = self._detect_context_requirements(question)
                missing_context_message = self._build_missing_context_message(
                    context_requirements=context_requirements,
                    jurisdiction=jurisdiction,
                    tax_year=tax_year,
                    entity_type=entity_type,
                    search_results=search_results
                )
                if missing_context_message:
                    return finalize({
                        "answer": missing_context_message,
                        "sources": self._format_sources(search_results, list(range(min(3, len(search_results))))),
                        "model": self.model,
                        "found_documents": True,
                        "num_documents_used": len(search_results),
                        "filters_used": filters_used,
                    })

            if detected_intent == "missing_info_check":
                missing_info_answer = self._build_missing_info_answer(search_results)
                return finalize({
                    "answer": missing_info_answer,
                    "sources": self._format_sources(search_results, list(range(min(5, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                    "warnings": consistency_notes,
                })
            if detected_intent in {"inconsistency_check", "advisory_points", "insurance_risk_check", "client_file_summary"}:
                workflow_answer = self._build_workflow_answer(detected_intent, search_results, question=question)
                return finalize({
                    "answer": workflow_answer,
                    "sources": self._format_sources(search_results, list(range(min(5, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                    "warnings": consistency_notes,
                })

            formula_started = perf_counter()
            formula_calc = self._try_formula_registry_calculation(
                question=question,
                base_results=search_results,
                corpus_type=corpus_type,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
            )
            timing["formula_ms"] = (perf_counter() - formula_started) * 1000
            if formula_calc:
                return finalize({
                    "answer": formula_calc["answer"],
                    "sources": formula_calc["sources"],
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                    "warnings": consistency_notes + formula_calc.get("warnings", []),
                })

            direct_calc_answer = self._try_direct_financial_calculation(question, search_results)
            if direct_calc_answer:
                return finalize({
                    "answer": direct_calc_answer,
                    "sources": self._format_sources(search_results, list(range(min(3, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                })

            direct_numeric_answer = self._try_direct_numeric_answer(question, search_results)
            if direct_numeric_answer:
                return finalize({
                    "answer": direct_numeric_answer,
                    "sources": self._format_sources(search_results, list(range(min(3, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                })

            llm_started = perf_counter()
            prompt_started = perf_counter()
            context_results = search_results[: self.final_context_chunks]
            context = self._build_context(context_results)
            prompt = self._create_prompt(
                question=question,
                context=context,
                history=history,
                intent=detected_intent,
            )
            timing["prompt_ms"] = (perf_counter() - prompt_started) * 1000

            ollama_started = perf_counter()
            answer = self._generate_llama_answer(prompt=prompt, temperature=temperature)
            timing["ollama_ms"] = (perf_counter() - ollama_started) * 1000

            post_started = perf_counter()
            answer = self._strip_disclaimer_section(answer)
            answer = self._apply_calculation_sandbox(
                answer=answer,
                search_results=search_results,
                question=question,
            )
            answer = self._sanitize_missing_info_section(answer, consistency_notes)
            timing["post_ms"] = (perf_counter() - post_started) * 1000
            timing["llm_ms"] = (perf_counter() - llm_started) * 1000

            cited_indices = self._extract_cited_indices(answer, len(search_results))
            if not cited_indices:
                cited_indices = list(range(min(3, len(search_results))))

            sources = self._format_sources(search_results, cited_indices)

            return finalize({
                "answer": answer,
                "sources": sources,
                "model": self.model,
                "found_documents": True,
                "num_documents_used": len(search_results),
                "filters_used": filters_used,
                "warnings": consistency_notes,
            })

        except Exception as e:
            logger.error(f"Error in Llama service: {e}", exc_info=True)
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "model": self.model,
                "found_documents": False,
                "num_documents_used": 0,
                "error": str(e),
            }

    def _deduplicate_results(
        self,
        search_results: List[Dict[str, Any]],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for row in search_results:
            key = (
                row.get("document_id"),
                row.get("filename"),
                row.get("page_number"),
                (row.get("snippet") or row.get("content") or "")[:200],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if limit is not None and len(deduped) >= limit:
                break
        return deduped

    def _retrieve_with_intent_strategy(
        self,
        question: str,
        detected_intent: str,
        retrieval_limit: int,
        system_context: Optional[str],
        jurisdiction: Optional[str],
        tax_year: Optional[int],
        entity_type: Optional[str],
        client_name: Optional[str],
        document_type: Optional[str],
        corpus_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        if detected_intent not in self.WORKFLOW_INTENTS:
            rows = self.es_client.search(
                query=question,
                limit=retrieval_limit,
                enable_fuzzy=True,
                system_context=system_context,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                corpus_type=corpus_type,
                use_vector=True,
            )
            return self._deduplicate_results(rows, limit=retrieval_limit)

        base_rows = self.es_client.search(
            query=question,
            limit=retrieval_limit,
            enable_fuzzy=True,
            system_context=system_context,
            jurisdiction=jurisdiction,
            tax_year=tax_year,
            entity_type=entity_type,
            client_name=client_name,
            document_type=document_type,
            corpus_type=corpus_type,
            use_vector=True,
        )
        uploaded_rows = self.es_client.search(
            query=question,
            limit=retrieval_limit,
            enable_fuzzy=True,
            system_context=system_context,
            jurisdiction=jurisdiction,
            tax_year=tax_year,
            entity_type=entity_type,
            client_name=client_name,
            document_type=document_type,
            corpus_type="uploaded",
            use_vector=True,
        )
        existing_rows = self.es_client.search(
            query=question,
            limit=max(3, retrieval_limit // 2),
            enable_fuzzy=True,
            system_context=system_context,
            jurisdiction=jurisdiction,
            tax_year=tax_year,
            entity_type=entity_type,
            client_name=client_name,
            document_type=document_type,
            corpus_type="existing",
            use_vector=True,
        )

        merged = self._deduplicate_results(uploaded_rows + base_rows + existing_rows, limit=None)
        prioritized = self._prioritize_advisor_results(merged)
        return prioritized[:retrieval_limit]

    def _prioritize_advisor_results(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rescored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            score = float(row.get("score", 0.0) or 0.0)
            text = self._row_text(row)
            filename = str(row.get("filename", "")).lower()
            corpus = str(row.get("corpus_type", "")).lower()

            if corpus == "uploaded":
                score += 5.0
            elif corpus == "existing":
                score += 1.0

            if any(tag in text or tag in filename for tag in self.CLIENT_DOC_HINTS):
                score += 2.5
            if any(tag in text or tag in filename for tag in self.REFERENCE_DOC_HINTS):
                score += 0.5

            rescored.append((score, row))

        rescored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in rescored]

    def _build_context(self, search_results: List[Dict[str, Any]], label_prefix: str = "Document") -> str:
        context_parts = []
        used_chars = 0
        for i, result in enumerate(search_results, 1):
            filename = result.get("filename", "Unknown")
            category = result.get("category", "other")
            page_num = result.get("page_number")
            section_reference = result.get("section_reference")
            jurisdiction = result.get("jurisdiction")
            tax_year = result.get("tax_year")
            entity_type = result.get("entity_type")

            content = (
                (result.get("summary") or "").strip()
                or (result.get("snippet") or "").strip()
                or (result.get("content") or "").strip()
            )
            content = self._compress_context_text(content)
            if len(content) > self.max_chars_per_chunk:
                content = content[: self.max_chars_per_chunk] + "..."

            location = f"page {page_num}" if page_num else "page unknown"
            if section_reference:
                location = f"{location}, section {section_reference}"

            context_block = (
                f"--- {label_prefix} {i} ---\n"
                f"Source: {filename}\n"
                f"Location: {location}\n"
                f"Type: {category}\n"
                f"Jurisdiction: {jurisdiction or 'unknown'}\n"
                f"Tax Year: {tax_year if tax_year is not None else 'unknown'}\n"
                f"Entity Type: {entity_type or 'unknown'}\n"
                f"Extract:\n{content}\n"
            )
            projected = used_chars + len(context_block)
            if projected > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining < 400:
                    break
                context_block = context_block[:remaining].rstrip() + "\n"
                context_parts.append(context_block)
                break
            context_parts.append(context_block)
            used_chars = projected

        return "\n".join(context_parts)

    def _compress_context_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        # Remove common page/chrome noise while keeping table/number rows.
        cleaned = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bconfidential\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def _create_prompt(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        intent: str = "normal_qna",
    ) -> str:
        history_text = ""
        if history:
            history_text = "Conversation History:\n"
            for msg in history[-3:]:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            history_text += "\n"

        formula_hints = ""
        if self._is_calc_intent_fast(question):
            formula_hints = self.formula_registry.render_prompt_hints(question, max_results=4)
        formula_hints_block = f"\n{formula_hints}\n" if formula_hints else ""
        workflow_output = self._workflow_output_format(intent)

        return (
            "You are a financial document assistant. You help users understand tax law, accounting documents, "
            "and financial records based only on the documents provided to you.\n\n"
            f"{self.BASE_PROMPT_RULES}\n"
            f"{self.WORKFLOW_PROMPT_RULES}\n"
            f"Retrieved documents:\n{context}\n{formula_hints_block}\n"
            f"{history_text}User Question: {question}\n\n"
            f"Answer in this exact structure:\n{workflow_output}\n\n"
            "Answer:"
        )

    def _workflow_output_format(self, intent: str) -> str:
        if intent == "missing_info_check":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting]\n\n"
                "Gevonden informatie:\n"
                "- [item], bron: [document/pagina]\n\n"
                "Ontbrekende of onduidelijke informatie:\n"
                "- [punt]\n\n"
                "Waarom dit belangrijk is:\n"
                "[korte toelichting]\n\n"
                "Volgende stap:\n"
                "[wat op te vragen of te controleren]\n\n"
                "Bronnen:\n"
                "[bronnenlijst]"
            )
        if intent == "inconsistency_check":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting]\n\n"
                "Mogelijke inconsistenties:\n"
                "1. [issue]\n"
                "   - Bewijs A: [document/pagina]\n"
                "   - Bewijs B: [document/pagina]\n"
                "   - Waarom dit mogelijk inconsistent is: [korte toelichting]\n\n"
                "Geen duidelijke inconsistentie gevonden:\n"
                "- [optioneel]\n\n"
                "Volgende stap:\n"
                "[wat handmatig te verifieren]\n\n"
                "Bronnen:\n"
                "[bronnenlijst]"
            )
        if intent == "advisory_points":
            return (
                "Kort antwoord:\n"
                "Hier zijn drie adviespunten om met de klant te bespreken.\n\n"
                "Adviespunten om met de klant te bespreken:\n"
                "1. [adviespunt]\n"
                "   - Waarom dit belangrijk is:\n"
                "   - Bewijs/bron:\n"
                "   - Vraag aan de klant:\n\n"
                "2. [adviespunt]\n"
                "   - Waarom dit belangrijk is:\n"
                "   - Bewijs/bron:\n"
                "   - Vraag aan de klant:\n\n"
                "3. [adviespunt]\n"
                "   - Waarom dit belangrijk is:\n"
                "   - Bewijs/bron:\n"
                "   - Vraag aan de klant:\n\n"
                "Belangrijke opmerking:\n"
                "Dit zijn voorbereidingspunten, geen definitief advies.\n\n"
                "Bronnen:\n"
                "[bronnenlijst]"
            )
        if intent == "insurance_risk_check":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting]\n\n"
                "Mogelijke verzekeringsrisico’s:\n"
                "1. [risico]\n"
                "   - Bewijs:\n"
                "   - Waarom dit belangrijk is:\n"
                "   - Wat te controleren:\n\n"
                "2. [risico]\n"
                "   - Bewijs:\n"
                "   - Waarom dit belangrijk is:\n"
                "   - Wat te controleren:\n\n"
                "Ontbrekende informatie:\n"
                "- [ontbrekend punt]\n\n"
                "Bronnen:\n"
                "[bronnenlijst]"
            )
        if intent == "client_file_summary":
            return (
                "Samenvatting klantdossier:\n\n"
                "Bedrijfsactiviteit:\n"
                "[samenvatting]\n\n"
                "Financiële punten:\n"
                "- [punt]\n\n"
                "Belasting-/btw-punten:\n"
                "- [punt]\n\n"
                "Risico’s of aandachtspunten:\n"
                "- [punt]\n\n"
                "Ontbrekende informatie:\n"
                "- [punt]\n\n"
                "Vervolgvragen:\n"
                "- [vraag]\n\n"
                "Bronnen:\n"
                "[bronnenlijst]"
            )
        return (
            "1. Direct answer\n"
            "2. Evidence from documents\n"
            "3. Important assumptions or missing information\n"
            "4. Suggested next step"
        )

    def _generate_llama_answer(self, prompt: str, temperature: float) -> str:
        try:
            return ollama_chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature,
                num_predict=self.ollama_num_predict
            )
        except OllamaError as exc:
            raise RuntimeError(str(exc)) from exc

    def _extract_cited_indices(self, answer: str, total_results: int) -> List[int]:
        cited_indices = set()
        for match in re.findall(r"Document\s+(\d+)", answer):
            try:
                idx = int(match) - 1
                if 0 <= idx < total_results:
                    cited_indices.add(idx)
            except ValueError:
                continue
        return sorted(cited_indices)

    def _apply_calculation_sandbox(
        self,
        answer: str,
        search_results: List[Dict[str, Any]],
        question: str,
    ) -> str:
        """Run an optional calc payload in the local sandbox."""
        if not answer:
            return answer

        match = re.search(r"```calc\s*(\{.*?\})\s*```", answer, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return answer

        raw_json = match.group(1)
        cleaned_answer = answer[:match.start()] + answer[match.end():]
        cleaned_answer = cleaned_answer.strip()

        try:
            payload = json.loads(raw_json)
            expression = payload.get("expression")
            variables = payload.get("variables", {})
            label = payload.get("label")
            unit = payload.get("unit")

            value_sources = self._collect_variable_sources(variables, search_results)
            if len(value_sources) != len(variables):
                logger.warning("Calculation skipped: one or more variables not found in retrieved evidence")
                return (
                    f"{cleaned_answer}\n\nCalculation could not be completed from the provided documents "
                    "because one or more required values were not found in the retrieved evidence."
                ).strip()

            value = CalculationSandbox.evaluate(expression=expression, variables=variables)
            value_text = f"{value:,.6f}".rstrip("0").rstrip(".")

            details = ["Calculation details:"]
            details.append(f"- Formula: {expression}")
            for var_name, meta in value_sources.items():
                details.append(
                    f"- {var_name} = {meta['value_text']} (source: {meta['source_label']})"
                )
            details.append(f"- Result: {label + ' = ' if label else ''}{value_text}{(' ' + unit) if unit else ''}")
            calc_line = "\n\n" + "\n".join(details)
            return cleaned_answer + calc_line
        except (json.JSONDecodeError, CalculationSandboxError, TypeError, ValueError) as exc:
            logger.warning(f"Calculation sandbox skipped due to invalid payload: {exc}")
            return cleaned_answer

    def _strip_disclaimer_section(self, answer: str) -> str:
        """Remove trailing disclaimer sections if the model adds them."""
        if not answer:
            return answer

        cleaned = answer
        cleaned = re.sub(
            r"\n?\s*5\.\s*Disclaimer:.*$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"\n?\s*Disclaimer:.*$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip()

    def _try_direct_financial_calculation(
        self,
        question: str,
        search_results: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Fallback for COGS when it is present as a line item in source text."""
        q = (question or "").lower()
        asks_cogs = (
            "cogs" in q
            or "cost of goods sold" in q
            or ("cost of revenues" in q)
            or ("cost of revenue" in q)
        )
        if not asks_cogs:
            return None

        values: List[str] = []
        supporting_docs: List[int] = []
        for idx, item in enumerate(search_results[:6], start=1):
            text = " ".join(
                [
                    str(item.get("content", "")),
                    str(item.get("summary", "")),
                    str(item.get("snippet", "")),
                ]
            )
            text = re.sub(r"\s+", " ", text)
            matches = re.findall(
                r"(?:costs?\s+of\s+revenues?|costs?\s+of\s+revenue|costs?\s+of\s+goods\s+sold|cogs)"
                r"[^\d$]{0,20}\$?\s*([0-9][0-9,]*(?:\.\d+)?)"
                r"(?:[^\d$]{1,15}\$?\s*([0-9][0-9,]*(?:\.\d+)?))?",
                text,
                flags=re.IGNORECASE,
            )
            for g1, g2 in matches:
                if g1:
                    values.append(g1)
                if g2:
                    values.append(g2)
                supporting_docs.append(idx)

        dedup_values: List[str] = []
        seen = set()
        for v in values:
            key = v.replace(",", "")
            if key in seen:
                continue
            seen.add(key)
            dedup_values.append(v)

        if not dedup_values:
            return None

        doc_refs = sorted(set(supporting_docs))[:3]
        refs = ", ".join([f"[Document {d}]" for d in doc_refs]) if doc_refs else ""
        value_text = ", ".join(dedup_values[:4])

        return (
            "1. Direct answer\n"
            f"Based on the retrieved documents, COGS corresponds to the reported cost of revenues: {value_text} {refs}.\n\n"
            "2. Evidence from documents\n"
            "The retrieved source text includes line items labeled 'cost of revenues/cost of revenue', which are used as COGS in this context.\n\n"
            "3. Important assumptions or missing information\n"
            "This assumes the document's 'cost of revenues' is the intended COGS definition for your question and reporting scope.\n\n"
            "4. Suggested next step\n"
            "Confirm the period/year column mapping in the source table (for example 2025 vs 2026) before final reporting."
        )

    def _try_direct_numeric_answer(
        self,
        question: str,
        search_results: List[Dict[str, Any]],
    ) -> Optional[str]:
        q = (question or "").lower()
        metric_aliases = {
            "revenue": "revenue",
            "revenues": "revenue",
            "net income": "net_income",
            "operating income": "operating_income",
            "gross profit": "gross_profit",
            "cost of revenue": "cost_of_revenue",
            "cost of sales": "cost_of_revenue",
            "total assets": "total_assets",
            "total liabilities": "total_liabilities",
            "current assets": "current_assets",
            "current liabilities": "current_liabilities",
        }
        requested = None
        for alias, variable in metric_aliases.items():
            if alias in q and ("what" in q or "show" in q or "value" in q or "amount" in q):
                requested = variable
                break
        if not requested:
            return None

        formulas = self.formula_registry.list_formulas()
        formula = formulas[0] if formulas else None
        if formula is None:
            return None
        value = self._find_value_for_variable(
            variable=requested,
            formula=formula,
            pool=search_results,
            desired_year=None,
            company_target=None,
        )
        if not value:
            return None

        return (
            "1. Direct answer\n"
            f"{value['display_label']}: {value['value_text']} (from {value['source_label']}).\n\n"
            "2. Evidence from documents\n"
            f"The retrieved source line is: \"{value['line_text']}\".\n\n"
            "3. Important assumptions or missing information\n"
            "The value is taken directly from the retrieved documents and may depend on the reported unit scale.\n\n"
            "4. Suggested next step\n"
            "Confirm the unit label (thousands, millions, or billions) and requested period before reporting."
        )

    def _is_calculation_question(self, question: str) -> bool:
        if not self._is_calc_intent_fast(question):
            return False
        return bool(self.formula_registry.find_by_question(question, max_results=1))

    def _detect_advisor_intent(self, question: str) -> str:
        return detect_advisor_intent(
            question=question,
            is_calculation=self._is_calc_intent_fast(question),
        )

    def _build_intent_retrieval_query(self, question: str, intent: str) -> str:
        terms = self.INTENT_RETRIEVAL_TERMS.get(intent)
        if not terms:
            return question

        base_tokens = set(re.findall(r"\w+", (question or "").lower()))
        extras: List[str] = []
        for term in terms:
            term_tokens = set(re.findall(r"\w+", term.lower()))
            if term_tokens.issubset(base_tokens):
                continue
            extras.append(term)

        if not extras:
            return question

        expanded = f"{question} {' '.join(extras)}".strip()
        logger.info("Intent retrieval expansion applied for %s with %d terms", intent, len(extras))
        return expanded

    def _build_missing_info_answer(self, search_results: List[Dict[str, Any]]) -> str:
        client_rows, reference_rows = self._split_client_reference_rows(search_results)
        found_rows: List[str] = []
        missing_rows: List[str] = []
        why_rows: List[str] = []
        next_rows: List[str] = []
        source_lines: List[str] = self._collect_source_lines(search_results, limit=8)

        for item in self.MISSING_INFO_CHECK_ITEMS:
            explicit_missing = None
            client_evidence = None
            checklist_evidence = None
            for row in client_rows:
                text = self._row_text(row)
                if any(term in text for term in item["terms"]) and self._contains_missing_signal(text):
                    explicit_missing = row
                    break
                if any(term in text for term in item["terms"]):
                    client_evidence = row
            if explicit_missing:
                missing_rows.append(
                    f"- {item['label']}: expliciet genoemd als ontbrekend/onduidelijk, bron: {self._row_source_label(explicit_missing)}"
                )
            elif client_evidence:
                found_rows.append(f"- {item['label']}, bron: {self._row_source_label(client_evidence)}")
            else:
                for row in reference_rows:
                    text = self._row_text(row)
                    if any(term in text for term in item["terms"]):
                        checklist_evidence = row
                        break
                if checklist_evidence:
                    missing_rows.append(
                        f"- {item['label']}: aandachtspunt op basis van checklist, niet bevestigd in klantdocumenten (bron checklist: {self._row_source_label(checklist_evidence)})"
                    )

            why_rows.append(f"- {item['label']}: {item['why']}")
            next_rows.append(f"- {item['label']}: {item['next_step']}")

        if not found_rows:
            found_rows.append("- Beperkte concrete onderbouwing in de opgehaalde klantdocumenten.")
        if not missing_rows:
            missing_rows.append("- Geen expliciet ontbrekende punten gevonden in de opgehaalde tekst. Controle op volledigheid blijft nodig.")

        return (
            "Kort antwoord:\n"
            "Op basis van de beschikbare documenten lijken meerdere jaarrekening-onderdelen nog onduidelijk of vragen ze extra onderbouwing.\n\n"
            "Gevonden informatie:\n"
            + "\n".join(found_rows[:10])
            + "\n\n"
            "Ontbrekende of onduidelijke informatie:\n"
            + "\n".join(missing_rows[:10])
            + "\n\n"
            "Waarom dit belangrijk is:\n"
            + "\n".join(why_rows[:5])
            + "\n\n"
            "Volgende stap:\n"
            + "\n".join(next_rows[:5])
            + "\n\n"
            "Bronnen:\n"
            + "\n".join(source_lines)
        ).strip()

    def _find_item_evidence(self, terms: List[str], search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for row in search_results:
            haystack = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("filename", "")),
                    str(row.get("summary", "")),
                    str(row.get("snippet", "")),
                    str(row.get("content", ""))[:4000],
                ]
            ).lower()
            for term in terms:
                if term.lower() in haystack:
                    return row
        return None

    def _build_workflow_answer(self, intent: str, search_results: List[Dict[str, Any]], question: Optional[str] = None) -> str:
        if intent == "inconsistency_check":
            return self._build_inconsistency_answer(search_results, question=question)
        if intent == "advisory_points":
            return self._build_advisory_points_answer(search_results)
        if intent == "insurance_risk_check":
            return self._build_insurance_risk_answer(search_results)
        if intent == "client_file_summary":
            return self._build_client_file_summary_answer(search_results)
        return self._build_missing_info_answer(search_results)

    def _row_text(self, row: Dict[str, Any]) -> str:
        return " ".join(
            [
                str(row.get("title", "")),
                str(row.get("filename", "")),
                str(row.get("summary", "")),
                str(row.get("snippet", "")),
                str(row.get("content", "")),
            ]
        ).lower()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def _detect_row_kind(self, row: Dict[str, Any]) -> str:
        doc_type = self._normalize_text(str(row.get("document_type", "")))
        doc_detail = self._normalize_text(str(row.get("document_type_detail", "")))
        category = self._normalize_text(str(row.get("category", "")))
        filename = self._normalize_text(str(row.get("filename", "")))
        title = self._normalize_text(str(row.get("title", "")))
        content = self._normalize_text(str(row.get("content", "")))
        summary = self._normalize_text(str(row.get("summary", "")))
        text = " ".join([doc_type, doc_detail, category, filename, title, content[:2500], summary[:1000]])

        note_terms = ["klantnotities", "klantnotitie", "client notes", "memo", "notitie"]
        vat_terms = ["btw-overzicht", "btw aangifte", "omzetbelasting", "vat overview", "taxable turnover", "vat turnover"]
        pnl_terms = ["winst-en-verliesrekening", "resultatenrekening", "profit and loss", "p&l", "income statement"]

        if any(t in text for t in note_terms):
            return "notes"
        if any(t in text for t in vat_terms) or doc_detail in {"vat_overview", "vat_guidance"}:
            return "vat"
        if any(t in text for t in pnl_terms) or doc_detail in {"profit_loss"}:
            return "pnl"
        if "contract" in text or "overeenkomst" in text:
            return "contract"
        return "other"

    def _row_source_label(self, row: Dict[str, Any]) -> str:
        return f"{row.get('filename', 'Unknown')}, page {row.get('page_number') or 'unknown'}"

    def _collect_source_lines(self, rows: List[Dict[str, Any]], limit: int = 6) -> List[str]:
        lines: List[str] = []
        seen = set()
        for row in rows:
            key = (row.get("filename"), row.get("page_number"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {self._row_source_label(row)}")
            if len(lines) >= limit:
                break
        return lines

    def _scan_workflow_hints(self, intent: str, rows: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, str]]:
        terms = self.WORKFLOW_HINT_TERMS.get(intent, [])
        if not terms:
            return []
        hints: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            text = self._row_text(row)
            source = self._row_source_label(row)
            for term in terms:
                term_l = term.lower()
                if term_l in text:
                    key = (term_l, source)
                    if key in seen:
                        continue
                    seen.add(key)
                    hints.append({"term": term_l, "source": source})
                    if len(hints) >= limit:
                        return hints
        return hints

    def _split_client_reference_rows(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        client_rows: List[Dict[str, Any]] = []
        reference_rows: List[Dict[str, Any]] = []
        for row in rows:
            corpus = str(row.get("corpus_type", "")).lower()
            category = str(row.get("category", "")).lower()
            text = self._row_text(row)
            if corpus == "uploaded":
                client_rows.append(row)
                continue
            if corpus == "existing":
                reference_rows.append(row)
                continue
            if category in {"tax_law", "regulation"} or any(t in text for t in ["referentie", "checklist", "guidance", "belastingdienst"]):
                reference_rows.append(row)
            else:
                client_rows.append(row)
        return client_rows, reference_rows

    def _extract_numbers(self, text: str) -> List[float]:
        values: List[float] = []
        for token in re.findall(r"(?:€|\$)?\s*\(?\d[\d\.,]*\)?", text or ""):
            val = self._parse_numeric_token(token)
            if val is None:
                continue
            values.append(val)
        return values

    def _row_primary_text(self, row: Dict[str, Any]) -> str:
        content = str(row.get("content", "") or "").strip()
        if content:
            return content
        summary = str(row.get("summary", "") or "").strip()
        if summary:
            return summary
        return str(row.get("snippet", "") or "")

    def _extract_named_values(self, rows: List[Dict[str, Any]], label_terms: List[str], max_hits: int = 6) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        seen = set()
        pattern = r"(?:€|\$)?\s*\(?\d[\d\.,]*\)?"
        for row in rows:
            raw = self._row_primary_text(row)
            line_candidates = re.split(r"[\n\r]|(?<=[\.;])\s+", raw)
            for line in line_candidates:
                line_l = line.lower()
                if not any(term in line_l for term in label_terms):
                    continue
                numbers = re.findall(pattern, line)
                if not numbers:
                    continue
                for num_txt in numbers[:2]:
                    value = self._parse_numeric_token(num_txt)
                    if value is None:
                        continue
                    if 1900 <= value <= 2100 and len(numbers) > 1:
                        # Skip year-like tokens when a line also contains real numeric amounts.
                        continue
                    key = (self._row_source_label(row), tuple(sorted(label_terms)), f"{value:.6f}")
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(
                        {
                            "value": value,
                            "value_text": num_txt.strip(),
                            "line": re.sub(r"\s+", " ", line).strip()[:220],
                            "source": row,
                        }
                    )
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def _derive_vat_turnover_total(self, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for row in rows:
            text = self._row_primary_text(row)
            quarter_matches = re.findall(r"q[1-4]\s+20\d{2}\s+([0-9\.,]+)", text.lower())
            values = []
            for token in quarter_matches:
                parsed = self._parse_numeric_token(token)
                if parsed is not None:
                    values.append(parsed)
            if len(values) >= 2:
                total = sum(values)
                return {
                    "value": total,
                    "value_text": f"{total:,.0f}",
                    "line": "Afgeleid uit kwartaalomzet in btw-overzicht",
                    "source": row,
                }
        return None

    def _contains_missing_signal(self, text: str) -> bool:
        signals = [
            "ontbreekt", "ontbrekend", "niet aangeleverd", "onduidelijk", "nog te ontvangen",
            "geen specificatie", "geen onderbouwing", "niet volledig", "niet gespecificeerd",
        ]
        t = (text or "").lower()
        return any(sig in t for sig in signals)

    def _format_eur(self, value: float) -> str:
        sign = "-" if value < 0 else ""
        n = abs(value)
        s = f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        if s.endswith(",00"):
            s = s[:-3]
        return f"{sign}€{s}"

    def _parse_year_from_question(self, question: Optional[str]) -> Optional[int]:
        if not question:
            return None
        years = re.findall(r"\b(20\d{2})\b", question)
        return max(int(y) for y in years) if years else None

    def _extract_labelled_value_candidates(
        self,
        rows: List[Dict[str, Any]],
        labels: List[str],
        reject_labels: Optional[List[str]] = None,
        max_hits: int = 10,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()
        reject_labels = reject_labels or []
        num_pattern = r"(?:€|\$)?\s*\(?\d[\d\.,]*\)?"

        for row in rows:
            raw = self._row_primary_text(row)
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in re.split(r"[\n\r]+", raw) if ln and ln.strip()]
            if not lines:
                lines = [re.sub(r"\s+", " ", raw).strip()]
            for i, line in enumerate(lines):
                line_l = line.lower()
                if not any(lbl in line_l for lbl in labels):
                    continue
                if any(lbl in line_l for lbl in reject_labels):
                    continue
                window_parts = [line]
                for j in range(1, 4):
                    if i + j < len(lines):
                        window_parts.append(lines[i + j])
                window = " ".join(window_parts)
                line_years = [int(y) for y in re.findall(r"\b(20\d{2})\b", window)]
                nums = re.findall(num_pattern, window)
                parsed_nums: List[Tuple[str, float]] = []
                for tok in nums:
                    val = self._parse_numeric_token(tok)
                    if val is None:
                        continue
                    if 1900 <= val <= 2100 and len(nums) > 1:
                        continue
                    parsed_nums.append((tok.strip(), val))
                if not parsed_nums:
                    continue

                row_year = row.get("tax_year") if isinstance(row.get("tax_year"), int) else None
                for idx, (tok, val) in enumerate(parsed_nums[:2]):
                    year = line_years[idx] if idx < len(line_years) else row_year
                    key = (self._row_source_label(row), tok, year)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "value": val,
                            "value_text": tok,
                            "year": year,
                            "line": window[:260],
                            "source": row,
                        }
                    )
                    if len(candidates) >= max_hits:
                        return candidates
        return candidates

    def _derive_vat_turnover_from_quarters(
        self, rows: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for row in rows:
            text = self._row_primary_text(row)
            compact = re.sub(r"\s+", " ", text.lower())
            values_by_year_q: Dict[int, Dict[int, float]] = {}
            values_text_by_year_q: Dict[int, Dict[int, str]] = {}
            matches = re.findall(
                r"\bq([1-4])\b(?:\s*(20\d{2}))?(?:\s+[a-z\-]+){0,3}\s+([0-9][0-9\.,]{1,})",
                compact,
            )
            for q_txt, y_txt, val_txt in matches:
                val = self._parse_numeric_token(val_txt)
                if val is None:
                    continue
                year = int(y_txt) if y_txt else (row.get("tax_year") if isinstance(row.get("tax_year"), int) else None)
                if year is None:
                    continue
                q_num = int(q_txt)
                values_by_year_q.setdefault(year, {})[q_num] = val
                values_text_by_year_q.setdefault(year, {})[q_num] = val_txt

            if not values_by_year_q:
                continue
            year = max(values_by_year_q.keys(), key=lambda y: len(values_by_year_q[y]))
            q_map = values_by_year_q[year]
            if len(q_map) < 2:
                continue
            total = sum(q_map.values())
            ordered_quarters = [f"Q{q}" for q in sorted(q_map.keys())]
            return {
                "value": total,
                "value_text": f"{int(total)}",
                "year": year,
                "line": f"Afgeleid uit kwartaalomzet ({'+'.join(ordered_quarters)} waar beschikbaar).",
                "source": row,
                "derived_quarters": [values_text_by_year_q.get(year, {}).get(q, "") for q in sorted(q_map.keys())],
            }
        return None

    def _pick_best_candidate_for_year(
        self, candidates: List[Dict[str, Any]], preferred_year: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        if preferred_year is not None:
            same_year = [c for c in candidates if c.get("year") == preferred_year]
            if same_year:
                return max(same_year, key=lambda c: c["value"])
        known_year = [c for c in candidates if c.get("year") is not None]
        if known_year:
            return max(known_year, key=lambda c: (c["year"], c["value"]))
        return max(candidates, key=lambda c: c["value"])

    def _build_inconsistency_answer(self, search_results: List[Dict[str, Any]], question: Optional[str] = None) -> str:
        pnl_rows = [r for r in search_results if self._detect_row_kind(r) == "pnl"]
        vat_rows = [r for r in search_results if self._detect_row_kind(r) == "vat"]
        note_rows = [r for r in search_results if self._detect_row_kind(r) == "notes"]

        # Fallback only when no concept rows were found.
        if not pnl_rows:
            pnl_rows = [r for r in search_results if any(t in self._row_text(r) for t in ["winst-en-verlies", "resultatenrekening", "profit and loss", "p&l", "income statement"])]
        if not vat_rows:
            vat_rows = [r for r in search_results if any(t in self._row_text(r) for t in ["btw-overzicht", "omzetbelasting", "vat overview", "taxable turnover", "vat turnover"])]
        if not note_rows:
            note_rows = [r for r in search_results if any(t in self._row_text(r) for t in ["klantnotities", "notitie", "client notes", "memo"])]

        pnl_values = self._extract_labelled_value_candidates(
            pnl_rows,
            ["totale omzet", "netto-omzet", "omzet", "opbrengsten", "total revenue", "revenue", "net sales", "sales"],
            reject_labels=["btw", "vat", "voorbelasting", "te betalen btw", "vat payable"],
            max_hits=10,
        )
        vat_turnover_values = self._extract_labelled_value_candidates(
            vat_rows,
            ["btw-omzet", "belastbare omzet", "omzet hoog tarief", "omzet laag tarief", "omzet per kwartaal", "grondslag", "taxable turnover", "vat turnover", "total vat turnover"],
            reject_labels=["voorbelasting", "te betalen btw", "vat payable", "input vat", "verschuldigde btw"],
            max_hits=10,
        )
        vat_payable_values = self._extract_labelled_value_candidates(
            vat_rows,
            ["verschuldigde btw", "btw omzet", "voorbelasting", "te betalen btw", "vat payable", "input vat"],
            max_hits=6,
        )
        note_signals = self._extract_labelled_value_candidates(
            note_rows,
            ["omzet gestegen", "omzet gedaald", "ontbrekende factuur", "correctie", "periodeverschil", "nog niet verwerkt", "contante omzet", "contractomzet", "timingverschil", "omzet"],
            max_hits=4,
        )

        requested_year = self._parse_year_from_question(question)
        derived_vat = self._derive_vat_turnover_from_quarters(vat_rows)
        if not vat_turnover_values and derived_vat:
            vat_turnover_values = [derived_vat]

        pnl_pick = self._pick_best_candidate_for_year(pnl_values, requested_year)
        vat_pick = self._pick_best_candidate_for_year(vat_turnover_values, requested_year)

        # If both picks resolve to the same source while alternative concept rows exist,
        # prefer a cross-source comparison to reduce accidental within-document mismatches.
        if pnl_pick and vat_pick:
            src_a = self._row_source_label(pnl_pick["source"])
            src_b = self._row_source_label(vat_pick["source"])
            if src_a == src_b:
                alt_pnl = [c for c in pnl_values if self._row_source_label(c["source"]) != src_b]
                alt_vat = [c for c in vat_turnover_values if self._row_source_label(c["source"]) != src_a]
                if alt_pnl:
                    pnl_pick = self._pick_best_candidate_for_year(alt_pnl, requested_year) or pnl_pick
                if alt_vat:
                    vat_pick = self._pick_best_candidate_for_year(alt_vat, requested_year) or vat_pick

        issues: List[str] = []
        missing: List[str] = []
        current_idx = 1

        if pnl_pick and vat_pick:
            year_a = pnl_pick.get("year")
            year_b = vat_pick.get("year")
            if year_a is not None and year_b is not None and year_a != year_b:
                issues.append(
                    f"{current_idx}. Waarden lijken uit verschillende jaren te komen\n"
                    f"   - Bewijs A: Totale omzet {self._format_eur(pnl_pick['value'])} ({year_a}), bron: {self._row_source_label(pnl_pick['source'])}\n"
                    f"   - Bewijs B: Btw-omzet {self._format_eur(vat_pick['value'])} ({year_b}), bron: {self._row_source_label(vat_pick['source'])}\n"
                    "   - Waarom dit mogelijk inconsistent is: vergelijking tussen verschillende jaren is niet direct valide."
                )
            else:
                diff = pnl_pick["value"] - vat_pick["value"]
                label_b = "Btw-omzet / belastbare omzet volgens btw-overzicht"
                if vat_pick is derived_vat:
                    label_b = "Btw-omzet totaal (afgeleid: Q1+Q2+Q3+Q4)"
                issues.append(
                    f"{current_idx}. Omzet volgens winst-en-verliesrekening sluit mogelijk niet aan op btw-overzicht\n"
                    f"   - Bewijs A: Totale omzet volgens winst-en-verliesrekening: {self._format_eur(pnl_pick['value'])}, bron: {self._row_source_label(pnl_pick['source'])}\n"
                    f"   - Bewijs B: {label_b}: {self._format_eur(vat_pick['value'])}, bron: {self._row_source_label(vat_pick['source'])}\n"
                    f"   - Verschil: {self._format_eur(pnl_pick['value'])} - {self._format_eur(vat_pick['value'])} = {self._format_eur(diff)}\n"
                    "   - Waarom dit mogelijk inconsistent is: omzet en btw-grondslag sluiten niet direct op elkaar aan. Dit kan verklaarbaar zijn door vrijgestelde omzet, timingverschillen, correcties of ontbrekende facturen, maar moet handmatig worden gecontroleerd."
                )
            current_idx += 1
        else:
            if not pnl_pick:
                missing.append("- Totale omzet uit winst-en-verliesrekening niet gevonden.")
            if not vat_pick:
                missing.append("- Btw-omzet uit btw-overzicht niet gevonden.")
                if vat_payable_values:
                    missing.append("- Alleen btw-bedragen zoals te betalen btw/voorbelasting gevonden; deze zijn niet gebruikt als omzetvergelijking.")

        if note_signals:
            first = note_signals[0]
            issues.append(
                f"{current_idx}. Klantnotities bevatten een mogelijk verklarend signaal\n"
                f"   - Bewijs A: {first['line']}, bron: {self._row_source_label(first['source'])}\n"
                "   - Bewijs B: Controleer of dit signaal terugkomt in winst-en-verliesrekening en btw-overzicht\n"
                "   - Waarom dit mogelijk inconsistent is: notities wijzen op mogelijke correcties of timingverschillen."
            )
            current_idx += 1
        elif note_rows:
            missing.append("- Klantnotities bevatten geen concrete verklaring voor het verschil.")

        lines = [
            "Kort antwoord:",
            "Mogelijke inconsistenties voor handmatige controle.",
            "",
            "Mogelijke inconsistenties:",
        ]
        if issues:
            lines.append("\n".join(issues))
        else:
            lines.append("1. Geen harde inconsistentie vastgesteld op basis van de gevonden waarden.")

        if not issues:
            lines.extend([
                "",
                "Geen duidelijke inconsistentie gevonden:",
                "- Controle blijft nodig: niet alle benodigde vergelijkingswaarden zijn gevonden.",
            ])

        if missing:
            lines.extend([""] + missing)

        lines.extend([
            "",
            "Volgende stap:",
            "Vergelijk de verschillen met onderliggende facturen, btw-aangiften, contracten of correctieboekingen per dezelfde periode.",
            "",
            "Bronnen:",
            *self._collect_source_lines(search_results),
        ])
        return "\n".join(lines).strip()

    def _build_inconsistency_answer_verified(self, search_results: List[Dict[str, Any]]) -> str:
        return self._build_inconsistency_answer(search_results, question=None)
    def _build_advisory_points_answer(self, search_results: List[Dict[str, Any]]) -> str:
        hints = self._scan_workflow_hints("advisory_points", search_results, limit=12)
        theme_candidates: List[Tuple[str, str, str]] = []
        if any(h["term"] in {"omzet", "marge", "groei", "daling"} for h in hints):
            theme_candidates.append(
                ("Omzet- en margeontwikkeling", "Inzicht in omzet en marge helpt om winstgevendheid te sturen.", "Welke oorzaken liggen achter recente omzet- of margeschommelingen?")
            )
        if any(h["term"] in {"btw", "facturen", "omzet"} for h in hints):
            theme_candidates.append(
                ("BTW-aansluiting en factuuronderbouwing", "Afstemming tussen btw-overzicht en administratie verlaagt controlerisico.", "Zijn alle btw-relevante facturen en correcties volledig verwerkt?")
            )
        if any(h["term"] in {"liquiditeit", "kosten", "voorraad", "contract", "verzekering", "ontbreekt"} for h in hints):
            theme_candidates.append(
                ("Documentatie en risicobeheersing", "Ontbrekende stukken of onduidelijke afspraken vergroten operationeel en financieel risico.", "Welke documenten of afspraken ontbreken nog voor een volledig dossier?")
            )
        if len(theme_candidates) < 3:
            theme_candidates.extend(
                [
                    ("Kostenstructuur en efficiency", "Grip op kosten ondersteunt stabiele marges.", "Welke kostenposten zijn het sterkst gestegen en waarom?"),
                    ("Liquiditeit en verplichtingen", "Vooruitkijken op kasstromen voorkomt knelpunten.", "Welke verplichtingen drukken de komende maanden het meest op de liquiditeit?"),
                    ("Contract- en afhankelijkheidsrisico", "Contractvoorwaarden en afhankelijkheden bepalen risico en onderhandelingsruimte.", "Waar zitten de grootste contractuele risico’s of afhankelijkheden?"),
                ]
            )
        points = theme_candidates[:3]
        src_lines = self._collect_source_lines(search_results)
        src1 = src_lines[0][2:] if len(src_lines) > 0 else "bron onbekend"
        src2 = src_lines[1][2:] if len(src_lines) > 1 else src1
        src3 = src_lines[2][2:] if len(src_lines) > 2 else src2
        sources_per_point = [src1, src2, src3]

        lines = [
            "Kort antwoord:",
            "Hier zijn drie adviespunten om met de klant te bespreken.",
            "",
            "Adviespunten om met de klant te bespreken:",
        ]
        for idx, (title, why, question) in enumerate(points, start=1):
            lines.append(f"{idx}. {title}")
            lines.append("   - Waarom dit belangrijk is:")
            lines.append(f"   {why}")
            lines.append("   - Bewijs/bron:")
            lines.append(f"   {sources_per_point[idx - 1]}")
            lines.append("   - Vraag aan de klant:")
            lines.append(f"   {question}")
            lines.append("")
        lines.append("Belangrijke opmerking:")
        lines.append("Dit zijn voorbereidingspunten, geen definitief advies.")
        lines.append("")
        lines.append("Bronnen:")
        lines.extend(src_lines)
        return "\n".join(lines).strip()

    def _build_insurance_risk_answer(self, search_results: List[Dict[str, Any]]) -> str:
        hints = self._scan_workflow_hints("insurance_risk_check", search_results, limit=12)
        risk_items: List[Tuple[str, str, str]] = []

        asset_values = self._extract_named_values(search_results, ["voorraad", "inventaris", "activa", "bedrijfsmiddelen"], max_hits=2)
        insured_values = self._extract_named_values(search_results, ["verzekerde som", "dekking", "polis", "insurance coverage"], max_hits=2)
        if asset_values and insured_values:
            asset = asset_values[0]
            ins = insured_values[0]
            if ins["value"] < asset["value"]:
                diff = asset["value"] - ins["value"]
                risk_items.append(
                    (
                        "Mogelijke onderverzekering van activa/voorraad",
                        f"Voorraadwaarde {asset['value_text']} in {self._row_source_label(asset['source'])}; verzekerde dekking {ins['value_text']} in {self._row_source_label(ins['source'])}. Verschil: {diff:,.2f}.",
                        "Controleer of verzekerde som en actuele waarde op elkaar aansluiten.",
                    )
                )

        if any(h["term"] in {"contract", "aansprakelijkheid", "productaansprakelijkheid", "beroepsaansprakelijkheid"} for h in hints):
            src = next((h["source"] for h in hints if h["term"] in {"contract", "aansprakelijkheid", "productaansprakelijkheid", "beroepsaansprakelijkheid"}), self._row_source_label(search_results[0]))
            risk_items.append(("Aansprakelijkheidsrisico vanuit contracten", src, "Controleer aansprakelijkheidsdekking, limieten en uitsluitingen."))
        if any(h["term"] in {"cyber", "klantdata"} for h in hints):
            src = next((h["source"] for h in hints if h["term"] in {"cyber", "klantdata"}), self._row_source_label(search_results[0]))
            risk_items.append(("Cyber- en datarisico", src, "Controleer cyberdekking, datalekrespons en eigen risico."))
        if any(h["term"] in {"transport"} for h in hints):
            src = next((h["source"] for h in hints if h["term"] == "transport"), self._row_source_label(search_results[0]))
            risk_items.append(("Transport- en leveringsrisico", src, "Controleer transportdekking en aansprakelijkheid tijdens vervoer."))

        dedup: List[Tuple[str, str, str]] = []
        seen = set()
        for item in risk_items:
            if item[0] in seen:
                continue
            seen.add(item[0])
            dedup.append(item)
        if not dedup:
            dedup.append(("Beperkte risicodekking zichtbaar in huidige context", self._row_source_label(search_results[0]), "Vraag polisvoorwaarden, limieten en uitsluitingen op."))

        lines = [
            "Kort antwoord:",
            "Dit zijn mogelijke verzekeringsrisico’s op basis van de beschikbare documenten.",
            "",
            "Mogelijke verzekeringsrisico’s:",
        ]
        for idx, (name, evidence, check) in enumerate(dedup[:3], start=1):
            lines.append(f"{idx}. {name}")
            lines.append("   - Bewijs:")
            lines.append(f"   {evidence}")
            lines.append("   - Waarom dit belangrijk is:")
            lines.append("   Onvoldoende of onduidelijke dekking kan leiden tot onverwachte financiële schade.")
            lines.append("   - Wat te controleren:")
            lines.append(f"   {check}")
            lines.append("")
        lines.append("Ontbrekende informatie:")
        lines.append("- Polisvoorwaarden met limieten en uitsluitingen")
        lines.append("- Actuele waardering van activa/voorraad per peildatum")
        lines.append("")
        lines.append("Bronnen:")
        lines.extend(self._collect_source_lines(search_results))
        return "\n".join(lines).strip()

    def _build_client_file_summary_answer(self, search_results: List[Dict[str, Any]]) -> str:
        text_blob = " ".join(self._row_text(r) for r in search_results[:6])
        hints = self._scan_workflow_hints("advisory_points", search_results, limit=8)
        business = "MKB-dossier met operationele en financiële bronstukken." if text_blob else "Onvoldoende context in opgehaalde bronnen."
        if "software" in text_blob or "it" in text_blob:
            business = "Bedrijfsactiviteiten lijken deels software/IT-gerelateerd."
        elif "voorraad" in text_blob or "inventory" in text_blob:
            business = "Bedrijfsactiviteiten bevatten waarschijnlijk voorraad- of handelscomponenten."

        financial_points = []
        if any(t in text_blob for t in ["omzet", "revenue"]):
            financial_points.append("- Omzetinformatie is aanwezig in de opgehaalde context.")
        if any(t in text_blob for t in ["kosten", "expenses"]):
            financial_points.append("- Kosteninformatie is aanwezig, controle op onderbouwing blijft nodig.")
        if any(t in text_blob for t in ["cash", "bank", "liquiditeit"]):
            financial_points.append("- Liquiditeitssignalen zijn zichtbaar in bank/cash-gerelateerde informatie.")
        if not financial_points:
            financial_points.append("- Beperkte financiële detailinformatie in de huidige retrievalset.")

        tax_points = []
        if any(t in text_blob for t in ["btw", "vat", "omzetbelasting"]):
            tax_points.append("- BTW/VAT informatie is aanwezig en bruikbaar voor aansluiting.")
        if any(t in text_blob for t in ["tax", "belasting"]):
            tax_points.append("- Fiscale context is aanwezig, maar detailniveau verschilt per document.")
        if not tax_points:
            tax_points.append("- Geen duidelijke fiscale details in de huidige topresultaten.")

        risks = []
        if any(t in text_blob for t in ["contract", "agreement"]):
            risks.append("- Contractverplichtingen kunnen impact hebben op omzetmoment en risico.")
        if any(t in text_blob for t in ["verzekering", "polis", "insurance"]):
            risks.append("- Verzekeringsdekking moet worden vergeleken met activiteiten en activa.")
        if not risks:
            risks.append("- Aanvullende bronstukken nodig om operationele risico’s beter te beoordelen.")

        missing = [
            "- Volledige bronset voor facturen/bankafschriften per periode",
            "- Expliciete aansluiting tussen jaarrekeningposten en btw-overzicht",
        ]
        if any(h["term"] == "ontbreekt" for h in hints):
            missing.append("- In de opgehaalde context staat expliciet dat informatie ontbreekt; verifieer welke stukken nog ontbreken.")

        follow_up = [
            "- Welke posten verschillen tussen interne notities en financiële overzichten?",
            "- Zijn alle contracten en polisvoorwaarden actueel opgenomen in het dossier?",
        ]

        return (
            "Samenvatting klantdossier:\n\n"
            "Bedrijfsactiviteit:\n"
            f"{business}\n\n"
            "Financiële punten:\n"
            + "\n".join(financial_points)
            + "\n\nBelasting-/btw-punten:\n"
            + "\n".join(tax_points)
            + "\n\nRisico’s of aandachtspunten:\n"
            + "\n".join(risks)
            + "\n\nOntbrekende informatie:\n"
            + "\n".join(missing)
            + "\n\nVervolgvragen:\n"
            + "\n".join(follow_up)
            + "\n\nBronnen:\n"
            + "\n".join(self._collect_source_lines(search_results))
        ).strip()

    def _is_calc_intent_fast(self, question: str) -> bool:
        q = (question or "").lower()
        if not q:
            return False
        calc_terms = [
            "calculate", "ratio", "margin", "growth", "cogs", "cost of revenue",
            "roe", "roa", "eps", "free cash flow", "debt to equity", "current ratio",
            "operating income", "net income", "gross profit", "ebitda",
            "bereken", "berekenen", "omzetgroei", "marge", "verhouding", "groei",
        ]
        return any(term in q for term in calc_terms)

    def _try_formula_registry_calculation(
        self,
        question: str,
        base_results: List[Dict[str, Any]],
        corpus_type: Optional[str],
        jurisdiction: Optional[str],
        tax_year: Optional[int],
        entity_type: Optional[str],
        client_name: Optional[str],
        document_type: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        detect_started = perf_counter()
        is_calc_question = self._is_calc_intent_fast(question)
        detect_ms = (perf_counter() - detect_started) * 1000
        if not is_calc_question:
            return None

        lookup_started = perf_counter()
        formulas = self.formula_registry.find_by_question(question, max_results=3)
        _ = (perf_counter() - lookup_started) * 1000
        if not formulas:
            return None

        compute_started = perf_counter()
        formula = formulas[0]
        constraints = self._extract_question_constraints(question)
        company_targets = constraints.get("companies", [])

        if len(company_targets) >= 2 and self._is_compare_question(question):
            result = self._calculate_formula_for_multiple_companies(
                question=question,
                formula=formula,
                company_targets=company_targets[:2],
                base_results=base_results,
                corpus_type=corpus_type,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                constraints=constraints,
            )
            _ = (perf_counter() - compute_started) * 1000
            return result

        company_target = company_targets[0] if company_targets else None
        calc = self._calculate_formula_for_context(
            formula=formula,
            question=question,
            base_results=base_results,
            corpus_type=corpus_type,
            jurisdiction=jurisdiction,
            tax_year=tax_year,
            entity_type=entity_type,
            client_name=client_name,
            document_type=document_type,
            constraints=constraints,
            company_target=company_target,
        )
        _ = (perf_counter() - compute_started) * 1000
        if not calc:
            return None
        return calc

    def _calculate_formula_for_multiple_companies(
        self,
        question: str,
        formula: FormulaDefinition,
        company_targets: List[str],
        base_results: List[Dict[str, Any]],
        corpus_type: Optional[str],
        jurisdiction: Optional[str],
        tax_year: Optional[int],
        entity_type: Optional[str],
        client_name: Optional[str],
        document_type: Optional[str],
        constraints: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        sections: List[str] = []
        all_sources: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for company in company_targets:
            company_calc = self._calculate_formula_for_context(
                formula=formula,
                question=question,
                base_results=base_results,
                corpus_type=corpus_type,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                constraints={**constraints, "companies": [company]},
                company_target=company,
            )
            if not company_calc:
                sections.append(f"{company.title()}:\nMetric could not be calculated from retrieved evidence.")
                continue
            sections.append(f"{company.title()}:\n{company_calc['answer']}")
            all_sources.extend(company_calc.get("sources", []))
            warnings.extend(company_calc.get("warnings", []))

        answer = "\n\n".join(sections).strip()
        return {"answer": answer, "sources": all_sources[:8], "warnings": warnings}

    def _calculate_formula_for_context(
        self,
        formula: FormulaDefinition,
        question: str,
        base_results: List[Dict[str, Any]],
        corpus_type: Optional[str],
        jurisdiction: Optional[str],
        tax_year: Optional[int],
        entity_type: Optional[str],
        client_name: Optional[str],
        document_type: Optional[str],
        constraints: Dict[str, List[str]],
        company_target: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        pool = self._deduplicate_results(list(base_results), limit=18)
        if company_target:
            targeted = [r for r in pool if self._row_mentions_entity(r, company_target)]
            if targeted:
                pool = targeted
        if not pool:
            return None

        requested_year = tax_year or self._pick_requested_year(constraints)
        resolved, missing, derived_notes = self._resolve_formula_inputs(
            formula=formula,
            pool=pool,
            requested_year=requested_year,
            constraints=constraints,
            company_target=company_target,
        )

        if missing:
            enriched_pool = self._build_calculation_pool(
                question=question,
                formula=formula,
                base_results=pool,
                corpus_type=corpus_type,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                company_target=company_target,
            )
            if enriched_pool and len(enriched_pool) > len(pool):
                resolved, missing, derived_notes = self._resolve_formula_inputs(
                    formula=formula,
                    pool=enriched_pool,
                    requested_year=requested_year,
                    constraints=constraints,
                    company_target=company_target,
                )
                pool = enriched_pool

        if missing:
            has_target_doc = bool(company_target and any(self._row_mentions_entity(r, company_target) for r in pool))
            answer = self._format_incomplete_calculation(
                formula=formula,
                metric_name=formula.name,
                resolved=resolved,
                missing=missing,
                has_target_doc=has_target_doc,
            )
            sources = self._collect_sources_from_values(resolved)
            return {"answer": answer, "sources": sources, "warnings": derived_notes}

        variables = {k: v["normalized_value"] for k, v in resolved.items()}
        try:
            result_value = CalculationSandbox.evaluate(formula.expression, variables)
        except CalculationSandboxError:
            return None

        answer = self._format_successful_calculation(
            formula=formula,
            metric_name=formula.name,
            resolved=resolved,
            result_value=result_value,
            derived_notes=derived_notes,
            requested_year=requested_year,
        )
        sources = self._collect_sources_from_values(resolved)
        return {"answer": answer, "sources": sources, "warnings": derived_notes}

    def _build_calculation_pool(
        self,
        question: str,
        formula: FormulaDefinition,
        base_results: List[Dict[str, Any]],
        corpus_type: Optional[str],
        jurisdiction: Optional[str],
        tax_year: Optional[int],
        entity_type: Optional[str],
        client_name: Optional[str],
        document_type: Optional[str],
        company_target: Optional[str],
    ) -> List[Dict[str, Any]]:
        pool = list(base_results)
        boosts = [
            "consolidated statements of operations",
            "consolidated statements of income",
            "consolidated balance sheets",
            "consolidated statements of cash flows",
            "revenue",
            "cost of revenue",
            "gross profit",
            "operating income",
            "net income",
            "total assets",
            "total liabilities",
            "stockholders equity",
            "shareholders equity",
            "current assets",
            "current liabilities",
            "cash and cash equivalents",
            "net cash provided by operating activities",
        ]
        formula_terms = [formula.name] + formula.aliases + [
            label
            for labels in formula.variable_labels.values()
            for label in labels
        ]
        query = " ".join([question, company_target or "", *boosts[:6], *formula_terms[:8]])
        extra = self.es_client.search(
            query=query,
            limit=12,
            enable_fuzzy=True,
            jurisdiction=jurisdiction,
            tax_year=tax_year,
            entity_type=entity_type,
            client_name=client_name,
            document_type=document_type,
            corpus_type=corpus_type,
            use_vector=False,
        )
        pool.extend(extra)
        pool = self._deduplicate_results(pool, limit=18)

        if company_target:
            targeted = [r for r in pool if self._row_mentions_entity(r, company_target)]
            if targeted:
                pool = targeted
        return pool

    def _resolve_formula_inputs(
        self,
        formula: FormulaDefinition,
        pool: List[Dict[str, Any]],
        requested_year: Optional[int],
        constraints: Dict[str, List[str]],
        company_target: Optional[str] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        resolve_started = perf_counter()
        value_extract_ms = 0.0
        resolved: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        derived_notes: List[str] = []
        lookup_cache: Dict[Tuple[str, Optional[int]], Optional[Dict[str, Any]]] = {}

        def get_cached_value(var_name: str, year: Optional[int]) -> Optional[Dict[str, Any]]:
            key = (var_name, year)
            if key in lookup_cache:
                return lookup_cache[key]
            value = self._find_value_for_variable(
                variable=var_name,
                formula=formula,
                pool=pool,
                desired_year=year,
                company_target=company_target,
            )
            lookup_cache[key] = value
            return value

        for variable in formula.variables:
            desired_year = self._desired_year_for_variable(variable, requested_year, constraints, pool)
            extract_started = perf_counter()
            value_meta = get_cached_value(variable, desired_year)
            value_extract_ms += (perf_counter() - extract_started) * 1000
            if value_meta:
                resolved[variable] = value_meta
                continue

            derive_started = perf_counter()
            derived = self._derive_missing_variable(
                variable=variable,
                formula=formula,
                resolved=resolved,
                pool=pool,
                desired_year=desired_year,
                company_target=company_target,
                lookup_cache=lookup_cache,
            )
            value_extract_ms += (perf_counter() - derive_started) * 1000
            if derived:
                resolved[variable] = derived
                derived_notes.append(f"Derived {variable} using {derived['derivation']}.")
                continue
            missing.append(variable)

        _ = (perf_counter() - resolve_started) * 1000
        return resolved, missing, derived_notes

    def _find_value_for_variable(
        self,
        variable: str,
        formula: FormulaDefinition,
        pool: List[Dict[str, Any]],
        desired_year: Optional[int],
        company_target: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        extracted = self.value_extractor.resolve_best_value(
            variable=variable,
            formula=formula,
            rows=pool,
            desired_year=desired_year,
            company_target=company_target,
        )
        if not extracted:
            return None
        return {
            "variable": extracted.variable,
            "display_label": extracted.display_label,
            "raw_value": extracted.raw_value,
            "normalized_value": extracted.normalized_value,
            "value_text": extracted.value_text,
            "unit_label": extracted.unit_label,
            "year": extracted.year,
            "source": extracted.source,
            "source_label": extracted.source_label,
            "line_text": extracted.line_text,
            "match_score": extracted.match_score,
            "derived": extracted.derived,
        }

    def _derive_missing_variable(
        self,
        variable: str,
        formula: FormulaDefinition,
        resolved: Dict[str, Dict[str, Any]],
        pool: List[Dict[str, Any]],
        desired_year: Optional[int],
        company_target: Optional[str] = None,
        lookup_cache: Optional[Dict[Tuple[str, Optional[int]], Optional[Dict[str, Any]]]] = None,
    ) -> Optional[Dict[str, Any]]:
        rule = (formula.derived_inputs or {}).get(variable)
        if not rule:
            return None
        expression = str(rule.get("expression", "")).strip()
        dep_vars = list(rule.get("variables", []))
        if not expression or not dep_vars:
            return None

        local_values: Dict[str, float] = {}
        source_parts: List[str] = []
        for dep in dep_vars:
            dep_value = resolved.get(dep)
            if dep_value is None and lookup_cache is not None:
                dep_value = lookup_cache.get((dep, desired_year))
            if dep_value is None:
                dep_value = self._find_value_for_variable(
                    dep,
                    formula,
                    pool,
                    desired_year,
                    company_target=company_target,
                )
                if lookup_cache is not None:
                    lookup_cache[(dep, desired_year)] = dep_value
            if not dep_value:
                return None
            resolved[dep] = dep_value
            local_values[dep] = dep_value["normalized_value"]
            source_parts.append(dep_value["source_label"])

        try:
            derived_val = CalculationSandbox.evaluate(expression, local_values)
        except CalculationSandboxError:
            return None

        return {
            "variable": variable,
            "display_label": variable.replace("_", " "),
            "raw_value": derived_val,
            "normalized_value": derived_val,
            "value_text": f"{derived_val:,.4f}".rstrip("0").rstrip("."),
            "unit_label": "derived",
            "year": desired_year,
            "source": resolved[dep_vars[0]]["source"],
            "source_label": "; ".join(sorted(set(source_parts))),
            "line_text": f"Derived using {expression}",
            "match_score": 0.0,
            "derived": True,
            "derivation": expression,
        }

    def _format_successful_calculation(
        self,
        formula: FormulaDefinition,
        metric_name: str,
        resolved: Dict[str, Dict[str, Any]],
        result_value: float,
        derived_notes: List[str],
        requested_year: Optional[int],
    ) -> str:
        lines = []
        formula_expression = formula.expression
        if len(formula.variables) == 1:
            only_var = formula.variables[0]
            only_meta = resolved.get(only_var, {})
            if only_meta.get("derived") and only_meta.get("derivation"):
                formula_expression = str(only_meta.get("derivation"))
            else:
                formula_expression = f"Direct line item: {only_var}"
        lines.append("Metric:")
        lines.append(metric_name)
        lines.append("")
        lines.append("Formula:")
        lines.append(formula_expression)
        lines.append("")
        lines.append("Values used:")
        for var in formula.variables:
            meta = resolved[var]
            derived_tag = " [derived]" if meta.get("derived") else " [direct]"
            unit_text = f" [{meta.get('unit_label')}]" if meta.get("unit_label") and meta.get("unit_label") != "units" else ""
            lines.append(
                f"- {var}: {meta['value_text']}{unit_text}{derived_tag}, from {meta['source_label']}"
            )
        if derived_notes:
            lines.append("")
            lines.append("Derived values, if any:")
            for note in derived_notes:
                lines.append(f"- {note}")
        lines.append("")
        lines.append("Calculation:")
        substituted = formula_expression
        for var in formula.variables:
            substituted = re.sub(rf"\b{re.escape(var)}\b", str(resolved[var]["normalized_value"]), substituted)
        for var, meta in resolved.items():
            substituted = re.sub(rf"\b{re.escape(var)}\b", str(meta["normalized_value"]), substituted)
        lines.append(substituted)
        lines.append("")
        lines.append("Result:")
        if formula.output_type == "percentage" or formula.unit == "%":
            lines.append(f"{result_value:,.4f}%")
        else:
            lines.append(f"{result_value:,.4f}".rstrip("0").rstrip("."))
        lines.append("")
        lines.append("Short interpretation:")
        if requested_year:
            lines.append(f"The value was calculated from the requested period/year ({requested_year}) using the retrieved financial statement evidence.")
        else:
            lines.append("The value was calculated from the latest available evidence found in the retrieved financial statements.")
        lines.append("")
        lines.append("Sources:")
        for src in self._collect_sources_from_values(resolved):
            lines.append(f"- {src['filename']}, page {src.get('page', 'unknown')}")
        return "\n".join(lines).strip()

    def _format_incomplete_calculation(
        self,
        formula: FormulaDefinition,
        metric_name: str,
        resolved: Dict[str, Dict[str, Any]],
        missing: List[str],
        has_target_doc: bool = False,
    ) -> str:
        lines = []
        lines.append("Metric:")
        lines.append(metric_name)
        lines.append("")
        lines.append("Formula:")
        lines.append(formula.expression)
        lines.append("")
        lines.append("Values found:")
        if resolved:
            for var, meta in resolved.items():
                lines.append(f"- {var}: {meta['value_text']}, from {meta['source_label']}")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("Missing values:")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Reason:")
        if has_target_doc:
            lines.append("The correct document was found, but the retrieved sections did not contain the values needed for this calculation.")
        else:
            lines.append("The calculation cannot be completed from the retrieved document evidence.")
        lines.append("")
        lines.append("Suggested next search:")
        lines.append("Search for consolidated statements of operations, consolidated balance sheets, cost of revenue, and gross profit labels.")
        return "\n".join(lines).strip()

    def _collect_sources_from_values(self, resolved: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for meta in resolved.values():
            row = meta.get("source") or {}
            key = (
                row.get("document_id"),
                row.get("filename"),
                row.get("page_number"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "document_id": row.get("document_id"),
                "filename": row.get("filename"),
                "title": row.get("title", row.get("filename", "Unknown")),
                "score": row.get("score", 0.0),
                "category": row.get("category", "other"),
                "document_type_detail": (row.get("metadata") or {}).get("document_type_detail"),
                "page": row.get("page_number") or row.get("chunk_index"),
                "snippet": (row.get("snippet") or row.get("summary") or "")[:350],
                "jurisdiction": row.get("jurisdiction"),
                "tax_year": row.get("tax_year"),
                "entity_type": row.get("entity_type"),
                "client_name": row.get("client_name"),
                "section_reference": row.get("section_reference"),
                "corpus_type": row.get("corpus_type"),
            })
        return out

    def _desired_year_for_variable(
        self,
        variable: str,
        requested_year: Optional[int],
        constraints: Dict[str, List[str]],
        pool: List[Dict[str, Any]],
    ) -> Optional[int]:
        if variable.endswith("_current"):
            return requested_year or self._latest_year_in_pool(pool)
        if variable.endswith("_previous"):
            base = requested_year or self._latest_year_in_pool(pool)
            return (base - 1) if base else None
        return requested_year

    def _pick_requested_year(self, constraints: Dict[str, List[str]]) -> Optional[int]:
        years = constraints.get("years", [])
        if not years:
            return None
        try:
            return max(int(y) for y in years)
        except ValueError:
            return None

    def _latest_year_in_pool(self, pool: List[Dict[str, Any]]) -> Optional[int]:
        years = []
        for row in pool:
            ty = row.get("tax_year")
            if isinstance(ty, int):
                years.append(ty)
            years.extend(int(y) for y in re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", str(row.get("content", ""))[:2000]))
        return max(years) if years else None

    def _detect_unit_multiplier(self, text: str) -> Tuple[float, str]:
        t = (text or "").lower()
        if "in billions" in t or "in billion" in t:
            return 1_000_000_000.0, "billions"
        if "in millions" in t or "in million" in t:
            return 1_000_000.0, "millions"
        if "in thousands" in t or "in thousand" in t:
            return 1_000.0, "thousands"
        return 1.0, "units"

    def _extract_year_from_window(self, text: str) -> Optional[int]:
        years = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", text or "")
        if not years:
            return None
        return max(int(y) for y in years)

    def _parse_numeric_token(self, token: str) -> Optional[float]:
        if not token:
            return None
        t = token.strip().replace("$", "").replace("€", "").replace(" ", "")
        negative = False
        if t.startswith("(") and t.endswith(")"):
            negative = True
            t = t[1:-1]
        # Handle Dutch and US separators:
        # - 372.000 => 372000
        # - 372,000 => 372000
        # - 18,5 => 18.5
        # - 18.5 => 18.5
        if "." in t and "," in t:
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "").replace(",", ".")
            else:
                t = t.replace(",", "")
        elif "," in t:
            if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", t):
                t = t.replace(",", "")
            else:
                t = t.replace(",", ".")
        elif "." in t:
            if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", t):
                t = t.replace(".", "")
        try:
            value = float(t)
            return -value if negative else value
        except ValueError:
            return None

    def _row_mentions_entity(self, row: Dict[str, Any], entity: str) -> bool:
        searchable = " ".join([
            str(row.get("filename", "")),
            str(row.get("title", "")),
            str(row.get("source_name", "")),
            str(row.get("content", ""))[:1500],
        ]).lower()
        return entity.lower() in searchable

    def _is_compare_question(self, question: str) -> bool:
        q = (question or "").lower()
        return any(token in q for token in ["compare", "versus", " vs ", "difference between"])

    def _format_sources(self, results: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for idx in indices:
            if idx >= len(results):
                continue
            item = results[idx]
            sources.append({
                "document_id": item.get("document_id"),
                "filename": item.get("filename"),
                "title": item.get("title", item.get("filename", "Unknown")),
                "score": item.get("score", 0.0),
                "category": item.get("category", "other"),
                "document_type_detail": (item.get("metadata") or {}).get("document_type_detail"),
                "page": item.get("page_number") or item.get("chunk_index"),
                "snippet": (item.get("snippet") or item.get("summary") or "")[:350],
                "jurisdiction": item.get("jurisdiction"),
                "tax_year": item.get("tax_year"),
                "entity_type": item.get("entity_type"),
                "client_name": item.get("client_name"),
                "section_reference": item.get("section_reference"),
                "corpus_type": item.get("corpus_type"),
            })
        return sources

    def _is_low_confidence(self, question: str, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True

        top_score = float(results[0].get("score", 0.0) or 0.0)
        significant_terms = [
            term.lower()
            for term in re.findall(r"[a-zA-Z]{4,}", question)
            if term.lower() not in {"what", "when", "where", "which", "that", "this", "with", "from", "into"}
        ]
        if not significant_terms:
            return False

        coverage_hits = 0
        for result in results[:3]:
            searchable = " ".join([
                str(result.get("title", "")),
                str(result.get("summary", "")),
                str(result.get("snippet", "")),
                str(result.get("content", ""))[:800],
            ]).lower()
            if any(term in searchable for term in significant_terms):
                coverage_hits += 1

        coverage_ratio = coverage_hits / min(3, len(results))
        return top_score < 0.2 and coverage_ratio < 0.34

    def _no_results_response(self, question: str) -> str:
        requirements = self._detect_context_requirements(question)
        missing = []
        if requirements.get("jurisdiction"):
            missing.append("jurisdiction")
        if requirements.get("tax_year"):
            missing.append("tax year")
        if requirements.get("entity_type"):
            missing.append("entity type")

        missing_text = ""
        if missing:
            missing_text = (
                "\nTo improve retrieval, please provide: "
                + ", ".join(missing)
                + "."
            )

        return (
            "I could not find this in the provided documents.\n"
            "Please upload relevant tax law, regulation, or financial records for this question."
            f"{missing_text}"
        )

    def _low_confidence_response(self) -> str:
        return (
            "I could not find this in the provided documents with enough confidence.\n"
            "The retrieved evidence is weak or only partially related. Please provide more specific source documents "
            "(for example the relevant jurisdiction, tax year, or client records) and try again."
        )

    def _detect_context_requirements(self, question: str) -> Dict[str, bool]:
        normalized = question.lower()
        requirements: Dict[str, bool] = {}
        for key, keywords in self.CONTEXT_KEYWORDS.items():
            requirements[key] = any(keyword in normalized for keyword in keywords)
        return requirements

    def _build_missing_context_message(
        self,
        context_requirements: Dict[str, bool],
        jurisdiction: Optional[str],
        tax_year: Optional[int],
        entity_type: Optional[str],
        search_results: List[Dict[str, Any]],
    ) -> Optional[str]:
        available_jurisdictions = {r.get("jurisdiction") for r in search_results if r.get("jurisdiction")}
        available_tax_years = {r.get("tax_year") for r in search_results if r.get("tax_year") is not None}
        available_entity_types = {r.get("entity_type") for r in search_results if r.get("entity_type")}

        missing_fields = []
        if context_requirements.get("jurisdiction") and not (jurisdiction or available_jurisdictions):
            missing_fields.append("jurisdiction")
        if context_requirements.get("tax_year") and not (tax_year or available_tax_years):
            missing_fields.append("tax year")
        if context_requirements.get("entity_type") and not (entity_type or available_entity_types):
            missing_fields.append("entity type")

        if not missing_fields:
            return None

        return (
            "I found related documents, but key context is missing for a reliable tax/accounting answer.\n"
            f"Please provide: {', '.join(missing_fields)}.\n"
            "I can then re-check the retrieved documents and answer with citations."
        )

    def _enforce_source_consistency(
        self,
        question: str,
        search_results: List[Dict[str, Any]],
        intent: str = "normal_qna",
    ) -> Tuple[List[Dict[str, Any]], Optional[str], List[str]]:
        constraints = self._extract_question_constraints(question, intent=intent)
        if not constraints:
            return search_results, None, []

        rescored: List[Tuple[float, Dict[str, Any], bool]] = []
        any_match = False
        for row in search_results:
            match_score, is_match = self._match_score(row, constraints)
            base_score = float(row.get("score", 0.0) or 0.0)
            combined = base_score + match_score
            rescored.append((combined, row, is_match))
            any_match = any_match or is_match

        rescored.sort(key=lambda x: x[0], reverse=True)
        sorted_results = [x[1] for x in rescored]

        if not any_match and intent in self.WORKFLOW_INTENTS:
            return sorted_results, None, ["Source consistency relaxed for advisor workflow intent."]

        if not any_match:
            expected = []
            if constraints.get("companies"):
                expected.append("company: " + ", ".join(constraints["companies"]))
            if constraints.get("tickers"):
                expected.append("ticker: " + ", ".join(constraints["tickers"]))
            if constraints.get("years"):
                expected.append("tax year/period: " + ", ".join(constraints["years"]))
            if constraints.get("doc_terms"):
                expected.append("document: " + ", ".join(constraints["doc_terms"]))
            expected_text = "; ".join(expected) if expected else "the requested scope"
            answer = (
                "1. Direct answer\n"
                "The retrieved evidence does not support the requested answer.\n\n"
                "2. Evidence from documents\n"
                "The retrieved sources do not match the company/document/period requested in your question.\n\n"
                "3. Important assumptions or missing information\n"
                f"Expected match: {expected_text}.\n\n"
                "4. Suggested next step\n"
                "Apply stricter filters or upload the correct source document for this company/period."
            )
            return sorted_results, answer, ["Source mismatch: retrieved documents do not match requested target."]

        notes: List[str] = []
        if self._is_single_entity_question(question, constraints):
            top_filename = (sorted_results[0].get("filename") or "").strip()
            if top_filename:
                narrowed = [r for r in sorted_results if (r.get("filename") or "").strip() == top_filename]
                if narrowed and len(narrowed) < len(sorted_results):
                    notes.append(f"Focused on one file for consistency: {top_filename}")
                    sorted_results = narrowed + [r for r in sorted_results if r not in narrowed]

        return sorted_results, None, notes

    def _extract_question_constraints(self, question: str, intent: str = "normal_qna") -> Dict[str, List[str]]:
        constraints: Dict[str, List[str]] = {}
        q = question or ""
        q_lower = q.lower()

        years = sorted(set(re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", q)))
        if years:
            constraints["years"] = years

        tickers: List[str] = []
        if intent in self.WORKFLOW_INTENTS:
            tickers = sorted(set(re.findall(r"\$([A-Z]{1,5})\b", q)))
        else:
            tickers = sorted(set(re.findall(r"\b[A-Z]{2,5}\b", q)))
        tickers = [t for t in tickers if t not in self.TICKER_STOPWORDS]
        if tickers:
            constraints["tickers"] = tickers

        company_hits: List[str] = []
        known_companies = [
            "palantir", "alphabet", "google", "microsoft", "apple", "amazon", "meta", "tesla",
        ]
        for name in known_companies:
            if re.search(rf"\b{re.escape(name)}\b", q_lower):
                company_hits.append(name)
        if company_hits:
            constraints["companies"] = sorted(set(company_hits))

        doc_terms: List[str] = []
        for term in ["earnings release", "annual report", "10-k", "10q", "income statement", "tax return"]:
            if term in q_lower:
                doc_terms.append(term)
        filename_mentions = re.findall(r"\b[\w\-.]+\.pdf\b", q_lower)
        doc_terms.extend(filename_mentions)
        if doc_terms:
            constraints["doc_terms"] = sorted(set(doc_terms))

        return constraints

    def _match_score(self, row: Dict[str, Any], constraints: Dict[str, List[str]]) -> Tuple[float, bool]:
        searchable = " ".join(
            [
                str(row.get("filename", "")),
                str(row.get("title", "")),
                str(row.get("source_name", "")),
                str(row.get("content", ""))[:2000],
            ]
        ).lower()
        score = 0.0
        matched = False

        for c in constraints.get("companies", []):
            if c in searchable:
                score += 3.0
                matched = True
        for t in constraints.get("tickers", []):
            if t.lower() in searchable:
                score += 3.0
                matched = True
        for term in constraints.get("doc_terms", []):
            if term in searchable:
                score += 2.0
                matched = True
        for y in constraints.get("years", []):
            if y in searchable or str(row.get("tax_year") or "") == y:
                score += 1.5
                matched = True

        return score, matched

    def _is_single_entity_question(self, question: str, constraints: Dict[str, List[str]]) -> bool:
        q = (question or "").lower()
        compare_words = ["compare", "versus", " vs ", "difference between"]
        is_compare = any(w in q for w in compare_words)
        entity_count = len(constraints.get("companies", [])) + len(constraints.get("tickers", []))
        return (not is_compare) and entity_count <= 1 and entity_count > 0

    def _collect_variable_sources(
        self,
        variables: Dict[str, Any],
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        found: Dict[str, Dict[str, str]] = {}
        if not isinstance(variables, dict):
            return found

        for var_name, raw_value in variables.items():
            candidate_texts = self._number_candidates(raw_value)
            for idx, item in enumerate(search_results, start=1):
                haystack = " ".join(
                    [
                        str(item.get("content", "")),
                        str(item.get("summary", "")),
                        str(item.get("snippet", "")),
                    ]
                )
                if any(token in haystack for token in candidate_texts):
                    source_label = f"Document {idx} ({item.get('filename', 'Unknown')}, page {item.get('page_number') or 'unknown'})"
                    found[var_name] = {
                        "value_text": str(raw_value),
                        "source_label": source_label,
                    }
                    break
        return found

    def _number_candidates(self, value: Any) -> List[str]:
        if isinstance(value, (int, float)):
            raw = f"{value}"
            compact_int = f"{int(value)}" if float(value).is_integer() else None
            formatted = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}".rstrip("0").rstrip(".")
            out = [raw, formatted]
            if compact_int:
                out.append(compact_int)
            return list(dict.fromkeys(out))
        return [str(value)]

    def _sanitize_missing_info_section(self, answer: str, notes: List[str]) -> str:
        if not answer:
            return answer
        cleaned = answer
        if re.search(r"Important assumptions or missing information\s*:?\s*(None|N/A|Not applicable)\b", cleaned, flags=re.IGNORECASE):
            if notes:
                cleaned = re.sub(
                    r"(Important assumptions or missing information\s*:?)\s*(None|N/A|Not applicable)\b\.?",
                    r"\1 " + " ".join(notes),
                    cleaned,
                    flags=re.IGNORECASE,
                )
            else:
                cleaned = re.sub(
                    r"\n?\s*3\.\s*Important assumptions or missing information\s*:?\s*(None|N/A|Not applicable)\b\.?",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
        return cleaned

    def list_available_models(self) -> List[str]:
        try:
            return ollama_list_models()
        except OllamaError as exc:
            logger.error(f"Error listing models: {exc}")
            return []


_llama_service: Optional[LlamaService] = None


def get_llama_service(model: str = settings.ollama_model) -> LlamaService:
    global _llama_service
    effective_model = settings.demo_ollama_model if settings.demo_mode and settings.demo_ollama_model else model
    if (
        _llama_service is None
        or getattr(_llama_service, "_service_version", None) != SERVICE_VERSION
        or _llama_service.model != effective_model
    ):
        _llama_service = LlamaService(model=effective_model)
    return _llama_service

