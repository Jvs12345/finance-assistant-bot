"""Llama/Ollama service for financial document Q&A."""

from typing import Dict, Any, List, Optional, Tuple
import re
import json
from time import perf_counter
from pathlib import Path

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
SERVICE_VERSION = 8


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
        "5. Do not present final certified accounting, tax, legal, or insurance advice.\n"
        "6. Include practical next steps.\n"
        "7. If the user asks in Dutch, answer in natural professional Dutch.\n"
        "8. If the user asks to summarize/extract from documents, provide an informational summary "
        "based on the retrieved text instead of refusing.\n"
    )
    INTENT_RETRIEVAL_TERMS = {
        "document_summary": [
            "summary", "purpose", "overview", "main sections", "obligations", "risks",
            "requirements", "conclusion", "introduction", "table of contents",
            "samenvatting", "doel", "inhoud", "hoofdonderdelen", "verplichtingen",
            "aandachtspunten", "risico", "conclusie", "introductie",
        ],
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
        "technical_requirements_check": [
            "technical requirements", "technische eisen", "verplichtingen", "responsibilities",
            "procedure", "steps", "systems", "components", "work permit", "veiligheidsinstructie",
        ],
        "risk_attention_check": [
            "risk", "risico", "hazard", "aandachtspunten", "incident", "near miss",
            "mitigation", "maatregelen", "safety controls",
        ],
        "ce_compliance_gap_check": [
            "ce", "ce-documentatie", "conformity", "conformiteit", "technical file",
            "technisch dossier", "risk assessment", "risicobeoordeling", "standards", "normen",
            "declaration of conformity", "verklaring van overeenstemming",
        ],
        "document_governance_check": [
            "metadata", "owner", "eigenaar", "revision", "revisie", "version", "versie", "status",
            "audit trail", "audit-proof", "tegenstrijdige instructies", "duplicated instructions",
            "standards without version", "normen zonder versie", "ocr noise", "ocr-ruis",
            "confidential", "vertrouwelijk", "change proposal", "wijzigingsvoorstel",
        ],
        "quotation_preparation": [
            "offerte", "quotation", "scope", "deliverables", "planning", "doorlooptijd",
            "randvoorwaarden", "acceptatiecriteria", "prijs", "kosten", "klantvraag",
        ],
        "ai_use_case_identification": [
            "ai use case", "ai-use-case", "repeterende taken", "documentintensief",
            "knowledge retrieval", "compliance check", "workflow automation",
        ],
        "use_case_prioritization": [
            "impact", "haalbaarheid", "feasibility", "databeschikbaarheid", "data availability",
            "prioriteren", "pilot value",
        ],
        "pilot_project_translation": [
            "pilotproject", "studententeam", "20 weken", "prototype", "stakeholders",
            "deliverables", "success metrics", "probleemdefinitie",
        ],
        "local_privacy_explanation": [
            "lokale ai", "local ai", "privacy", "data security", "on-premise", "on premise",
            "chatgpt", "claude", "documenten lokaal",
        ],
    }
    WORKFLOW_INTENTS = {
        "document_summary",
        "missing_info_check",
        "inconsistency_check",
        "advisory_points",
        "insurance_risk_check",
        "client_file_summary",
        "technical_requirements_check",
        "risk_attention_check",
        "ce_compliance_gap_check",
        "document_governance_check",
        "quotation_preparation",
        "ai_use_case_identification",
        "use_case_prioritization",
        "pilot_project_translation",
        "local_privacy_explanation",
    }
    SUMMARY_HEADING_TERMS = [
        "inhoud", "contents", "section", "sectie", "hoofdstuk", "chapter",
        "introduction", "introductie", "conclusion", "conclusie",
        "appendix", "bijlage", "scope", "purpose", "doel", "overview", "samenvatting",
    ]
    SUMMARY_REQUIREMENT_TERMS = [
        "must", "shall", "required", "requirement", "obligation", "risk", "mitigation", "action",
        "moet", "dient", "verplicht", "verplichting", "risico", "maatregel", "actie", "controle",
    ]
    SUMMARY_SAFETY_TERMS = [
        "veilig", "veiligheid", "gezondheid", "incident", "pbm", "ppe", "helm",
        "gehoorbescherming", "risicoanalyse", "work permit", "werkvergunning",
    ]
    SUMMARY_TERM_NORMALIZATION_MAP = {
        "V&G-plan": "veiligheids- en gezondheidsplan",
        "VGM": "veiligheid, gezondheid en milieu",
    }
    SUMMARY_DUTCH_CHAR_FIXES = {
        "geÄ«mplementeerd": "geïmplementeerd",
        "GeÄ«mplementeerd": "Geïmplementeerd",
        "geiÌˆmplementeerd": "geïmplementeerd",
        "geīmplementeerd": "geïmplementeerd",
        "Geīmplementeerd": "Geïmplementeerd",
    }
    TICKER_STOPWORDS = {
        "MKB", "SME", "BV", "NV", "ZZP", "BTW", "VAT", "KPI", "AI", "API",
        "P", "L", "IB", "VPB", "EU", "NL", "PDF", "OCR", "IFRS", "GAAP",
        "CEO", "CFO", "COO", "CTO", "CIO", "CPO",
    }
    CLIENT_DOC_HINTS = [
        "winst", "verlies", "balans", "btw", "klantnotities", "contract", "polis",
        "bank", "factuur", "invoice", "voorraad", "liquiditeit",
    ]
    REFERENCE_DOC_HINTS = ["referentie", "checklist", "belastingdienst", "guidance", "framework"]
    XAF_HINT_TERMS = [
        "xaf",
        "auditfile",
        "audit file",
        "xml auditfile",
        "taxregident",
        "btw nummer",
        "vat number",
        "fiscal year",
        "boekjaar",
        "grootboek",
        "tax records",
        "btw records",
        "vat records",
    ]
    NUMERIC_HINT_TERMS = [
        "cijfers",
        "cijfer",
        "bedrag",
        "bedragen",
        "omzet",
        "winst",
        "verlies",
        "kosten",
        "btw",
        "vat",
        "saldo",
        "total",
        "totaal",
    ]
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
            "why": "De voorraadwaardering beÃ¯nvloedt zowel het resultaat als de balans.",
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
            "why": "Contractvoorwaarden helpen om omzetmomenten, verplichtingen en risicoâ€™s te onderbouwen.",
            "next_step": "Controleer contracten op looptijd, prijsafspraken, en verplichtingen.",
        },
        {
            "key": "insurance_coverage",
            "label": "Verzekeringsinformatie",
            "terms": ["verzekering", "polis", "coverage", "insured amount", "aansprakelijkheid"],
            "why": "Verzekeringsinformatie is relevant voor risicobeoordeling en continuÃ¯teit.",
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
        "technical_requirements_check": [
            "eis", "vereiste", "verplicht", "moet", "dient", "procedure", "stap", "instructie",
            "norm", "richtlijn", "werkvergunning", "systeem", "component", "constructie",
        ],
        "risk_attention_check": [
            "risico", "gevaar", "incident", "bijna-ongeval", "aandachtspunt", "beheersmaatregel",
            "verboden", "waarschuwing", "veiligheid",
        ],
        "ce_compliance_gap_check": [
            "ce", "conformiteit", "technisch dossier", "risicobeoordeling",
            "verklaring van overeenstemming", "norm", "richtlijn", "testverslag",
        ],
        "document_governance_check": [
            "revisie", "versie", "version", "status", "eigenaar", "owner", "datum",
            "norm", "richtlijn", "confidential", "vertrouwelijk", "verboden", "toegestaan",
            "ocr", "scan", "bijlage", "audit",
        ],
        "quotation_preparation": [
            "offerte", "scope", "klantvraag", "eis", "levering", "planning",
            "randvoorwaarde", "prijs", "doorlooptijd", "acceptatie",
        ],
    }
    GENERIC_STOPWORDS_NL = {
        "de", "het", "een", "en", "van", "in", "op", "voor", "met", "te", "tot", "dat", "dit",
        "die", "is", "zijn", "wordt", "werd", "aan", "als", "bij", "door", "uit", "om", "naar",
        "of", "niet", "nog", "dan", "ook", "maar", "kan", "kunnen", "moet", "dient", "zal",
        "document", "bestaat", "staan", "worden",
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
                if isinstance(answer_text, str):
                    payload["answer"] = self._repair_text_artifacts(answer_text)
                    answer_text = payload["answer"]
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

            effective_corpus_type = corpus_type if corpus_type in {"uploaded", "existing"} else "uploaded"

            filters_used = {
                "corpus_type": effective_corpus_type,
                "document_type": document_type,
                "jurisdiction": jurisdiction,
                "tax_year": tax_year,
                "entity_type": entity_type,
                "client_name": client_name,
                "intent": detected_intent,
            }
            retrieval_query = self._build_intent_retrieval_query(question, detected_intent)
            retrieval_limit = max(1, min(max_context_docs or self.retrieval_top_k, 20))
            if detected_intent in {
                "document_summary",
                "technical_requirements_check",
                "risk_attention_check",
                "ce_compliance_gap_check",
                "document_governance_check",
                "quotation_preparation",
                "ai_use_case_identification",
                "use_case_prioritization",
                "pilot_project_translation",
            }:
                retrieval_limit = max(retrieval_limit, 18)

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
                corpus_type=effective_corpus_type,
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

            if self._is_xaf_existence_question(question):
                return finalize({
                    "answer": self._build_xaf_existence_answer(search_results),
                    "sources": self._format_sources(search_results, list(range(min(5, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                    "warnings": consistency_notes,
                })

            if self._is_xaf_amount_listing_question(question):
                return finalize({
                    "answer": self._build_xaf_amount_listing_answer(search_results, question),
                    "sources": self._format_sources(search_results, list(range(min(5, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results),
                    "filters_used": filters_used,
                    "warnings": consistency_notes,
                })

            if self._is_xaf_tax_record_count_question(question):
                return finalize({
                    "answer": self._build_xaf_tax_record_count_answer(search_results, question),
                    "sources": self._format_sources(search_results, list(range(min(5, len(search_results))))),
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
            if detected_intent == "document_summary":
                if self._is_summary_question(question):
                    warnings = list(consistency_notes)
                else:
                    warnings = list(consistency_notes) + [
                        "Vraag lijkt niet eenduidig als samenvatting geformuleerd; document-summary workflow toegepast."
                    ]
                consistency_notes = warnings
            if detected_intent in {
                "inconsistency_check",
                "advisory_points",
                "insurance_risk_check",
                "client_file_summary",
                "technical_requirements_check",
                "risk_attention_check",
                "ce_compliance_gap_check",
                "quotation_preparation",
                "ai_use_case_identification",
                "use_case_prioritization",
                "pilot_project_translation",
                "local_privacy_explanation",
            }:
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

            if detected_intent != "document_summary":
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
            if detected_intent == "document_summary":
                context_results = search_results[: min(len(search_results), max(14, self.final_context_chunks))]
                context, context_noise_ratio = self._build_document_summary_context(context_results)
                prompt = self._create_document_summary_prompt(
                    question=question,
                    context=context,
                    history=history,
                    noisy_context=context_noise_ratio >= 0.28,
                )
            else:
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
            if detected_intent == "document_summary":
                answer = self._post_process_document_summary_answer(
                    answer=answer,
                    question=question,
                    noisy_context=(context_noise_ratio >= 0.28),
                    search_results=context_results,
                )
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
        xaf_focus = self._is_xaf_focused_question(question)
        if xaf_focus:
            xaf_rows = self.es_client.search(
                query=question,
                limit=max(retrieval_limit, 8),
                enable_fuzzy=True,
                system_context=system_context,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                corpus_type=corpus_type,
                file_type="xaf",
                use_vector=False,
            )
            if xaf_rows:
                xaf_only = self._deduplicate_results(xaf_rows, limit=None)
                return self._prioritize_xaf_results_for_question(xaf_only, question)[:retrieval_limit]

        if detected_intent == "document_summary":
            return self._retrieve_document_summary_results(
                question=question,
                retrieval_limit=retrieval_limit,
                system_context=system_context,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                corpus_type=corpus_type,
            )

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
        uploaded_rows: List[Dict[str, Any]] = []
        existing_rows: List[Dict[str, Any]] = []
        if corpus_type in (None, "uploaded"):
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
        if corpus_type == "existing":
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

        merged_inputs = uploaded_rows + base_rows + existing_rows
        merged = self._deduplicate_results(merged_inputs, limit=None)
        prioritized = self._prioritize_advisor_results(merged)
        return prioritized[:retrieval_limit]

    def _build_document_summary_queries(self, question: str) -> List[str]:
        q = (question or "").strip()
        generic_nl = (
            "samenvatting doel inhoud hoofdonderdelen verplichtingen "
            "aandachtspunten risico conclusie introductie"
        )
        generic_en = (
            "summary purpose overview main sections obligations "
            "requirements risks conclusion introduction"
        )
        return [
            q,
            f"{q} {generic_nl}",
            f"{q} {generic_en}",
            generic_nl,
            generic_en,
        ]

    def _retrieve_document_summary_results(
        self,
        question: str,
        retrieval_limit: int,
        system_context: Optional[str],
        jurisdiction: Optional[str],
        tax_year: Optional[int],
        entity_type: Optional[str],
        client_name: Optional[str],
        document_type: Optional[str],
        corpus_type: Optional[str],
    ) -> List[Dict[str, Any]]:
        queries = self._build_document_summary_queries(question)
        merged_rows: List[Dict[str, Any]] = []
        per_query_limit = max(retrieval_limit, 14)

        for i, q in enumerate(queries):
            if not q:
                continue
            rows = self.es_client.search(
                query=q,
                limit=per_query_limit,
                enable_fuzzy=True,
                system_context=system_context,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type,
                corpus_type=corpus_type,
                use_vector=(i < 3),
            )
            merged_rows.extend(rows)

        deduped = self._deduplicate_results(merged_rows, limit=None)
        representative = self._select_representative_summary_rows(deduped, retrieval_limit=max(retrieval_limit, 16))
        return representative[: max(retrieval_limit, 14)]

    def _select_representative_summary_rows(
        self,
        rows: List[Dict[str, Any]],
        retrieval_limit: int,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        scored = self._rank_summary_rows(rows)
        if not scored:
            return self._deduplicate_results(rows, limit=retrieval_limit)

        filename_counts: Dict[str, int] = {}
        for _, row in scored:
            fname = str(row.get("filename", "")).strip()
            if not fname:
                continue
            filename_counts[fname] = filename_counts.get(fname, 0) + 1
        focus_filename = max(filename_counts, key=filename_counts.get) if filename_counts else ""
        focus_rows = [pair for pair in scored if str(pair[1].get("filename", "")).strip() == focus_filename]
        if len(focus_rows) >= 5:
            scored = focus_rows + [pair for pair in scored if pair not in focus_rows]

        heading_rows = [pair for pair in scored if self._has_heading_like_content(pair[1])]
        requirement_rows = [
            pair for pair in scored
            if any(term in self._row_text(pair[1]) for term in self.SUMMARY_REQUIREMENT_TERMS)
        ]

        selected: List[Dict[str, Any]] = []
        selected.extend([row for _, row in heading_rows[:4]])
        selected.extend([row for _, row in requirement_rows[:4]])
        selected.extend(self._select_summary_rows_by_page_ranges([row for _, row in scored], per_range=2))
        selected.extend([row for _, row in scored[: max(6, retrieval_limit)]])

        return self._deduplicate_summary_rows(selected, limit=retrieval_limit)

    def _rank_summary_rows(self, rows: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            text = self._clean_summary_text(self._row_primary_text(row))
            if len(text) < 60:
                continue
            score = float(row.get("score", 0.0) or 0.0)
            info_density = min(3.0, len(set(re.findall(r"\w+", text.lower()))) / 60.0)
            heading_boost = 1.8 if self._has_heading_like_content(row) else 0.0
            req_hits = sum(1 for term in self.SUMMARY_REQUIREMENT_TERMS if term in text.lower())
            req_boost = min(2.0, req_hits * 0.15)
            noise_penalty = self._summary_text_noise_ratio(text) * 3.0
            score = score + info_density + heading_boost + req_boost - noise_penalty
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _select_summary_rows_by_page_ranges(self, rows: List[Dict[str, Any]], per_range: int = 2) -> List[Dict[str, Any]]:
        with_pages = [r for r in rows if isinstance(r.get("page_number"), int)]
        if len(with_pages) < 4:
            return with_pages[: max(3, per_range * 2)]

        max_page = max(int(r.get("page_number")) for r in with_pages)
        if max_page <= 1:
            return with_pages[: max(3, per_range * 2)]

        early: List[Dict[str, Any]] = []
        middle: List[Dict[str, Any]] = []
        late: List[Dict[str, Any]] = []

        for row in with_pages:
            p = int(row.get("page_number"))
            ratio = p / max_page
            if ratio <= 0.33:
                early.append(row)
            elif ratio <= 0.66:
                middle.append(row)
            else:
                late.append(row)

        selected = early[:per_range] + middle[:per_range] + late[:per_range]
        return selected

    def _deduplicate_summary_rows(self, rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_page = set()
        seen_signature = set()

        for row in rows:
            page_key = (row.get("filename"), row.get("page_number"), row.get("chunk_index"))
            if page_key in seen_page:
                continue
            text = self._clean_summary_text(self._row_primary_text(row))
            signature = " ".join(re.findall(r"\w+", text.lower())[:20])
            if not signature:
                continue
            if signature in seen_signature:
                continue
            seen_page.add(page_key)
            seen_signature.add(signature)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    def _has_heading_like_content(self, row: Dict[str, Any]) -> bool:
        title = str(row.get("title", "") or "")
        content = self._clean_summary_text(self._row_primary_text(row))[:500]
        text = f"{title} {content}".lower()
        if any(term in text for term in self.SUMMARY_HEADING_TERMS):
            return True
        return bool(re.search(r"\b\d+(\.\d+){1,3}\b", text))

    def _summary_text_noise_ratio(self, text: str) -> float:
        if not text:
            return 1.0
        cleaned = text.strip()
        allowed_punct = {".", ",", ";", ":", "!", "?", "-", "(", ")", "/", "%", "$", "€"}
        bad = 0
        for ch in cleaned:
            if ch.isalnum() or ch.isspace() or ch in allowed_punct:
                continue
            bad += 1
        return bad / max(1, len(cleaned))

    def _clean_summary_text(self, text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"[\uFFFD]+", " ", cleaned)
        cleaned = re.sub(r"[|_=*~`]{2,}", " ", cleaned)
        cleaned = re.sub(r"[^\w\s\.,;:!\?\-\(\)\/%â‚¬$]", " ", cleaned)
        cleaned = re.sub(r"\bpage\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bpagina\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        words = re.findall(r"\w+", cleaned)
        if len(words) < 6:
            return ""
        alpha = sum(1 for ch in cleaned if ch.isalpha())
        if alpha / max(1, len(cleaned)) < 0.45:
            return ""
        return cleaned

    def _is_xaf_focused_question(self, question: str) -> bool:
        q = (question or "").lower()
        return any(term in q for term in self.XAF_HINT_TERMS) or self._is_xaf_tax_record_count_question(question)

    def _is_xaf_existence_question(self, question: str) -> bool:
        q = (question or "").lower()
        return any(
            token in q
            for token in [
                "is er een xaf",
                "bestaat er een xaf",
                "hebben we een xaf",
                "xaf bestand",
                "xaf-bestand",
                "welke xaf",
            ]
        )

    def _is_xaf_amount_listing_question(self, question: str) -> bool:
        q = (question or "").lower()
        asks_list = any(token in q for token in ["noem", "geef", "toon", "list", "show"])
        asks_amounts = any(
            token in q
            for token in [
                "bedrag",
                "bedragen",
                "amnt",
                "vatamnt",
                "vatperc",
                "btw",
                "cijfers",
                "waarden",
            ]
        )
        return asks_list and asks_amounts and self._is_xaf_focused_question(question)

    def _is_xaf_tax_record_count_question(self, question: str) -> bool:
        q = (question or "").lower()
        asks_count = any(token in q for token in ["aantal", "hoeveel", "count", "number of"])
        tax_terms = any(
            token in q
            for token in [
                "tax records",
                "tax record",
                "btw records",
                "btw record",
                "vat records",
                "vat record",
                "belasting records",
                "taxcode",
                "vatcode",
            ]
        )
        return asks_count and tax_terms

    def _extract_preferred_xaf_year(self, question: str) -> Optional[int]:
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question or "")]
        if not years:
            return None
        return max(years)

    def _candidate_xaf_paths(self, rows: List[Dict[str, Any]], preferred_year: Optional[int]) -> List[Path]:
        filenames = []
        for row in rows:
            if str(row.get("file_type", "")).lower() != "xaf":
                continue
            name = str(row.get("filename", "")).strip()
            if name:
                filenames.append(name)

        unique_names = []
        seen = set()
        for name in filenames:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        if preferred_year is not None:
            year_tag = str(preferred_year)
            preferred = [n for n in unique_names if year_tag in n]
            if preferred:
                unique_names = preferred + [n for n in unique_names if n not in preferred]

        roots = [Path("Source_files"), Path("Existing_files"), Path("/app/Source_files"), Path("/app/Existing_files")]
        candidates: List[Path] = []
        for name in unique_names:
            for root in roots:
                path = root / name
                if path.exists():
                    candidates.append(path)
                    break
        return candidates

    def _count_xaf_tax_records(self, xaf_path: Path) -> Optional[int]:
        try:
            from lxml import etree

            parser = etree.XMLParser(recover=True, huge_tree=True)
            tree = etree.parse(str(xaf_path), parser)
            root = tree.getroot()
            names = {"vat", "vatcode", "taxcode", "tax"}
            count = 0
            for element in root.iter():
                tag = str(element.tag)
                local_name = tag.split("}", 1)[1] if "}" in tag else tag
                if local_name.lower() in names:
                    count += 1
            return count
        except Exception:
            return None

    def _build_xaf_tax_record_count_answer(self, rows: List[Dict[str, Any]], question: str) -> str:
        preferred_year = self._extract_preferred_xaf_year(question)
        candidates = self._candidate_xaf_paths(rows, preferred_year)
        if not candidates:
            return (
                "1. Direct answer\n"
                "Ik kon geen XAF-bestand vinden om het aantal tax records te tellen.\n\n"
                "2. Evidence from documents\n"
                "Er zijn geen bruikbare XAF-bronnen gevonden in de huidige resultaten.\n\n"
                "3. Important assumptions or missing information\n"
                "Voor deze vraag is een lokaal beschikbaar .xaf-bestand nodig.\n\n"
                "4. Suggested next step\n"
                "Indexeer de XAF-bestanden opnieuw en stel de vraag opnieuw met een jaartal (bijv. 2025)."
            )

        for xaf_path in candidates:
            count = self._count_xaf_tax_records(xaf_path)
            if count is None:
                continue
            return (
                "1. Direct answer\n"
                f"Het aantal tax records is {count}.\n\n"
                "2. Evidence from documents\n"
                f"Geteld uit XAF-bestand: {xaf_path.name}.\n\n"
                "3. Important assumptions or missing information\n"
                "Tax records zijn geteld via VAT/tax-sectie in het XAF-bestand.\n\n"
                "4. Suggested next step\n"
                "Als je wilt, kan ik de records uitsplitsen per vatCode/taxCode."
            )

        return (
            "1. Direct answer\n"
            "Ik kon het aantal tax records niet betrouwbaar tellen uit de gevonden XAF-bronnen.\n\n"
            "2. Evidence from documents\n"
            "Er zijn XAF-bestanden gevonden, maar de tax-sectie kon niet worden uitgelezen.\n\n"
            "3. Important assumptions or missing information\n"
            "Het XAF-formaat of de inhoud kan afwijken van de verwachte structuur.\n\n"
            "4. Suggested next step\n"
            "Controleer het betreffende XAF-bestand en vraag eventueel om telling per specifieke sectie."
        )

    def _prioritize_xaf_results(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rescored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            score = float(row.get("score", 0.0) or 0.0)
            file_type = str(row.get("file_type", "")).lower()
            filename = str(row.get("filename", "")).lower()
            if file_type == "xaf":
                score += 4.0
            if "auditfile" in filename or filename.endswith(".xaf"):
                score += 1.5
            rescored.append((score, row))
        rescored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in rescored]

    def _prioritize_xaf_results_for_question(self, rows: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """
        Prioritize XAF rows, with an additional boost for numeric-heavy chunks
        when the question asks for figures/amounts.
        """
        wants_numbers = self._question_requests_numeric_data(question)
        rescored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            score = float(row.get("score", 0.0) or 0.0)
            file_type = str(row.get("file_type", "")).lower()
            filename = str(row.get("filename", "")).lower()
            text_blob = " ".join([
                str(row.get("content", "")),
                str(row.get("excerpt", "")),
                str(row.get("snippet", "")),
            ])

            if file_type == "xaf":
                score += 4.0
            if "auditfile" in filename or filename.endswith(".xaf"):
                score += 1.5

            if wants_numbers:
                numeric_hits = re.findall(r"(?:â‚¬|\$)?\s*\(?\d[\d\.,]*\)?", text_blob)
                score += min(3.0, len(numeric_hits) * 0.03)

            rescored.append((score, row))

        rescored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in rescored]

    def _question_requests_numeric_data(self, question: str) -> bool:
        q = (question or "").lower()
        return any(term in q for term in self.NUMERIC_HINT_TERMS)

    def _build_xaf_existence_answer(self, rows: List[Dict[str, Any]]) -> str:
        xaf_rows = [r for r in rows if str(r.get("file_type", "")).lower() == "xaf"]
        if not xaf_rows:
            return (
                "1. Direct answer\n"
                "Ik zie geen XAF-bestanden in de opgehaalde bronnen.\n\n"
                "2. Evidence from documents\n"
                "Er is geen bron met bestandstype XAF teruggevonden in deze resultaten.\n\n"
                "3. Important assumptions or missing information\n"
                "Dit oordeel is gebaseerd op de huidige zoekresultaten.\n\n"
                "4. Suggested next step\n"
                "Controleer of .xaf-bestanden zijn geÃ¯ndexeerd en vraag daarna opnieuw."
            )

        filenames = sorted({str(r.get("filename", "Unknown")) for r in xaf_rows})
        years_from_meta = sorted(
            {
                int(r.get("tax_year"))
                for r in xaf_rows
                if isinstance(r.get("tax_year"), int) and 1990 <= int(r.get("tax_year")) <= 2099
            }
        )
        years_from_names = []
        for name in filenames:
            years_from_names.extend(int(y) for y in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", name))
        years = sorted(set(years_from_meta + years_from_names))
        year_text = ", ".join(str(y) for y in years) if years else "onbekend"
        file_lines = "\n".join(f"- {name}" for name in filenames[:8])

        return (
            "1. Direct answer\n"
            "Ja, er zijn XAF-bestanden aanwezig in de opgehaalde bronnen.\n\n"
            "2. Evidence from documents\n"
            f"Gevonden XAF-bestanden:\n{file_lines}\n\n"
            "3. Important assumptions or missing information\n"
            f"Afgeleide boekjaren uit metadata: {year_text}.\n\n"
            "4. Suggested next step\n"
            "Stel een concrete cijfervraag (bijv. btw-bedragen, omzet of saldo), dan haal ik exacte waarden uit deze XAF-bestanden."
        )

    def _build_xaf_amount_listing_answer(self, rows: List[Dict[str, Any]], question: str) -> str:
        xaf_rows = [r for r in rows if str(r.get("file_type", "")).lower() == "xaf"]
        if not xaf_rows:
            return (
                "1. Direct answer\n"
                "Ik kon geen XAF-bronnen vinden om bedragen uit te halen.\n\n"
                "2. Evidence from documents\n"
                "Er zijn geen resultaten met bestandstype XAF in de huidige retrievalset.\n\n"
                "3. Important assumptions or missing information\n"
                "Deze extractie werkt alleen op opgehaalde XAF-chunks.\n\n"
                "4. Suggested next step\n"
                "Stel de vraag opnieuw met 'XAF' en eventueel jaartal (bijv. 2025)."
            )

        max_items = 10
        count_match = re.search(r"\b([1-9]|[1-4]\d|50)\b", question or "")
        if count_match:
            max_items = max(1, min(50, int(count_match.group(1))))

        requested_labels: List[str] = []
        q = (question or "").lower()
        if "amnt" in q:
            requested_labels.append("amnt")
        if "vatamnt" in q:
            requested_labels.append("vatAmnt")
        if "vatperc" in q:
            requested_labels.append("vatPerc")
        if not requested_labels:
            requested_labels = ["amnt", "vatAmnt", "vatPerc"]

        label_patterns = {
            "amnt": r"\bamnt\s*:\s*([0-9][0-9\.,]*)\b",
            "vatAmnt": r"\bvatamnt\s*:\s*([0-9][0-9\.,]*)\b",
            "vatPerc": r"\bvatperc\s*:\s*([0-9][0-9\.,]*)\b",
        }

        extracted: List[Dict[str, Any]] = []
        seen = set()
        for row in xaf_rows:
            text = str(row.get("content") or row.get("snippet") or "")
            if not text:
                continue
            compact = re.sub(r"\s+", " ", text)
            source_label = self._row_source_label(row)
            for label in requested_labels:
                pattern = label_patterns.get(label)
                if not pattern:
                    continue
                for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
                    value = match.group(1).strip()
                    key = (label.lower(), value, source_label)
                    if key in seen:
                        continue
                    seen.add(key)
                    extracted.append(
                        {
                            "label": label,
                            "value": value,
                            "source": source_label,
                        }
                    )
                    if len(extracted) >= max_items:
                        break
                if len(extracted) >= max_items:
                    break
            if len(extracted) >= max_items:
                break

        if not extracted:
            labels_text = ", ".join(requested_labels)
            return (
                "1. Direct answer\n"
                f"Ik kon geen waarden vinden voor: {labels_text} in de opgehaalde XAF-chunks.\n\n"
                "2. Evidence from documents\n"
                "De huidige chunks bevatten geen herkenbare velden in het formaat '<veld>: <waarde>' voor de gevraagde labels.\n\n"
                "3. Important assumptions or missing information\n"
                "Deze extractie zoekt specifiek op veldnamen zoals amnt, vatAmnt en vatPerc.\n\n"
                "4. Suggested next step\n"
                "Vraag opnieuw met expliciete labels en eventueel jaartal/chunk-context."
            )

        lines = []
        for idx, item in enumerate(extracted, 1):
            lines.append(f"{idx}. {item['label']}: {item['value']} ({item['source']})")

        return (
            "1. Direct answer\n"
            f"Hier zijn {len(extracted)} bedragen/waarden uit de XAF-bronnen:\n"
            + "\n".join(lines)
            + "\n\n2. Evidence from documents\n"
            "De waarden zijn direct geÃ«xtraheerd uit opgehaalde XAF-chunks op basis van veldnamen.\n\n"
            "3. Important assumptions or missing information\n"
            "De lijst is beperkt tot de huidige retrievalset en gevraagde labels.\n\n"
            "4. Suggested next step\n"
            "Als je wilt, kan ik deze ook groeperen per factuur, periode of btw-percentage."
        )

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
            chunk_index = result.get("chunk_index")
            section_reference = result.get("section_reference")
            jurisdiction = result.get("jurisdiction")
            tax_year = result.get("tax_year")
            entity_type = result.get("entity_type")

            # Prefer full chunk content so numeric values remain available for answering.
            content = (
                (result.get("content") or "").strip()
                or (result.get("summary") or "").strip()
                or (result.get("snippet") or "").strip()
            )
            content = self._compress_context_text(content)
            if len(content) > self.max_chars_per_chunk:
                content = content[: self.max_chars_per_chunk] + "..."

            if page_num:
                location = f"page {page_num}"
            elif chunk_index:
                location = f"chunk {chunk_index}"
            else:
                location = "page/chunk unknown"
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
        workflow_output = self._workflow_output_format(intent, question=question)

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

    def _workflow_output_format(self, intent: str, question: Optional[str] = None) -> str:
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
                "Mogelijke verzekeringsrisicoâ€™s:\n"
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
                "FinanciÃ«le punten:\n"
                "- [punt]\n\n"
                "Belasting-/btw-punten:\n"
                "- [punt]\n\n"
                "Risicoâ€™s of aandachtspunten:\n"
                "- [punt]\n\n"
                "Ontbrekende informatie:\n"
                "- [punt]\n\n"
                "Vervolgvragen:\n"
                "- [vraag]\n\n"
                "Bronnen:\n"
                "[bronnenlijst]"
            )
        if intent == "technical_requirements_check":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting van technische eisen en verplichtingen]\n\n"
                "Gevonden informatie:\n"
                "- [eis/verplichting], bron: [document/pagina]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [onduidelijk punt]\n\n"
                "Mogelijke vervolgstap:\n"
                "[concrete controle- of opvolgstap]"
            )
        if intent == "risk_attention_check":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting van risicoâ€™s en aandachtspunten]\n\n"
                "Gevonden informatie:\n"
                "- [risico/aandachtspunt], bron: [document/pagina]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [onduidelijk of niet onderbouwd risico]\n\n"
                "Mogelijke vervolgstap:\n"
                "[concrete beheersmaatregel of controle]"
            )
        if intent == "ce_compliance_gap_check":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting van CE-relevante bevindingen]\n\n"
                "Gevonden informatie:\n"
                "- [gevonden CE-eis/onderdeel], bron: [document/pagina]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [ontbrekend CE-onderdeel of onderbouwing]\n\n"
                "Mogelijke vervolgstap:\n"
                "[welke CE-documentatie nu op te vragen of te valideren]"
            )
        if intent == "document_governance_check":
            return (
                "Kort antwoord:\n"
                "[korte governance-samenvatting]\n\n"
                "Gevonden informatie:\n"
                "- [metadata/revisie/instructiebevinding], bron: [document/pagina]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [ontbrekende metadata, mogelijke tegenstrijdigheid, OCR-ruis of vertrouwelijkheidsrisico]\n\n"
                "Mogelijke vervolgstap:\n"
                "[concrete beheeractie voor volgende revisie of audit]"
            )
        if intent == "quotation_preparation":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting van offerte-input]\n\n"
                "Gevonden informatie:\n"
                "- [beschikbare offerte-input], bron: [document/pagina]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [ontbrekende scope/prijs/planning/randvoorwaarde]\n\n"
                "Mogelijke vervolgstap:\n"
                "[volgende stap om een betrouwbare eerste offerte op te stellen]"
            )
        if intent == "ai_use_case_identification":
            return (
                "Kort antwoord:\n"
                "[korte samenvatting van kansrijke AI-use-cases]\n\n"
                "Gevonden informatie:\n"
                "- [use-case + documentevidence]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [beperking/data-gat]\n\n"
                "Mogelijke vervolgstap:\n"
                "[hoe deze use-cases te valideren in een pilot]"
            )
        if intent == "use_case_prioritization":
            return (
                "Kort antwoord:\n"
                "[korte prioriteringsconclusie]\n\n"
                "Gevonden informatie:\n"
                "- [use-case score op impact/haalbaarheid/databeschikbaarheid]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [aannames of data-onzekerheden]\n\n"
                "Mogelijke vervolgstap:\n"
                "[welke use-case eerst te testen en waarom]"
            )
        if intent == "pilot_project_translation":
            return (
                "Kort antwoord:\n"
                "[korte pilotsamenvatting]\n\n"
                "Gevonden informatie:\n"
                "- [probleem, data, stakeholders, prototypekans]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [risicoâ€™s/afhankelijkheden voor pilot]\n\n"
                "Mogelijke vervolgstap:\n"
                "[concrete startstap voor een 20-weken pilot]"
            )
        if intent == "local_privacy_explanation":
            return (
                "Kort antwoord:\n"
                "[korte uitleg waarom lokaal relevant is]\n\n"
                "Gevonden informatie:\n"
                "- [lokale/verwerkingsgerelateerde bevinding]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [beperking of noodzakelijke menselijke controle]\n\n"
                "Mogelijke vervolgstap:\n"
                "[welke governance/controlemaatregel nu te borgen]"
            )
        if intent == "document_summary":
            if self._is_dutch_question(question or ""):
                return (
                    "Kort antwoord:\n"
                    "[2-4 zinnen over waar het document over gaat]\n\n"
                    "Belangrijkste onderdelen:\n"
                    "- [onderdeel 1]\n"
                    "- [onderdeel 2]\n"
                    "- [onderdeel 3]\n"
                    "- [onderdeel 4]\n\n"
                    "Belangrijke verplichtingen of aandachtspunten:\n"
                    "- [verplichting/aandachtspunt]\n"
                    "- [verplichting/aandachtspunt]\n\n"
                    "Mogelijke acties voor de gebruiker:\n"
                    "- [actie]\n"
                    "- [actie]"
                )
            return (
                "Short answer:\n"
                "[2-4 sentence overview of what the document is about]\n\n"
                "Main sections:\n"
                "- [section 1]\n"
                "- [section 2]\n"
                "- [section 3]\n"
                "- [section 4]\n\n"
                "Key obligations or attention points:\n"
                "- [obligation/attention point]\n"
                "- [obligation/attention point]\n\n"
                "Possible next actions:\n"
                "- [action]\n"
                "- [action]"
            )
        if self._is_dutch_question(question or ""):
            return (
                "Kort antwoord:\n"
                "[kort antwoord]\n\n"
                "Gevonden informatie:\n"
                "- [bewijs uit document]\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- [onzekerheid of ontbrekende data]\n\n"
                "Mogelijke vervolgstap:\n"
                "[praktische volgende stap]"
            )
        return (
            "1. Direct answer\n"
            "2. Evidence from documents\n"
            "3. Important assumptions or missing information\n"
            "4. Suggested next step"
        )

    def _is_summary_question(self, question: str) -> bool:
        q = self._normalize_question_text(question)
        if not q:
            return False
        summary_terms = ["samenvat", "samenvatting", "summary", "summarise", "summarize", "overview", "overzicht", "vat"]
        doc_terms = ["document", "bestand", "file", "pdf", "dossier"]
        about_patterns = [
            r"\bwaar gaat\b.*\b(document|bestand|file|pdf)\b.*\bover\b",
            r"\bwat staat er in\b.*\b(document|bestand|file|pdf)\b",
            r"\bwhat is\b.*\b(document|file|pdf)\b.*\babout\b",
            r"\bvat\b.*\b(document|bestand|file|pdf)\b.*\bsamen\b",
        ]
        if any(re.search(p, q) for p in about_patterns):
            return True
        return any(t in q for t in summary_terms) and any(d in q for d in doc_terms)

    def _normalize_question_text(self, question: str) -> str:
        q = (question or "").lower()
        q = re.sub(r"[^\w\s\-\?]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _is_dutch_question(self, question: str) -> bool:
        q = (question or "").lower()
        dutch_markers = [
            "wat", "waar", "welke", "geef", "maak", "samenvatting", "vat", "samen",
            "document", "bestand", "verplichtingen", "aandachtspunten",
        ]
        english_markers = ["what", "which", "give", "summarize", "summary", "document", "file"]
        dutch_hits = sum(1 for m in dutch_markers if m in q)
        english_hits = sum(1 for m in english_markers if m in q)
        return dutch_hits >= english_hits

    def _build_document_summary_context(self, search_results: List[Dict[str, Any]]) -> Tuple[str, float]:
        context_parts: List[str] = []
        used_chars = 0
        noisy = 0
        total = 0

        for i, result in enumerate(search_results, 1):
            filename = result.get("filename", "Unknown")
            page_num = result.get("page_number")
            chunk_index = result.get("chunk_index")
            content = self._clean_summary_text(
                (result.get("content") or "").strip()
                or (result.get("summary") or "").strip()
                or (result.get("snippet") or "").strip()
            )
            if not content:
                continue
            total += 1
            if self._summary_text_noise_ratio(content) >= 0.22:
                noisy += 1

            if len(content) > self.max_chars_per_chunk:
                content = content[: self.max_chars_per_chunk] + "..."

            if page_num:
                location = f"page {page_num}"
            elif chunk_index:
                location = f"chunk {chunk_index}"
            else:
                location = "page/chunk unknown"

            context_block = (
                f"--- Document {i} ---\n"
                f"Source: {filename}\n"
                f"Location: {location}\n"
                f"Extract:\n{content}\n"
            )
            projected = used_chars + len(context_block)
            if projected > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining < 350:
                    break
                context_parts.append(context_block[:remaining].rstrip() + "\n")
                break
            context_parts.append(context_block)
            used_chars = projected

        ratio = (noisy / total) if total else 0.0
        return "\n".join(context_parts), ratio

    def _create_document_summary_prompt(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        noisy_context: bool = False,
    ) -> str:
        history_text = ""
        if history:
            history_text = "Conversation History:\n"
            for msg in history[-3:]:
                history_text += f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}\n"
            history_text += "\n"

        structure = self._workflow_output_format("document_summary", question=question)
        noise_note = (
            "The extracted text appears noisy in places. Avoid copying broken OCR fragments.\n"
            if noisy_context
            else ""
        )
        return (
            "You are summarising a document from retrieved local snippets.\n"
            "Write a clear summary in the user's language.\n"
            "Do not copy broken OCR text directly.\n"
            "Clean up wording where possible, but do not invent facts.\n"
            "If snippets are incomplete or noisy, say so briefly.\n"
            "Focus on document purpose, main sections, obligations, risks, and practical next steps.\n"
            "Use only the provided snippets.\n"
            "Do not add a final 'Sources:' or 'Bronnen:' section in the answer body.\n"
            f"{noise_note}\n"
            f"Retrieved snippets:\n{context}\n\n"
            f"{history_text}User Question: {question}\n\n"
            f"Answer using this exact structure:\n{structure}\n\n"
            "Answer:"
        )

    def _post_process_document_summary_answer(
        self,
        answer: str,
        question: str,
        noisy_context: bool,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        cleaned = (answer or "").strip()
        cleaned = self._strip_summary_sources_section(cleaned)
        cleaned = self._normalize_summary_terminology(cleaned, question)
        cleaned = self._clean_summary_answer_lines(cleaned)
        if self._summary_answer_needs_fallback(cleaned, question):
            cleaned = self._build_document_summary_fallback_answer(search_results or [], question)
        cleaned = self._trim_summary_tail_to_word_boundary(cleaned)
        if noisy_context:
            cleaned = self._ensure_summary_noise_note(cleaned, question)
        return cleaned

    def _clean_summary_answer_lines(self, answer: str) -> str:
        lines = answer.splitlines()
        kept: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            if self._summary_text_noise_ratio(stripped) > 0.30 and len(stripped) < 80:
                continue
            kept.append(line)
        cleaned = "\n".join(kept)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _normalize_summary_terminology(self, answer: str, question: str) -> str:
        text = answer or ""

        for source, target in self.SUMMARY_DUTCH_CHAR_FIXES.items():
            text = text.replace(source, target)

        for source, target in self.SUMMARY_TERM_NORMALIZATION_MAP.items():
            text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.IGNORECASE)

        if self._summary_has_safety_context(text, question):
            text = re.sub(
                r"\bPBM\s*\(\s*Preventief\s+Beheer\s*\)",
                "PBM (persoonlijke beschermingsmiddelen)",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bPPE\b",
                "PBM (persoonlijke beschermingsmiddelen)",
                text,
                flags=re.IGNORECASE,
            )
            if re.search(r"\bPBM\b(?!\s*\()", text, flags=re.IGNORECASE):
                text = re.sub(
                    r"\bPBM\b(?!\s*\()",
                    "PBM (persoonlijke beschermingsmiddelen)",
                    text,
                    count=1,
                    flags=re.IGNORECASE,
                )

        if re.search(r"\bBTW\b", text):
            if self._is_dutch_question(question):
                text = re.sub(
                    r"\bBTW\b",
                    "btw (belasting over de toegevoegde waarde)",
                    text,
                    count=1,
                )
                text = re.sub(r"\bBTW\b", "btw", text)
            else:
                text = re.sub(r"\bBTW\b", "VAT", text)
        return text

    def _repair_text_artifacts(self, text: str) -> str:
        value = str(text or "")
        replacements = {
            "â€™": "'",
            "â": "'",
            "â€˜": "'",
            "â": "'",
            "â€œ": "\"",
            "â": "\"",
            "â€": "\"",
            "â": "\"",
            "â€“": "-",
            "â": "-",
            "â€”": "-",
            "â": "-",
            "â€¦": "...",
            "â¦": "...",
            "â‚¬": "€",
            "Ã©": "é",
            "Ã¨": "è",
            "Ã«": "ë",
            "Ãª": "ê",
            "Ã¡": "á",
            "Ã ": "à",
            "Ã¶": "ö",
            "Ã¼": "ü",
            "Ã¯": "ï",
            "Ã§": "ç",
            "Ã²": "ò",
            "Ã´": "ô",
            "Ã¢â‚¬â„¢": "'",
            "Ã¢â‚¬Å“": "\"",
            "Ã¢â‚¬Â": "\"",
            "Ã¢â‚¬â€œ": "-",
            "Ã¢â‚¬â€": "-",
            "ÃƒÂ©": "é",
            "ÃƒÂ¨": "è",
            "ÃƒÂ«": "ë",
            "ÃƒÂª": "ê",
            "ÃƒÂ¡": "á",
            "Ãƒ ": "à",
            "ÃƒÂ¶": "ö",
            "ÃƒÂ¼": "ü",
            "ÃƒÂ¯": "ï",
            "ÃƒÂ§": "ç",
            "ÃƒÂ²": "ò",
            "ÃƒÂ´": "ô",
            "Ã¢â€šÂ¬": "€",
        }
        for source, target in replacements.items():
            value = value.replace(source, target)
        value = re.sub(r"[ \t]{2,}", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    def _summary_has_safety_context(self, answer: str, question: str) -> bool:
        context = f"{answer}\n{question}".lower()
        return any(term in context for term in self.SUMMARY_SAFETY_TERMS)

    def _trim_summary_tail_to_word_boundary(self, answer: str) -> str:
        text = (answer or "").rstrip()
        if not text:
            return text
        if re.search(r"[.!?:\)\]\"]$", text):
            return text
        if text.endswith(":"):
            return text
        # Only trim aggressively for unusually long outputs that likely hit output limits.
        if len(text) < 1800:
            return text
        if re.search(r"[A-Za-zÃ€-Ã–Ã˜-Ã¶Ã¸-Ã¿0-9]$", text):
            boundary = re.search(r"\s+\S*$", text)
            if boundary:
                text = text[:boundary.start()].rstrip()
            if text and not text.endswith("..."):
                text = f"{text}..."
        return text

    def _strip_summary_sources_section(self, answer: str) -> str:
        patterns = [
            r"\n\s*Bronnen\s*:\s*.*$",
            r"\n\s*Sources\s*:\s*.*$",
        ]
        cleaned = answer
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def _ensure_summary_noise_note(self, answer: str, question: str) -> str:
        dutch_note = (
            "De tekstextractie lijkt op sommige plekken rommelig, dus controleer belangrijke passages "
            "in het originele document."
        )
        english_note = (
            "Text extraction appears noisy in places, so verify important passages in the original document."
        )
        if dutch_note.lower() in answer.lower() or english_note.lower() in answer.lower():
            return answer
        note = dutch_note if self._is_dutch_question(question) else english_note
        return f"{answer}\n\n{note}".strip()

    def _summary_answer_needs_fallback(self, answer: str, question: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return True
        if self._is_dutch_question(question):
            required_headers = [
                "Kort antwoord:",
                "Belangrijkste onderdelen:",
                "Belangrijke verplichtingen of aandachtspunten:",
                "Mogelijke acties voor de gebruiker:",
            ]
            matched_headers = sum(1 for h in required_headers if h in text)
            if matched_headers == 0 and len(text) < 260:
                return True
        if text.startswith("1.") and "Document summary" in text:
            return True
        bad_patterns = [r"=<{1,}", r"\bvanoers\w+", r"[~`]{2,}", r"\bwisoxrevisor\b", r"\bfss\b"]
        if any(re.search(p, text, flags=re.IGNORECASE) for p in bad_patterns):
            return True
        noisy_lines = 0
        content_lines = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            content_lines += 1
            if self._summary_text_noise_ratio(stripped) > 0.18 and len(stripped) < 120:
                noisy_lines += 1
        return content_lines > 0 and (noisy_lines / max(1, content_lines)) > 0.25

    def _build_document_summary_fallback_answer(self, search_results: List[Dict[str, Any]], question: str) -> str:
        rows = self._select_representative_summary_rows(search_results, retrieval_limit=min(12, max(6, len(search_results))))
        if not rows:
            if self._is_dutch_question(question):
                return (
                    "Kort antwoord:\n"
                    "Ik kon geen bruikbare tekstfragmenten vinden om een betrouwbare samenvatting te maken.\n\n"
                    "Belangrijkste onderdelen:\n"
                    "- Geen bruikbare fragmenten gevonden\n\n"
                    "Belangrijke verplichtingen of aandachtspunten:\n"
                    "- Controleer of de documenten correct zijn geÃ¯ndexeerd en leesbaar zijn.\n\n"
                    "Mogelijke acties voor de gebruiker:\n"
                    "- Herindexeer de relevante bestanden en stel de vraag opnieuw."
                )
            return (
                "Short answer:\n"
                "I could not find usable snippets to produce a reliable summary.\n\n"
                "Main sections:\n"
                "- No usable snippets found\n\n"
                "Key obligations or attention points:\n"
                "- Check whether documents were indexed correctly and are readable.\n\n"
                "Possible next actions:\n"
                "- Re-index the relevant files and ask again."
            )

        sentence_items = self._collect_summary_sentence_items(rows, max_items=20)
        section_items = self._collect_section_like_items(rows, max_items=4)
        obligation_items = [s for s in sentence_items if self._is_requirement_or_risk_sentence(s["text"])]
        if not obligation_items:
            obligation_items = sentence_items[:4]

        if self._is_dutch_question(question):
            short = self._build_short_summary_text(sentence_items, language="nl")
            main_sections = section_items or [s["text"] for s in sentence_items[:4]]
            obligations = [s["text"] for s in obligation_items[:4]]
            actions = self._build_summary_actions_from_evidence(rows, language="nl")
            return (
                "Kort antwoord:\n"
                f"{short}\n\n"
                "Belangrijkste onderdelen:\n"
                + "\n".join(f"- {item}" for item in main_sections[:4])
                + "\n\nBelangrijke verplichtingen of aandachtspunten:\n"
                + "\n".join(f"- {item}" for item in obligations[:4])
                + "\n\nMogelijke acties voor de gebruiker:\n"
                + "\n".join(f"- {item}" for item in actions[:3])
            ).strip()

        short = self._build_short_summary_text(sentence_items, language="en")
        main_sections = section_items or [s["text"] for s in sentence_items[:4]]
        obligations = [s["text"] for s in obligation_items[:4]]
        actions = self._build_summary_actions_from_evidence(rows, language="en")
        return (
            "Short answer:\n"
            f"{short}\n\n"
            "Main sections:\n"
            + "\n".join(f"- {item}" for item in main_sections[:4])
            + "\n\nKey obligations or attention points:\n"
            + "\n".join(f"- {item}" for item in obligations[:4])
            + "\n\nPossible next actions:\n"
            + "\n".join(f"- {item}" for item in actions[:3])
        ).strip()

    def _collect_summary_sentence_items(self, rows: List[Dict[str, Any]], max_items: int = 20) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            text = self._clean_summary_text(self._row_primary_text(row))
            if not text:
                continue
            parts = re.split(r"(?<=[\.\!\?;:])\s+|\n+", text)
            for part in parts:
                sentence = re.sub(r"\s+", " ", part).strip(" -\t")
                if len(sentence) < 45 or len(sentence) > 240:
                    continue
                sig = " ".join(re.findall(r"\w+", sentence.lower())[:12])
                if sig in seen:
                    continue
                seen.add(sig)
                items.append({"text": sentence, "source": self._row_source_label(row)})
                if len(items) >= max_items:
                    return items
        return items

    def _collect_section_like_items(self, rows: List[Dict[str, Any]], max_items: int = 4) -> List[str]:
        section_items: List[str] = []
        seen = set()
        for row in rows:
            title = re.sub(r"\s+", " ", str(row.get("title", "") or "")).strip(" -\t")
            if not title or title.lower() in {"unknown", "untitled"}:
                continue
            if len(title) < 3 or len(title) > 110:
                continue
            lower = title.lower()
            if lower in seen:
                continue
            seen.add(lower)
            section_items.append(title)
            if len(section_items) >= max_items:
                break
        return section_items

    def _is_requirement_or_risk_sentence(self, text: str) -> bool:
        t = (text or "").lower()
        risk_terms = [
            "moet", "dient", "verplicht", "vereist", "risico", "incident", "beheersmaatregel",
            "shall", "must", "required", "obligation", "hazard", "mitigation",
        ]
        return any(term in t for term in risk_terms)

    def _build_short_summary_text(self, sentence_items: List[Dict[str, str]], language: str = "nl") -> str:
        if not sentence_items:
            if language == "nl":
                return "De beschikbare fragmenten geven te weinig informatie voor een volledige samenvatting."
            return "Available snippets provide too little information for a complete summary."
        first = sentence_items[0]["text"]
        second = sentence_items[1]["text"] if len(sentence_items) > 1 else ""
        if language == "nl":
            base = f"Dit document beschrijft voornamelijk: {first}"
            if second:
                base += f" Daarnaast benadrukt het document: {second}"
            return base
        base = f"This document mainly describes: {first}"
        if second:
            base += f" It also highlights: {second}"
        return base

    def _build_summary_actions_from_evidence(self, rows: List[Dict[str, Any]], language: str = "nl") -> List[str]:
        joined = " ".join(self._row_text(r) for r in rows[:8])
        actions: List[str] = []
        if any(t in joined for t in ["risico", "incident", "veiligheid", "veiligheidsinstructie", "pbm"]):
            actions.append(
                "Controleer of alle veiligheidsinstructies, risicoanalyses en werkvergunningen actueel zijn."
                if language == "nl"
                else "Verify that safety instructions, risk assessments, and work permits are up to date."
            )
        if any(t in joined for t in ["contract", "verplicht", "aansprakelijkheid", "verzekering"]):
            actions.append(
                "Loop contractuele verplichtingen en verzekeringsvoorwaarden na met de verantwoordelijke projectleider."
                if language == "nl"
                else "Review contractual obligations and insurance conditions with the responsible project lead."
            )
        actions.append(
            "Valideer kritieke passages in het originele document voordat je formele beslissingen neemt."
            if language == "nl"
            else "Validate critical passages in the original document before making formal decisions."
        )
        return actions[:3]

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
        if intent == "technical_requirements_check":
            return self._build_technical_requirements_answer(search_results)
        if intent == "risk_attention_check":
            return self._build_risk_attention_answer(search_results)
        if intent == "ce_compliance_gap_check":
            return self._build_ce_compliance_gap_answer(search_results)
        if intent == "document_governance_check":
            return self._build_document_governance_answer(search_results)
        if intent == "quotation_preparation":
            return self._build_quotation_preparation_answer(search_results)
        if intent == "ai_use_case_identification":
            return self._build_ai_use_case_identification_answer(search_results)
        if intent == "use_case_prioritization":
            return self._build_use_case_prioritization_answer(search_results)
        if intent == "pilot_project_translation":
            return self._build_pilot_project_translation_answer(search_results)
        if intent == "local_privacy_explanation":
            return self._build_local_privacy_explanation_answer(search_results)
        return self._build_missing_info_answer(search_results)

    def _build_technical_requirements_answer(self, search_results: List[Dict[str, Any]]) -> str:
        requirement_terms = [
            "moet", "dient", "verplicht", "vereist", "eis", "procedure", "stap",
            "work permit", "werkvergunning", "instructie", "norm", "richtlijn",
        ]
        component_terms = [
            "systeem", "installatie", "onderdeel", "component", "machine", "constructie", "equipment",
        ]
        req_items = self._collect_evidence_sentences(search_results, requirement_terms, max_items=6)
        comp_items = self._collect_evidence_sentences(search_results, component_terms, max_items=4)
        if not req_items and not comp_items:
            return (
                "Kort antwoord:\n"
                "Ik vond geen expliciete technische eisen in de huidige retrievalset.\n\n"
                "Gevonden informatie:\n"
                "- Geen duidelijke eiszinnen gedetecteerd in de opgehaalde fragmenten.\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- Controleer of de juiste technische specificaties, bijlagen of werkmethodes zijn geÃ¯ndexeerd.\n\n"
                "Mogelijke vervolgstap:\n"
                "- Indexeer aanvullende technische documenten en stel de vraag opnieuw met documenttype of projectcontext."
            )
        found_lines = []
        for item in (req_items + comp_items)[:8]:
            found_lines.append(f"- {item['text']} (bron: {item['source']})")
        return (
            "Kort antwoord:\n"
            "De documenten bevatten technische eisen en uitvoeringsverplichtingen die vooraf gecontroleerd moeten worden.\n\n"
            "Gevonden informatie:\n"
            + "\n".join(found_lines)
            + "\n\nAandachtspunten of ontbrekende informatie:\n"
            "- Niet alle eisen zijn gekwantificeerd (bijv. toleranties, acceptatiecriteria of exacte normversies).\n"
            "- Verifieer dat de geldende versie van procedures en normen is gebruikt.\n\n"
            "Mogelijke vervolgstap:\n"
            "- Maak een controlelijst per eis: bron, eigenaar, verificatiemethode en bewijs van naleving."
        ).strip()

    def _build_risk_attention_answer(self, search_results: List[Dict[str, Any]]) -> str:
        risk_terms = [
            "risico", "gevaar", "incident", "bijna-ongeval", "aandachtspunt", "verboden",
            "veiligheid", "beheersmaatregel", "waarschuwing", "hazard", "mitigation",
        ]
        risk_items = self._collect_evidence_sentences(search_results, risk_terms, max_items=8)
        if not risk_items:
            return (
                "Kort antwoord:\n"
                "Er zijn in de huidige snippets geen expliciete risicopassages gevonden.\n\n"
                "Gevonden informatie:\n"
                "- Geen directe risicozinnen in de opgehaalde context.\n\n"
                "Aandachtspunten of ontbrekende informatie:\n"
                "- Mogelijk ontbreken HSE-bijlagen, risicoanalyses of incidentprocedures in de index.\n\n"
                "Mogelijke vervolgstap:\n"
                "- Vraag gericht op risicoanalyse, incidentmelding of veiligheidsmaatregelen per processtap."
            )
        found_lines = [f"- {item['text']} (bron: {item['source']})" for item in risk_items[:8]]
        return (
            "Kort antwoord:\n"
            "Het document benoemt meerdere risicoâ€™s en aandachtspunten die operationele en veiligheidsimpact kunnen hebben.\n\n"
            "Gevonden informatie:\n"
            + "\n".join(found_lines)
            + "\n\nAandachtspunten of ontbrekende informatie:\n"
            "- Niet elk risico is voorzien van kans/impact-classificatie of meetbare acceptatiegrens.\n"
            "- Controleer of mitigerende acties en verantwoordelijken expliciet zijn toegewezen.\n\n"
            "Mogelijke vervolgstap:\n"
            "- Zet de gevonden risicoâ€™s om naar een risicoregister met eigenaar, deadline en bewijs van mitigatie."
        ).strip()

    def _build_ce_compliance_gap_answer(self, search_results: List[Dict[str, Any]]) -> str:
        ce_requirements = [
            ("Technisch dossier", ["technisch dossier", "technical file"]),
            ("Risicobeoordeling", ["risicobeoordeling", "risk assessment", "hazard analysis"]),
            ("Normen en richtlijnen", ["norm", "richtlijn", "standard", "directive"]),
            ("Verklaring van overeenstemming", ["verklaring van overeenstemming", "declaration of conformity"]),
            ("Test- of validatierapporten", ["testrapport", "test report", "validatie", "verification"]),
            ("Gebruikers- en veiligheidsinstructies", ["gebruikershandleiding", "veiligheidsinstructie", "instructions for use"]),
        ]
        found: List[str] = []
        missing: List[str] = []
        for label, terms in ce_requirements:
            evidence = self._collect_evidence_sentences(search_results, terms, max_items=1)
            if evidence:
                found.append(f"- {label}: {evidence[0]['text']} (bron: {evidence[0]['source']})")
            else:
                missing.append(f"- {label}: geen expliciete onderbouwing gevonden in de opgehaalde snippets.")
        return (
            "Kort antwoord:\n"
            "Op basis van de huidige documenten is CE-relevante informatie deels aanwezig, maar volledige compliance is niet aantoonbaar zonder aanvullende onderbouwing.\n\n"
            "Gevonden informatie:\n"
            + ("\n".join(found) if found else "- Geen expliciete CE-onderdelen gevonden.")
            + "\n\nAandachtspunten of ontbrekende informatie:\n"
            + ("\n".join(missing) if missing else "- Geen directe hiaten gevonden; controle op volledigheid blijft noodzakelijk.")
            + "\n\nMogelijke vervolgstap:\n"
            "- Maak een CE-checkmatrix met per vereist onderdeel: aanwezig bewijs, ontbrekend bewijs, verantwoordelijke en deadline."
        ).strip()

    def _build_document_governance_answer(self, search_results: List[Dict[str, Any]]) -> str:
        metadata = self._scan_metadata_signals(search_results)
        conflict_items = self._scan_instruction_conflicts(search_results, max_items=4)
        standard_gaps = self._scan_standard_references_without_version(search_results, max_items=4)
        ocr_items = self._scan_ocr_noise_candidates(search_results, max_items=4)
        confidentiality_items = self._collect_evidence_sentences(
            search_results,
            ["vertrouwelijk", "confidential", "directie", "schriftelijke toelating", "privacy", "gegevens"],
            max_items=3,
        )

        found_lines: List[str] = []
        missing_lines: List[str] = []
        next_steps: List[str] = []

        if metadata["found_lines"]:
            found_lines.extend(metadata["found_lines"][:4])
        else:
            found_lines.append("- Geen expliciete metadata-velden in de huidige snippets gevonden.")

        if conflict_items:
            found_lines.extend([f"- Mogelijke tegenstrijdigheid: {item}" for item in conflict_items[:2]])
        if standard_gaps:
            found_lines.extend([f"- Norm/richtlijn zonder versie: {item}" for item in standard_gaps[:2]])
        if confidentiality_items:
            found_lines.extend(
                [f"- Vertrouwelijkheidsrelevant: {item['text']} (bron: {item['source']})" for item in confidentiality_items[:2]]
            )

        if metadata["missing_fields"]:
            missing_lines.append("- Mogelijk ontbrekende metadata: " + ", ".join(metadata["missing_fields"]) + ".")
            next_steps.append("Vul ontbrekende metadata aan in de documentkop of documentregister.")
        else:
            missing_lines.append("- Kernmetadata lijkt deels aanwezig; controle op consistentie tussen documenten blijft nodig.")

        if standard_gaps:
            missing_lines.append("- Meerdere verwijzingen naar normen/richtlijnen zonder concrete versie of jaartal.")
            next_steps.append("Leg per norm vast: code, versie, publicatiedatum en toepassingsscope.")
        if conflict_items:
            missing_lines.append("- Mogelijke dubbeling of tegenstrijdigheid in instructies vraagt handmatige review.")
            next_steps.append("Maak een conflictlijst met bronlocaties en wijs per punt een eigenaar toe.")
        if ocr_items:
            missing_lines.append("- Er zijn OCR-gevoelige fragmenten die handmatige validatie vereisen.")
            next_steps.append("Hercontroleer OCR-gevoelige passages in het originele PDF-bestand.")

        if not next_steps:
            next_steps.append("Plan een korte documentreview op metadata, versiebeheer en traceerbaarheid.")
        next_steps.append("Koppel deze bevindingen aan de volgende revisieplanning met prioriteit en deadline.")

        return (
            "Kort antwoord:\n"
            "Het dossier bevat bruikbare inhoud, maar voor robuust documentbeheer zijn metadata, versieverwijzingen en kwaliteitscontroles nog aan te scherpen.\n\n"
            "Gevonden informatie:\n"
            + "\n".join(found_lines[:8])
            + "\n\nAandachtspunten of ontbrekende informatie:\n"
            + "\n".join(missing_lines[:8])
            + "\n\nMogelijke vervolgstap:\n"
            + "\n".join(f"- {step}" for step in next_steps[:4])
        ).strip()

    def _scan_metadata_signals(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        patterns = {
            "datum": [r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", r"\b20\d{2}\b"],
            "revisie": [r"\brev(?:ision)?\.?\s*[a-z0-9\-_/]+\b", r"\brevisie\b", r"\bversion\b", r"\bversie\b"],
            "status": [r"\bstatus\b", r"\bdraft\b", r"\bfinal\b", r"\bgoedgekeurd\b", r"\bapproved\b"],
            "eigenaar": [r"\beigenaar\b", r"\bowner\b", r"\bauteur\b", r"\bauthor\b", r"\bdocumentbeheerder\b"],
        }
        found_fields = set()
        found_lines: List[str] = []
        for row in search_results[:12]:
            text = self._compress_context_text(self._row_primary_text(row))
            if not text:
                continue
            for field, pats in patterns.items():
                if field in found_fields:
                    continue
                if any(re.search(p, text, flags=re.IGNORECASE) for p in pats):
                    found_fields.add(field)
                    found_lines.append(f"- Metadata-signaal {field}: aanwezig (bron: {self._row_source_label(row)})")
        required = {"datum", "revisie", "status", "eigenaar"}
        missing_fields = sorted(list(required - found_fields))
        return {"found_lines": found_lines, "missing_fields": missing_fields}

    def _scan_instruction_conflicts(self, search_results: List[Dict[str, Any]], max_items: int = 4) -> List[str]:
        restrictive = self._collect_evidence_sentences(
            search_results,
            ["verboden", "niet toegestaan", "mag niet", "prohibited", "not allowed"],
            max_items=8,
        )
        permissive = self._collect_evidence_sentences(
            search_results,
            ["toegestaan", "mag", "allowed", "permitted"],
            max_items=8,
        )
        items: List[str] = []
        if restrictive and permissive:
            for left in restrictive[:2]:
                for right in permissive[:2]:
                    left_sig = set(re.findall(r"\w+", left["text"].lower()))
                    right_sig = set(re.findall(r"\w+", right["text"].lower()))
                    overlap = left_sig.intersection(right_sig) - self.GENERIC_STOPWORDS_NL
                    if len(overlap) >= 2:
                        term = ", ".join(sorted(list(overlap))[:3])
                        items.append(
                            f"mogelijk conflict rond [{term}] tussen {left['source']} en {right['source']}"
                        )
                        if len(items) >= max_items:
                            return items
        return items

    def _scan_standard_references_without_version(
        self, search_results: List[Dict[str, Any]], max_items: int = 4
    ) -> List[str]:
        terms = ["norm", "richtlijn", "directive", "standard", "iso", "nen", "en "]
        items: List[str] = []
        seen = set()
        for row in search_results[:14]:
            text = self._compress_context_text(self._row_primary_text(row))
            if not text:
                continue
            parts = re.split(r"(?<=[\.\!\?;:])\s+|\n+", text)
            for part in parts:
                sentence = re.sub(r"\s+", " ", part).strip()
                if len(sentence) < 30:
                    continue
                low = sentence.lower()
                if not any(t in low for t in terms):
                    continue
                has_version = bool(
                    re.search(r"\b(19|20)\d{2}\b", low)
                    or re.search(r"\b(iso|nen|en|iec)\s*\d{2,6}([:-]\d{2,4})?\b", low)
                    or re.search(r"\bversion\s*\d+(\.\d+)?\b", low)
                    or re.search(r"\brev(?:ision)?\.?\s*[a-z0-9\-_/]+\b", low)
                )
                if has_version:
                    continue
                cleaned = self._clean_evidence_sentence(sentence)
                if self._is_low_quality_evidence_sentence(cleaned):
                    continue
                sig = " ".join(re.findall(r"\w+", cleaned.lower())[:10])
                if sig in seen:
                    continue
                seen.add(sig)
                items.append(f"{cleaned} (bron: {self._row_source_label(row)})")
                if len(items) >= max_items:
                    return items
        return items

    def _scan_ocr_noise_candidates(self, search_results: List[Dict[str, Any]], max_items: int = 4) -> List[str]:
        items: List[str] = []
        for row in search_results[:16]:
            text = self._row_primary_text(row)
            if not text:
                continue
            ratio = self._summary_text_noise_ratio(text)
            if ratio < 0.16:
                continue
            preview = re.sub(r"\s+", " ", text).strip()[:110]
            preview = self._clean_evidence_sentence(preview)
            if not preview:
                continue
            items.append(f"{preview}... (bron: {self._row_source_label(row)})")
            if len(items) >= max_items:
                break
        return items

    def _build_quotation_preparation_answer(self, search_results: List[Dict[str, Any]]) -> str:
        info_blocks = [
            ("Klantvraag en scope", ["scope", "klantvraag", "requirements", "eisen", "deliverable", "levering"]),
            ("Technische randvoorwaarden", ["technisch", "specificatie", "norm", "richtlijn", "interface", "component"]),
            ("Planning en doorlooptijd", ["planning", "doorlooptijd", "deadline", "mijlpaal", "lead time"]),
            ("Risicoâ€™s en aannames", ["risico", "assumptie", "aandachtspunt", "beperking"]),
            ("Prijs- en kostengrondslag", ["prijs", "kosten", "rate", "tarief", "budget"]),
            ("Contractuele voorwaarden", ["contract", "aansprakelijkheid", "garantie", "acceptatie"]),
        ]
        found: List[str] = []
        missing: List[str] = []
        for label, terms in info_blocks:
            evidence = self._collect_evidence_sentences(search_results, terms, max_items=1)
            if evidence:
                found.append(f"- {label}: {evidence[0]['text']} (bron: {evidence[0]['source']})")
            else:
                missing.append(f"- {label}: niet expliciet gevonden in de huidige snippets.")
        return (
            "Kort antwoord:\n"
            "De documenten bevatten bruikbare input voor een eerste offerte-opzet, maar belangrijke commerciÃ«le en scopegegevens kunnen nog ontbreken.\n\n"
            "Gevonden informatie:\n"
            + ("\n".join(found) if found else "- Beperkte offerte-relevante informatie in de huidige retrievalset.")
            + "\n\nAandachtspunten of ontbrekende informatie:\n"
            + ("\n".join(missing) if missing else "- Geen kritieke hiaten gedetecteerd, verifieer wel volledigheid.")
            + "\n- Geen prijzen, doorlooptijden of toezeggingen invullen zonder expliciete brondata.\n\n"
            "Mogelijke vervolgstap:\n"
            "- Bouw een offertechecklist met secties: scope, technische eisen, planning, risicoâ€™s, aannames, prijsgrondslag en contractvoorwaarden."
        ).strip()

    def _build_ai_use_case_identification_answer(self, search_results: List[Dict[str, Any]]) -> str:
        use_cases = self._infer_candidate_use_cases(search_results)
        lines = []
        for uc in use_cases[:4]:
            lines.append(
                f"- {uc['name']}: {uc['why']} (evidence: {uc['evidence']})"
            )
        return (
            "Kort antwoord:\n"
            "Op basis van de documentpatronen zijn meerdere praktische AI-use-cases haalbaar als ondersteunende laag bovenop bestaande processen.\n\n"
            "Gevonden informatie:\n"
            + "\n".join(lines)
            + "\n\nAandachtspunten of ontbrekende informatie:\n"
            "- Datakwaliteit (OCR, versiebeheer, metadata) bepaalt direct de betrouwbaarheid van antwoorden.\n"
            "- AI ondersteunt voorbereiding, maar neemt geen formele compliance- of contractbeslissingen.\n\n"
            "Mogelijke vervolgstap:\n"
            "- Kies 1 use-case met duidelijke documentstroom en meetbaar resultaat om als pilot te starten."
        ).strip()

    def _build_use_case_prioritization_answer(self, search_results: List[Dict[str, Any]]) -> str:
        use_cases = self._infer_candidate_use_cases(search_results)
        ranked = sorted(use_cases, key=lambda x: x["total_score"], reverse=True)
        rows: List[str] = []
        for uc in ranked[:4]:
            rows.append(
                f"- {uc['name']}: impact {uc['impact']}/5, haalbaarheid {uc['feasibility']}/5, databeschikbaarheid {uc['data_availability']}/5 (totaal {uc['total_score']}/15)"
            )
        top = ranked[0] if ranked else None
        top_line = top["name"] if top else "Nog geen duidelijke kandidaat"
        return (
            "Kort antwoord:\n"
            f"De meest kansrijke eerste pilot is: {top_line}.\n\n"
            "Gevonden informatie:\n"
            + ("\n".join(rows) if rows else "- Onvoldoende basis om use-cases te scoren.")
            + "\n\nAandachtspunten of ontbrekende informatie:\n"
            "- Scores zijn indicatief en afhankelijk van documentdekking en procescontext.\n"
            "- Valideer met proceseigenaren of de benodigde data operationeel beschikbaar is.\n\n"
            "Mogelijke vervolgstap:\n"
            "- Start met een timeboxed pilot voor de top-1 use-case en meet tijdswinst, kwaliteit en adoptie."
        ).strip()

    def _build_pilot_project_translation_answer(self, search_results: List[Dict[str, Any]]) -> str:
        use_cases = self._infer_candidate_use_cases(search_results)
        ranked = sorted(use_cases, key=lambda x: x["total_score"], reverse=True)
        chosen = ranked[0] if ranked else {
            "name": "Interne document-Q&A en samenvatting",
            "problem": "Medewerkers verliezen tijd met zoeken in versnipperde documentatie.",
            "required_data": "Projectdocumenten, procedures, contractbijlagen, versiehistorie.",
            "stakeholders": "Projectleiding, engineering, kwaliteit/HSE, IT/data.",
            "prototype": "Lokale chatinterface met bronkaarten, samenvatting en eisenextractie.",
            "success_metrics": "Zoektijd per vraag, % bruikbare antwoorden, correctievolume door experts.",
        }
        return (
            "Kort antwoord:\n"
            f"Een realistisch pilotproject is: {chosen['name']}.\n\n"
            "Gevonden informatie:\n"
            f"- Probleem: {chosen['problem']}\n"
            f"- Benodigde data: {chosen['required_data']}\n"
            f"- Betrokkenen: {chosen['stakeholders']}\n"
            f"- Prototype (20 weken): {chosen['prototype']}\n"
            f"- Verwachte opbrengst: {chosen['success_metrics']}\n\n"
            "Aandachtspunten of ontbrekende informatie:\n"
            "- Borg dat documentkwaliteit, toegangsrechten en validatiecriteria vooraf zijn vastgelegd.\n"
            "- Plan periodieke reviewmomenten met domeinexperts om hallucinaties te beperken.\n\n"
            "Mogelijke vervolgstap:\n"
            "- Stel een pilotcharter op met scope, dataset, evaluatiekader en weekplanning (0-4 setup, 5-12 build, 13-20 evaluatie)."
        ).strip()

    def _build_local_privacy_explanation_answer(self, search_results: List[Dict[str, Any]]) -> str:
        has_local_docs = any(str(r.get("corpus_type", "")).lower() in {"uploaded", "existing"} for r in search_results)
        local_line = (
            "Documenten worden lokaal geÃ¯ndexeerd en geraadpleegd binnen deze omgeving."
            if has_local_docs
            else "Deze setup is ontworpen voor lokale documentverwerking en retrieval."
        )
        return (
            "Kort antwoord:\n"
            "Een lokale AI-assistent is relevant omdat gevoelige bedrijfsdocumentatie binnen de eigen omgeving verwerkt kan blijven.\n\n"
            "Gevonden informatie:\n"
            f"- {local_line}\n"
            "- Antwoorden worden gebaseerd op lokale retrievalresultaten met bronkaarten voor controle.\n\n"
            "Aandachtspunten of ontbrekende informatie:\n"
            "- Lokale AI blijft afhankelijk van datakwaliteit, indexdekking en modelcapaciteit.\n"
            "- Menselijke controle blijft nodig voor compliance, contracten en formele besluitvorming.\n\n"
            "Mogelijke vervolgstap:\n"
            "- Leg een governance-kader vast: toegangsrechten, logging, reviewproces en escalatie voor kritieke beslissingen."
        ).strip()

    def _collect_evidence_sentences(
        self,
        search_results: List[Dict[str, Any]],
        terms: List[str],
        max_items: int = 6,
    ) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        candidates: List[Tuple[float, Dict[str, str]]] = []
        seen = set()
        terms_lower = [t.lower() for t in terms]
        for row in search_results:
            text = self._clean_summary_text(self._row_primary_text(row))
            if not text:
                text = self._compress_context_text(self._row_primary_text(row))
            if not text:
                continue
            parts = re.split(r"(?<=[\.\!\?;:])\s+|\n+", text)
            for part in parts:
                sentence = re.sub(r"\s+", " ", part).strip(" -\t")
                if len(sentence) < 35 or len(sentence) > 220:
                    continue
                if self._summary_text_noise_ratio(sentence) > 0.12:
                    continue
                low = sentence.lower()
                words = re.findall(r"\w+", low)
                if len(words) < 6:
                    continue
                matched_terms = [term for term in terms_lower if term in low]
                if not matched_terms:
                    continue
                sentence = self._extract_relevant_clause(sentence, matched_terms)
                sentence = self._clean_evidence_sentence(sentence)
                if self._is_low_quality_evidence_sentence(sentence):
                    continue
                signature = " ".join(re.findall(r"\w+", sentence.lower())[:12])
                if signature in seen:
                    continue
                seen.add(signature)
                score = self._score_evidence_sentence(sentence, matched_terms)
                candidates.append((score, {"text": sentence, "source": self._row_source_label(row)}))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, item in candidates[: max_items * 2]:
            items.append(item)
            if len(items) >= max_items:
                break
        return items

    def _extract_relevant_clause(self, sentence: str, matched_terms: List[str]) -> str:
        s = str(sentence or "").strip()
        if not s or not matched_terms:
            return s
        low = s.lower()
        first_pos = None
        first_term = ""
        for term in matched_terms:
            pos = low.find(term)
            if pos >= 0 and (first_pos is None or pos < first_pos):
                first_pos = pos
                first_term = term
        if first_pos is None:
            return s

        start = max(0, first_pos - 70)
        end = min(len(s), first_pos + max(120, len(first_term) + 90))
        window = s[start:end]
        # Keep the clause as natural as possible by trimming to nearby separators.
        left_cut = max(window.rfind(". "), window.rfind("; "), window.rfind(": "))
        if left_cut >= 0:
            window = window[left_cut + 2 :]
        right_candidates = [idx for idx in [window.find(". "), window.find("; ")] if idx >= 0]
        if right_candidates:
            window = window[: min(right_candidates) + 1]
        return window.strip(" -\t")

    def _clean_evidence_sentence(self, sentence: str) -> str:
        s = str(sentence or "")
        s = re.sub(r"^[\W_]+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[\|_/]{2,}", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        # Remove obvious orphan section numbering at line start.
        s = re.sub(r"^\d+(\.\d+){0,3}\s*", "", s).strip()
        return s

    def _is_low_quality_evidence_sentence(self, sentence: str) -> bool:
        s = str(sentence or "").strip()
        if len(s) < 35:
            return True
        words = re.findall(r"\w+", s.lower())
        if len(words) < 6:
            return True
        alpha_ratio = sum(ch.isalpha() for ch in s) / max(1, len(s))
        if alpha_ratio < 0.6:
            return True
        long_odd_tokens = 0
        for w in words:
            if len(w) >= 11 and not re.search(r"[aeiouyÃ¡Ã Ã¤Ã¢Ã©Ã¨Ã«ÃªÃ­Ã¬Ã¯Ã®Ã³Ã²Ã¶Ã´ÃºÃ¹Ã¼Ã»]", w):
                long_odd_tokens += 1
        if long_odd_tokens >= 2:
            return True
        # Reject lines that are mostly title/header fragments.
        if re.search(r"\b(page|pagina)\b\s*\d*", s.lower()) and len(words) < 10:
            return True
        return False

    def _score_evidence_sentence(self, sentence: str, matched_terms: List[str]) -> float:
        s = str(sentence or "")
        low = s.lower()
        words = re.findall(r"\w+", low)
        if not words:
            return -999.0
        score = 0.0
        score += min(3.0, len(words) / 6.0)
        score += min(2.5, sum(1 for t in matched_terms if t in low) * 0.8)
        score += 1.0 if re.search(r"[.;:]", s) else 0.0
        score -= self._summary_text_noise_ratio(s) * 8.0

        weird_tokens = 0
        for w in words:
            if len(w) < 8:
                continue
            vowels = len(re.findall(r"[aeiouyÃ¡Ã Ã¤Ã¢Ã©Ã¨Ã«ÃªÃ­Ã¬Ã¯Ã®Ã³Ã²Ã¶Ã´ÃºÃ¹Ã¼Ã»]", w))
            ratio = vowels / max(1, len(w))
            if ratio < 0.22:
                weird_tokens += 1
        score -= weird_tokens * 0.9

        if re.match(r"^[a-z]{10,}\b", low):
            score -= 1.0
        return score

    def _infer_candidate_use_cases(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_blob = " ".join(self._row_text(r) for r in search_results[:14])
        source_count = max(1, len(search_results))
        templates = [
            {
                "name": "Veiligheids- en verplichtingencheck uit documentatie",
                "signals": ["veiligheid", "risico", "incident", "verplicht", "werkvergunning", "instructie"],
                "problem": "Belangrijke veiligheids- en verplichtingsinformatie staat verspreid over meerdere documenten.",
                "required_data": "HSE-documenten, contractverplichtingen, werkprocedures, incidentregistraties.",
                "stakeholders": "HSE, projectleiding, uitvoering, kwaliteitscoÃ¶rdinatie.",
                "prototype": "Lokale assistent die eisen/risicoâ€™s per vraag samenvat met bronkaarten.",
                "success_metrics": "Minder zoektijd, hogere dekking van controles, minder gemiste verplichtingen.",
            },
            {
                "name": "CE-dossier volledigheidscontrole",
                "signals": ["ce", "conformiteit", "technisch dossier", "norm", "richtlijn", "risicobeoordeling"],
                "problem": "CE-gerelateerde bewijsstukken zijn vaak incompleet of lastig traceerbaar.",
                "required_data": "Normenoverzicht, testverslagen, risicobeoordeling, conformiteitsverklaring.",
                "stakeholders": "Engineering, quality/compliance, projectleiding.",
                "prototype": "Checklist-engine die per CE-onderdeel aanwezige en ontbrekende bewijsstukken markeert.",
                "success_metrics": "Snellere dossiercontrole, minder ontbrekende CE-items bij review.",
            },
            {
                "name": "Offertevoorbereiding op basis van technische documentatie",
                "signals": ["offerte", "scope", "eis", "planning", "randvoorwaarde", "contract"],
                "problem": "Offerte-input moet handmatig uit verschillende documenten worden verzameld.",
                "required_data": "Klantvraag, technische eisen, planning, voorwaarden, historische projectdocumenten.",
                "stakeholders": "Sales, engineering, calculatie, projectmanagement.",
                "prototype": "Assistent die een offertechecklist en conceptopbouw maakt met bronverwijzingen.",
                "success_metrics": "Kortere offertedoorlooptijd en minder iteraties door ontbrekende input.",
            },
            {
                "name": "Interne kennisassistent voor technische documentvragen",
                "signals": ["procedure", "stap", "verantwoord", "onderdeel", "systeem", "handleiding"],
                "problem": "Nieuwe en bestaande medewerkers verliezen tijd aan handmatig zoeken in documentatie.",
                "required_data": "Procedures, handleidingen, projectdocumenten en FAQ-achtige notities.",
                "stakeholders": "Engineering, operations, onboarding, IT.",
                "prototype": "Lokale Q&A met bronkaarten en samenvattingen per onderwerp.",
                "success_metrics": "Minder zoektijd, hogere first-time-right beantwoording, snellere onboarding.",
            },
        ]
        use_cases: List[Dict[str, Any]] = []
        for template in templates:
            hits = sum(1 for s in template["signals"] if s in text_blob)
            coverage = hits / max(1, len(template["signals"]))
            evidence_items = self._collect_evidence_sentences(search_results, template["signals"], max_items=2)
            evidence_text = "; ".join(f"{e['text']} ({e['source']})" for e in evidence_items) if evidence_items else "beperkte directe bewijsregels"
            impact = min(5, max(2, int(round(1 + (coverage * 4)))))
            feasibility = min(5, max(2, int(round(1 + (len(evidence_items) * 1.5) + (1 if source_count >= 8 else 0)))))
            data_availability = min(5, max(1, int(round(1 + min(4, source_count / 6) + coverage))))
            total = impact + feasibility + data_availability
            why = "Sterke documentevidence voor repeterende documenttaken." if hits >= 3 else "Bruikbaar, maar extra documentdekking kan nodig zijn."
            use_cases.append(
                {
                    "name": template["name"],
                    "impact": impact,
                    "feasibility": feasibility,
                    "data_availability": min(5, data_availability),
                    "total_score": min(15, total),
                    "why": why,
                    "evidence": evidence_text,
                    "problem": template["problem"],
                    "required_data": template["required_data"],
                    "stakeholders": template["stakeholders"],
                    "prototype": template["prototype"],
                    "success_metrics": template["success_metrics"],
                }
            )
        return use_cases

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
        filename = row.get("filename", "Unknown")
        page_number = row.get("page_number")
        chunk_index = row.get("chunk_index")
        if page_number:
            return f"{filename}, page {page_number}"
        if chunk_index:
            return f"{filename}, chunk {chunk_index}"
        return f"{filename}, page/chunk unknown"

    def _collect_source_lines(self, rows: List[Dict[str, Any]], limit: int = 6) -> List[str]:
        lines: List[str] = []
        seen = set()
        for row in rows:
            page_number = row.get("page_number")
            chunk_index = row.get("chunk_index")
            location_key = page_number if page_number is not None else f"chunk:{chunk_index}"
            key = (row.get("filename"), location_key)
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
        for token in re.findall(r"(?:â‚¬|\$)?\s*\(?\d[\d\.,]*\)?", text or ""):
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
        pattern = r"(?:â‚¬|\$)?\s*\(?\d[\d\.,]*\)?"
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
        return f"{sign}â‚¬{s}"

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
        num_pattern = r"(?:â‚¬|\$)?\s*\(?\d[\d\.,]*\)?"

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
                    ("Contract- en afhankelijkheidsrisico", "Contractvoorwaarden en afhankelijkheden bepalen risico en onderhandelingsruimte.", "Waar zitten de grootste contractuele risicoâ€™s of afhankelijkheden?"),
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
            "Dit zijn mogelijke verzekeringsrisicoâ€™s op basis van de beschikbare documenten.",
            "",
            "Mogelijke verzekeringsrisicoâ€™s:",
        ]
        for idx, (name, evidence, check) in enumerate(dedup[:3], start=1):
            lines.append(f"{idx}. {name}")
            lines.append("   - Bewijs:")
            lines.append(f"   {evidence}")
            lines.append("   - Waarom dit belangrijk is:")
            lines.append("   Onvoldoende of onduidelijke dekking kan leiden tot onverwachte financiÃ«le schade.")
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
        business = "MKB-dossier met operationele en financiÃ«le bronstukken." if text_blob else "Onvoldoende context in opgehaalde bronnen."
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
            financial_points.append("- Beperkte financiÃ«le detailinformatie in de huidige retrievalset.")

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
            risks.append("- Aanvullende bronstukken nodig om operationele risicoâ€™s beter te beoordelen.")

        missing = [
            "- Volledige bronset voor facturen/bankafschriften per periode",
            "- Expliciete aansluiting tussen jaarrekeningposten en btw-overzicht",
        ]
        if any(h["term"] == "ontbreekt" for h in hints):
            missing.append("- In de opgehaalde context staat expliciet dat informatie ontbreekt; verifieer welke stukken nog ontbreken.")

        follow_up = [
            "- Welke posten verschillen tussen interne notities en financiÃ«le overzichten?",
            "- Zijn alle contracten en polisvoorwaarden actueel opgenomen in het dossier?",
        ]

        return (
            "Samenvatting klantdossier:\n\n"
            "Bedrijfsactiviteit:\n"
            f"{business}\n\n"
            "FinanciÃ«le punten:\n"
            + "\n".join(financial_points)
            + "\n\nBelasting-/btw-punten:\n"
            + "\n".join(tax_points)
            + "\n\nRisicoâ€™s of aandachtspunten:\n"
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
            resolved_tax_year = self._normalized_source_tax_year(row.get("filename"), row.get("tax_year"))
            out.append({
                "document_id": row.get("document_id"),
                "filename": row.get("filename"),
                "view_filename": row.get("view_filename"),
                "title": row.get("title", row.get("filename", "Unknown")),
                "score": row.get("score", 0.0),
                "category": row.get("category", "other"),
                "document_type_detail": (row.get("metadata") or {}).get("document_type_detail"),
                "page": row.get("page_number") or row.get("chunk_index"),
                "snippet": (row.get("snippet") or row.get("summary") or "")[:350],
                "jurisdiction": row.get("jurisdiction"),
                "tax_year": resolved_tax_year,
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
            if isinstance(ty, int) and 1990 <= ty <= 2099:
                years.append(ty)
            years.extend(int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", str(row.get("content", ""))[:2000]))
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
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
        if not years:
            return None
        return max(int(y) for y in years)

    def _normalized_source_tax_year(self, filename: Any, raw_tax_year: Any) -> Optional[int]:
        if isinstance(raw_tax_year, int) and 1990 <= raw_tax_year <= 2099:
            return raw_tax_year
        name = str(filename or "")
        name_years = [int(y) for y in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", name)]
        return max(name_years) if name_years else None

    def _parse_numeric_token(self, token: str) -> Optional[float]:
        if not token:
            return None
        t = token.strip().replace("$", "").replace("â‚¬", "").replace(" ", "")
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
            resolved_tax_year = self._normalized_source_tax_year(item.get("filename"), item.get("tax_year"))
            sources.append({
                "document_id": item.get("document_id"),
                "filename": item.get("filename"),
                "view_filename": item.get("view_filename"),
                "title": item.get("title", item.get("filename", "Unknown")),
                "score": item.get("score", 0.0),
                "category": item.get("category", "other"),
                "document_type_detail": (item.get("metadata") or {}).get("document_type_detail"),
                "page": item.get("page_number") or item.get("chunk_index"),
                "snippet": (item.get("snippet") or item.get("summary") or "")[:350],
                "jurisdiction": item.get("jurisdiction"),
                "tax_year": resolved_tax_year,
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

