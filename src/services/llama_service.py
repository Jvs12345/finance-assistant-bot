"""Llama/Ollama service for financial document Q&A."""

from typing import Dict, Any, List, Optional
import re
import json

from src.db.elasticsearch_client import get_elasticsearch_client
from src.services.ollama_client import chat as ollama_chat, list_models as ollama_list_models, OllamaError
from src.services.calculation_sandbox import CalculationSandbox, CalculationSandboxError
from src.services.formula_registry import get_formula_registry
from src.utils.logging import get_logger

logger = get_logger(__name__)
SERVICE_VERSION = 3


class LlamaService:
    """Q&A service using Elasticsearch retrieval and Ollama."""

    CONTEXT_KEYWORDS = {
        "jurisdiction": ["tax rate", "vat", "deductible", "deduction", "threshold", "deadline", "exemption", "filing"],
        "tax_year": ["tax year", "fiscal year", "deadline", "threshold", "rate", "allowance", "deductible"],
        "entity_type": ["deductible", "corporate", "income tax", "filing", "return", "allowance"],
    }

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.es_client = get_elasticsearch_client()
        self.formula_registry = get_formula_registry()
        self._service_version = SERVICE_VERSION

    def ask(
        self,
        question: str,
        max_context_docs: int = 5,
        temperature: float = 0.3,
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

            search_results = self.es_client.search(
                query=question,
                limit=max_context_docs,
                enable_fuzzy=True,
                system_context=system_context,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                client_name=client_name,
                document_type=document_type
            )

            if not search_results:
                return {
                    "answer": self._no_results_response(question),
                    "sources": [],
                    "model": self.model,
                    "found_documents": False
                }

            if self._is_low_confidence(question, search_results):
                return {
                    "answer": self._low_confidence_response(),
                    "sources": self._format_sources(search_results, list(range(min(2, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results)
                }

            context_requirements = self._detect_context_requirements(question)
            missing_context_message = self._build_missing_context_message(
                context_requirements=context_requirements,
                jurisdiction=jurisdiction,
                tax_year=tax_year,
                entity_type=entity_type,
                search_results=search_results
            )
            if missing_context_message:
                return {
                    "answer": missing_context_message,
                    "sources": self._format_sources(search_results, list(range(min(3, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results)
                }

            direct_calc_answer = self._try_direct_financial_calculation(question, search_results)
            if direct_calc_answer:
                return {
                    "answer": direct_calc_answer,
                    "sources": self._format_sources(search_results, list(range(min(3, len(search_results))))),
                    "model": self.model,
                    "found_documents": True,
                    "num_documents_used": len(search_results)
                }

            context = self._build_context(search_results)
            prompt = self._create_prompt(
                question=question,
                context=context,
                history=history
            )

            answer = self._generate_llama_answer(prompt=prompt, temperature=temperature)
            answer = self._strip_disclaimer_section(answer)
            answer = self._apply_calculation_sandbox(answer)

            cited_indices = self._extract_cited_indices(answer, len(search_results))
            if not cited_indices:
                cited_indices = list(range(min(3, len(search_results))))

            sources = self._format_sources(search_results, cited_indices)

            return {
                "answer": answer,
                "sources": sources,
                "model": self.model,
                "found_documents": True,
                "num_documents_used": len(search_results)
            }

        except Exception as e:
            logger.error(f"Error in Llama service: {e}", exc_info=True)
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "model": self.model,
                "found_documents": False,
                "num_documents_used": 0,
                "error": str(e)
            }

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        context_parts = []
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
            if len(content) > 1200:
                content = content[:1200] + "..."

            location = f"page {page_num}" if page_num else "page unknown"
            if section_reference:
                location = f"{location}, section {section_reference}"

            context_parts.append(
                f"--- Document {i} ---\n"
                f"Source: {filename}\n"
                f"Location: {location}\n"
                f"Type: {category}\n"
                f"Jurisdiction: {jurisdiction or 'unknown'}\n"
                f"Tax Year: {tax_year if tax_year is not None else 'unknown'}\n"
                f"Entity Type: {entity_type or 'unknown'}\n"
                f"Extract:\n{content}\n"
            )

        return "\n".join(context_parts)

    def _create_prompt(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        history_text = ""
        if history:
            history_text = "Conversation History:\n"
            for msg in history[-3:]:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            history_text += "\n"

        formula_hints = self.formula_registry.render_prompt_hints(question, max_results=6)
        formula_hints_block = f"\n{formula_hints}\n" if formula_hints else ""

        return f"""You are a financial document assistant. You help users understand tax law, accounting documents, and financial records based only on the documents provided to you.

Rules you must follow:
1. Ground every statement in the retrieved documents.
2. Do NOT invent tax rates, thresholds, deadlines, exemptions, or legal interpretations.
3. If documents are insufficient, say: "I could not find this in the provided documents."
4. Distinguish factual extraction from interpretation.
5. If multiple documents conflict, explicitly describe the conflict and cite both.
6. CITATION REQUIREMENT: Cite sources using [Document X] in the answer body.
7. If the user asks for a numeric calculation and the needed numbers are in the retrieved documents, include a calculation payload in this exact block format:
```calc
{{"expression":"(revenue_2026 - revenue_2025) / revenue_2025 * 100","variables":{{"revenue_2026":109896,"revenue_2025":90234}},"label":"Revenue growth","unit":"%"}}
```
Use only numbers found in retrieved documents. If data is missing, do not include this block.

Retrieved documents:
{context}
{formula_hints_block}

{history_text}User Question: {question}

Answer in this exact structure:
1. Direct answer
2. Evidence from documents
3. Important assumptions or missing information
4. Suggested next step

Answer:"""

    def _generate_llama_answer(self, prompt: str, temperature: float) -> str:
        try:
            return ollama_chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature,
                num_predict=700
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

    def _apply_calculation_sandbox(self, answer: str) -> str:
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

            value = CalculationSandbox.evaluate(expression=expression, variables=variables)
            value_text = f"{value:,.6f}".rstrip("0").rstrip(".")
            calc_line = f"\n\nCalculation (sandbox): {label + ' = ' if label else ''}{value_text}{(' ' + unit) if unit else ''}"
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
                "page": item.get("page_number") or item.get("chunk_index"),
                "snippet": (item.get("snippet") or item.get("summary") or "")[:350],
                "jurisdiction": item.get("jurisdiction"),
                "tax_year": item.get("tax_year"),
                "entity_type": item.get("entity_type"),
                "client_name": item.get("client_name"),
                "section_reference": item.get("section_reference"),
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

    def list_available_models(self) -> List[str]:
        try:
            return ollama_list_models()
        except OllamaError as exc:
            logger.error(f"Error listing models: {exc}")
            return []


_llama_service: Optional[LlamaService] = None


def get_llama_service(model: str = "llama3.2") -> LlamaService:
    global _llama_service
    if (
        _llama_service is None
        or getattr(_llama_service, "_service_version", None) != SERVICE_VERSION
        or _llama_service.model != model
    ):
        _llama_service = LlamaService(model=model)
    return _llama_service
