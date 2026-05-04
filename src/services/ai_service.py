"""Compatibility wrapper around the Llama service."""

from typing import Optional, List, Dict, Any
import json
from datetime import datetime
from pathlib import Path

from src.services.llama_service import get_llama_service
from src.services.ollama_client import list_models as ollama_list_models, OllamaError
from src.utils.logging import get_logger

logger = get_logger(__name__)

SERVICE_VERSION = 5
FEEDBACK_FILE = Path("data/feedback_history.json")


class AIService:
    """Compatibility layer that forwards calls to LlamaService."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self._service_version = SERVICE_VERSION
        self._ensure_feedback_file()

    def _ensure_feedback_file(self):
        if not FEEDBACK_FILE.parent.exists():
            FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def provide_feedback(self, question: str, answer: str, rating: str):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)

            history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "answer": answer,
                    "rating": rating,
                }
            )
            history = history[-200:]

            with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")

    def ask(
        self,
        question: str,
        max_context_docs: int = 5,
        jurisdiction: Optional[str] = None,
        tax_year: Optional[int] = None,
        entity_type: Optional[str] = None,
        client_name: Optional[str] = None,
        document_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        llama_service = get_llama_service(model=self.model)
        result = llama_service.ask(
            question=question,
            max_context_docs=max_context_docs,
            jurisdiction=jurisdiction,
            tax_year=tax_year,
            entity_type=entity_type,
            client_name=client_name,
            document_type=document_type,
            history=history,
        )
        result["search_results_count"] = len(result.get("sources", []))
        return result

    def list_available_models(self) -> List[str]:
        try:
            return ollama_list_models()
        except OllamaError as exc:
            logger.error(f"Error listing models: {exc}")
            return []

    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        logger.warning("get_document_by_id is deprecated in AIService compatibility wrapper")
        return None


_ai_services: Dict[str, AIService] = {}


def get_ai_service(model: str = "llama3.2") -> AIService:
    global _ai_services
    service = _ai_services.get(model)
    if service is None or getattr(service, "_service_version", None) != SERVICE_VERSION:
        service = AIService(model=model)
        _ai_services[model] = service
    return service
