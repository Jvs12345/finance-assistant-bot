from src.services.llama_service import LlamaService
from src.services.formula_registry import get_formula_registry


def _build_service():
    service = LlamaService.__new__(LlamaService)
    service.formula_registry = get_formula_registry()
    return service


def test_missing_info_retrieval_query_expands():
    service = _build_service()
    question = "Welke informatie ontbreekt nog voordat de jaarrekening kan worden opgesteld?"
    expanded = service._build_intent_retrieval_query(question, "missing_info_check")
    assert expanded != question
    assert "btw-overzicht" in expanded.lower()
    assert "winst-en-verliesrekening" in expanded.lower()


def test_inconsistency_retrieval_query_expands():
    service = _build_service()
    question = "Controleer of er inconsistenties zijn tussen documenten."
    expanded = service._build_intent_retrieval_query(question, "inconsistency_check")
    assert expanded != question
    assert "btw-overzicht" in expanded.lower()
    assert "winst-en-verliesrekening" in expanded.lower()


def test_normal_qna_query_not_expanded():
    service = _build_service()
    question = "Welke btw regels zijn relevant?"
    expanded = service._build_intent_retrieval_query(question, "normal_qna")
    assert expanded == question


def test_financial_calculation_query_not_expanded():
    service = _build_service()
    question = "Bereken de omzetgroei voor 2025."
    expanded = service._build_intent_retrieval_query(question, "financial_calculation")
    assert expanded == question
