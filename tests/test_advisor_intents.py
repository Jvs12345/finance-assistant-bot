from src.services.advisor_intents import detect_advisor_intent
from src.services.llama_service import LlamaService
from src.services.formula_registry import get_formula_registry
from src.services.financial_value_extractor import FinancialValueExtractor


def test_detects_missing_info_check_intent_dutch():
    question = "Welke informatie ontbreekt nog voordat de jaarrekening kan worden opgesteld?"
    assert detect_advisor_intent(question) == "missing_info_check"


def test_detects_inconsistency_check_intent_dutch():
    question = "Controleer of er inconsistenties zijn tussen de winst-en-verliesrekening, btw-overzicht en klantnotities."
    assert detect_advisor_intent(question) == "inconsistency_check"


def test_detects_advisory_points_intent_dutch():
    question = "Welke drie adviespunten kan ik met deze MKB-klant bespreken op basis van de documenten?"
    assert detect_advisor_intent(question) == "advisory_points"


def test_detects_insurance_risk_check_intent_dutch():
    question = "Welke verzekeringsrisico’s zie je op basis van de bedrijfsactiviteiten, activa en contractinformatie?"
    assert detect_advisor_intent(question) == "insurance_risk_check"


def test_detects_client_file_summary_intent():
    question = "Vat de belangrijkste punten uit dit klantdossier samen."
    assert detect_advisor_intent(question) == "client_file_summary"


def test_detects_financial_calculation_intent_when_flagged():
    question = "Bereken de omzetgroei als de benodigde cijfers beschikbaar zijn."
    assert detect_advisor_intent(question, is_calculation=True) == "financial_calculation"


def test_falls_back_to_normal_qna():
    question = "Welke bron noemde de btw deadline?"
    assert detect_advisor_intent(question) == "normal_qna"


def test_mkb_not_detected_as_ticker_in_constraints():
    service = LlamaService.__new__(LlamaService)
    service.formula_registry = get_formula_registry()
    service.value_extractor = FinancialValueExtractor(service.formula_registry)
    constraints = service._extract_question_constraints(
        "Welke drie adviespunten kan ik met deze MKB-klant bespreken op basis van de documenten?",
        intent="advisory_points",
    )
    assert "tickers" not in constraints
