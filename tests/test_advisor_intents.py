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


def test_detects_short_insurance_risk_question_before_generic_risk():
    question = "Welke verzekeringsrisico's zie je?"
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


def test_summary_question_stays_normal_qna():
    question = "Vat dit document samen."
    assert detect_advisor_intent(question) == "document_summary"


def test_summary_question_variants_detect_document_summary():
    samples = [
        "Vat bestand samen",
        "Vat het document samen",
        "Geef een samenvatting van dit document",
        "Waar gaat dit document over?",
        "summarise this document",
    ]
    for question in samples:
        assert detect_advisor_intent(question) == "document_summary"


def test_mkb_not_detected_as_ticker_in_constraints():
    service = LlamaService.__new__(LlamaService)
    service.formula_registry = get_formula_registry()
    service.value_extractor = FinancialValueExtractor(service.formula_registry)
    constraints = service._extract_question_constraints(
        "Welke drie adviespunten kan ik met deze MKB-klant bespreken op basis van de documenten?",
        intent="advisory_points",
    )
    assert "tickers" not in constraints


def test_common_role_acronyms_not_detected_as_tickers():
    service = LlamaService.__new__(LlamaService)
    constraints = service._extract_question_constraints(
        "Wat is de kleur van de auto van de CEO volgens deze documenten?"
    )
    assert "tickers" not in constraints


def test_engineering_demo_question_intent_routing():
    mapping = {
        "Vat dit document samen in maximaal 8 punten.": "document_summary",
        "Welke technische eisen of verplichtingen worden genoemd?": "technical_requirements_check",
        "Welke risico’s of aandachtspunten staan in het document?": "risk_attention_check",
        "Welke informatie ontbreekt nog voordat CE-documentatie compleet is?": "ce_compliance_gap_check",
        "Welke informatie is nodig om een eerste offerte op te stellen?": "quotation_preparation",
        "Welke AI-use-cases zie je op basis van deze documenten?": "ai_use_case_identification",
        "Scoor deze use-cases op impact, haalbaarheid en databeschikbaarheid.": "use_case_prioritization",
        "Werk de beste use-case uit als pilotproject voor studenten: wat is het probleem, welke data is nodig, wie is betrokken en wat kan een prototype opleveren?": "pilot_project_translation",
    }
    for question, expected in mapping.items():
        assert detect_advisor_intent(question) == expected


def test_document_governance_intent_detection_variants():
    samples = [
        "Welke metadata ontbreekt in dit document voor audit trail?",
        "Is het versiebeheer en de revisiestatus duidelijk vastgelegd?",
        "Zitten er tegenstrijdige instructies in de procedure?",
        "Welke normen worden genoemd zonder versie of jaartal?",
        "Wat moet de documentbeheerder prioriteren in de volgende revisie?",
    ]
    for question in samples:
        assert detect_advisor_intent(question) == "document_governance_check"
