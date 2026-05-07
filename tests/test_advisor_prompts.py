from src.services.llama_service import LlamaService
from src.services.formula_registry import get_formula_registry


def _build_service():
    service = LlamaService.__new__(LlamaService)
    service.formula_registry = get_formula_registry()
    return service


def test_missing_info_prompt_structure():
    service = _build_service()
    block = service._workflow_output_format("missing_info_check")
    assert "Gevonden informatie:" in block
    assert "Ontbrekende of onduidelijke informatie:" in block
    assert "Waarom dit belangrijk is:" in block
    assert "Bronnen:" in block


def test_inconsistency_prompt_structure():
    service = _build_service()
    block = service._workflow_output_format("inconsistency_check")
    assert "Mogelijke inconsistenties:" in block
    assert "Bewijs A:" in block
    assert "Bewijs B:" in block
    assert "Geen duidelijke inconsistentie gevonden:" in block


def test_advisory_points_prompt_structure():
    service = _build_service()
    block = service._workflow_output_format("advisory_points")
    assert "Adviespunten om met de klant te bespreken:" in block
    assert "Vraag aan de klant:" in block
    assert "Belangrijke opmerking:" in block


def test_insurance_risk_prompt_structure():
    service = _build_service()
    block = service._workflow_output_format("insurance_risk_check")
    assert "Mogelijke verzekeringsrisico" in block
    assert "Ontbrekende informatie:" in block
    assert "Wat te controleren:" in block


def test_client_file_summary_prompt_structure():
    service = _build_service()
    block = service._workflow_output_format("client_file_summary")
    assert "Samenvatting klantdossier:" in block
    assert "Bedrijfsactiviteit:" in block
    assert "Belasting-/btw-punten:" in block
    assert "Vervolgvragen:" in block


def test_create_prompt_uses_intent_format():
    service = _build_service()
    prompt = service._create_prompt(
        question="Welke informatie ontbreekt nog voordat de jaarrekening kan worden opgesteld?",
        context="--- Document 1 ---\nSource: test.pdf\nExtract:\nvoorbeeld",
        history=None,
        intent="missing_info_check",
    )
    assert "Workflow rules:" in prompt
    assert "Ontbrekende of onduidelijke informatie:" in prompt
