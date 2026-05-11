from src.services.llama_service import LlamaService
from src.services.formula_registry import get_formula_registry


def _build_service():
    service = LlamaService.__new__(LlamaService)
    service.formula_registry = get_formula_registry()
    return service


def _row(filename: str, page: int, content: str):
    return {
        "document_id": f"{filename}-{page}",
        "filename": filename,
        "title": filename,
        "summary": "",
        "snippet": "",
        "content": content,
        "category": "other",
        "file_type": "pdf",
        "page_number": page,
        "score": 1.0,
    }


def test_inconsistency_workflow_output_structure():
    service = _build_service()
    rows = [
        {**_row("winst_verlies_test.pdf", 2, "winst-en-verliesrekening totale omzet 500000 kosten 300000"), "corpus_type": "uploaded"},
        {**_row("btw_aangifte_test.pdf", 1, "btw-overzicht belastbare omzet 420000"), "corpus_type": "uploaded"},
        {**_row("klantnotities_test.pdf", 1, "klantnotities correctie omzet nog niet verwerkt"), "corpus_type": "uploaded"},
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows)
    assert "Kort antwoord:" in answer
    assert "Mogelijke inconsistenties:" in answer
    assert "Volgende stap:" in answer
    assert "Bronnen:" in answer
    assert "1. Omzet volgens winst-en-verliesrekening sluit mogelijk niet aan op btw-overzicht" in answer
    assert "Bewijs A: Totale omzet volgens winst-en-verliesrekening:" in answer
    assert "500.000" in answer
    assert "Bewijs B: Btw-omzet / belastbare omzet volgens btw-overzicht:" in answer
    assert "420.000" in answer
    assert "Terminologie-signaal voor controle" not in answer


def test_inconsistency_verified_builder_prefers_statement_vs_vat_sources():
    service = _build_service()
    rows = [
        {**_row("winst_verlies_test.pdf", 1, "winst-en-verliesrekening totale omzet 372000"), "corpus_type": "uploaded"},
        {**_row("btw_aangifte_test.pdf", 1, "btw-overzicht kwartaal Q1 2025 88000 Q2 2025 96000 Q3 2025 91000 Q4 2025 97000"), "corpus_type": "uploaded"},
    ]
    answer = service._build_inconsistency_answer_verified(rows)
    assert "bron: winst_verlies_test.pdf, page 1" in answer
    assert "bron: btw_aangifte_test.pdf, page 1" in answer


def test_advisory_points_workflow_output_structure():
    service = _build_service()
    rows = [
        {**_row("balans_test.pdf", 1, "omzet marge kosten"), "corpus_type": "uploaded"},
        {**_row("btw_aangifte_test.pdf", 1, "btw omzetbelasting factuur"), "corpus_type": "uploaded"},
        {**_row("klantnotities_test.pdf", 1, "cash bank liquiditeit"), "corpus_type": "uploaded"},
    ]
    answer = service._build_workflow_answer("advisory_points", rows)
    assert "Adviespunten om met de klant te bespreken:" in answer
    assert "Belangrijke opmerking:" in answer
    assert "Bronnen:" in answer
    assert "Mogelijke verzekeringsrisico" not in answer


def test_insurance_risk_workflow_output_structure():
    service = _build_service()
    rows = [
        {**_row("verzekering_test.pdf", 1, "verzekering polis inventaris voorraad"), "corpus_type": "uploaded"},
        {**_row("contract_test.pdf", 2, "contract aansprakelijkheid service level"), "corpus_type": "uploaded"},
    ]
    answer = service._build_workflow_answer("insurance_risk_check", rows)
    assert "Mogelijke verzekeringsrisico" in answer
    assert "Ontbrekende informatie:" in answer
    assert "Bronnen:" in answer
    assert "Adviespunten om met de klant te bespreken:" not in answer


def test_client_file_summary_workflow_output_structure():
    service = _build_service()
    rows = [
        {**_row("klantnotities_test.pdf", 1, "klantnotities activiteiten software omzet"), "corpus_type": "uploaded"},
        {**_row("btw_aangifte_test.pdf", 1, "btw omzetbelasting"), "corpus_type": "uploaded"},
        {**_row("contract_test.pdf", 1, "contract agreement"), "corpus_type": "uploaded"},
    ]
    answer = service._build_workflow_answer("client_file_summary", rows)
    assert "Samenvatting klantdossier:" in answer
    assert "Bedrijfsactiviteit:" in answer
    assert "Belasting-/btw-punten:" in answer
    assert "Vervolgvragen:" in answer


def test_dutch_calc_detection_for_omzetgroei():
    service = _build_service()
    assert service._is_calc_intent_fast("Bereken de omzetgroei voor 2025")


def test_advisory_workflow_not_blocked_by_mkb_ticker_mismatch():
    service = _build_service()
    rows = [
        _row("reference_test.pdf", 1, "btw-overzicht controle"),
        _row("client_notes_test.pdf", 1, "mkb klantnotities omzet"),
    ]
    filtered, err, notes = service._enforce_source_consistency(
        "Welke drie adviespunten kan ik met deze MKB-klant bespreken op basis van de documenten?",
        rows,
        intent="advisory_points",
    )
    assert err is None
    assert filtered
    assert all("ticker: MKB" not in n for n in notes)


def test_advisor_workflow_prioritizes_uploaded_client_docs():
    service = _build_service()
    rows = [
        {
            **_row("reference_test.pdf", 1, "referentie btw guidance"),
            "corpus_type": "existing",
            "score": 2.5,
        },
        {
            **_row("balans_test.pdf", 1, "winst en verliesrekening omzet kosten"),
            "corpus_type": "uploaded",
            "score": 1.0,
        },
    ]
    ranked = service._prioritize_advisor_results(rows)
    assert ranked[0]["filename"] == "balans_test.pdf"


def test_ce_gap_workflow_returns_ce_sections():
    service = _build_service()
    rows = [
        {**_row("ce_note_test.pdf", 1, "CE conformiteit technisch dossier en risicobeoordeling"), "corpus_type": "uploaded"},
        {**_row("testreport_test.pdf", 2, "test report validatie"), "corpus_type": "uploaded"},
    ]
    answer = service._build_workflow_answer("ce_compliance_gap_check", rows)
    assert "Kort antwoord:" in answer
    assert "Gevonden informatie:" in answer
    assert "Aandachtspunten of ontbrekende informatie:" in answer
    assert "Mogelijke vervolgstap:" in answer


def test_quotation_workflow_warns_about_missing_price_or_scope():
    service = _build_service()
    rows = [
        {**_row("project_scope_test.pdf", 1, "scope technische eisen planning"), "corpus_type": "uploaded"},
    ]
    answer = service._build_workflow_answer("quotation_preparation", rows)
    assert "Kort antwoord:" in answer
    assert "Geen prijzen, doorlooptijden of toezeggingen invullen" in answer


def test_document_governance_workflow_returns_actionable_sections():
    service = _build_service()
    rows = [
        {**_row("governance_test.pdf", 1, "Revisie B, status draft, document owner projectleiding"), "corpus_type": "uploaded"},
        {**_row("governance_test.pdf", 2, "Volgens de norm moet de aannemer voldoen aan richtlijnen."), "corpus_type": "uploaded"},
        {**_row("governance_test.pdf", 3, "Gebruik van mobiele telefoon is verboden op de werkvloer."), "corpus_type": "uploaded"},
        {**_row("governance_test.pdf", 4, "Gebruik van mobiele telefoon is toegestaan in de werkzone."), "corpus_type": "uploaded"},
    ]
    answer = service._build_workflow_answer("document_governance_check", rows)
    assert "Kort antwoord:" in answer
    assert "Gevonden informatie:" in answer
    assert "Aandachtspunten of ontbrekende informatie:" in answer
    assert "Mogelijke vervolgstap:" in answer
    assert "Mogelijke tegenstrijdigheid" in answer
