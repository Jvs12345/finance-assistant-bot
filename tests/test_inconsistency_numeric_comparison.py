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
        "tax_year": None,
    }


def test_inconsistency_compares_revenue_vs_vat_quarter_sum():
    service = _build_service()
    rows = [
        _row("financieel_overzicht.pdf", 1, "Winst-en-verliesrekening 2025 Totale omzet 372000"),
        _row("aangifte_omzetbelasting.pdf", 1, "Btw-overzicht 2025 Q1 2025 88000 Q2 2025 96000 Q3 2025 90000 Q4 2025 83000"),
        _row("klantnotities_test.pdf", 1, "Klantnotities omzet gestegen, periodeverschil mogelijk"),
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows, question="Controleer inconsistenties 2025")
    assert "Bewijs A: Totale omzet volgens winst-en-verliesrekening: €372.000" in answer
    assert "Btw-omzet totaal (afgeleid: Q1+Q2+Q3+Q4): €357.000" in answer
    assert "Verschil: €372.000 - €357.000 = €15.000" in answer


def test_inconsistency_does_not_compare_different_years():
    service = _build_service()
    rows = [
        {**_row("financieel_overzicht.pdf", 1, "Winst-en-verliesrekening Totale omzet 372000"), "tax_year": 2025},
        {**_row("aangifte_omzetbelasting.pdf", 1, "Btw-overzicht belastbare omzet 350000"), "tax_year": 2024},
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows, question="Controleer inconsistenties")
    assert "Waarden lijken uit verschillende jaren te komen" in answer


def test_inconsistency_ignores_vat_payable_as_turnover():
    service = _build_service()
    rows = [
        _row("financieel_overzicht.pdf", 1, "Winst-en-verliesrekening Totale omzet 372000"),
        _row("aangifte_omzetbelasting.pdf", 1, "Btw-overzicht Te betalen btw 9000 Voorbelasting 11000"),
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows, question="Controleer inconsistenties")
    assert "Btw-omzet uit btw-overzicht niet gevonden." in answer
    assert "Alleen btw-bedragen zoals te betalen btw/voorbelasting gevonden" in answer


def test_inconsistency_parses_dutch_number_format():
    service = _build_service()
    rows = [
        _row("financieel_overzicht.pdf", 1, "Winst-en-verliesrekening Totale omzet 372.000"),
        _row("aangifte_omzetbelasting.pdf", 1, "Btw-overzicht belastbare omzet 357.000"),
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows, question="Controleer inconsistenties")
    assert "€372.000" in answer
    assert "€357.000" in answer
    assert "€15.000" in answer


def test_inconsistency_works_with_generic_filenames():
    service = _build_service()
    rows = [
        _row("bestand_a.pdf", 1, "profit and loss total revenue 120000"),
        _row("bestand_b.pdf", 1, "vat overview taxable turnover 100000"),
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows, question="check consistency")
    assert "Bewijs A: Totale omzet volgens winst-en-verliesrekening" in answer
    assert "Bewijs B:" in answer


def test_inconsistency_prefers_cross_source_comparison_when_available():
    service = _build_service()
    rows = [
        _row("financieel_overzicht.pdf", 1, "winst-en-verliesrekening 2025 Totale omzet 372000"),
        _row("aangifte_omzetbelasting.pdf", 1, "btw-overzicht 2025 Totale omzet 350000 belastbare omzet 357000"),
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows, question="Controleer inconsistenties 2025")
    assert "bron: financieel_overzicht.pdf, page 1" in answer
    assert "bron: aangifte_omzetbelasting.pdf, page 1" in answer


def test_inconsistency_does_not_double_count_repeated_quarters():
    service = _build_service()
    rows = [
        _row("financieel_overzicht.pdf", 1, "Winst-en-verliesrekening 2025 Totale omzet 372000"),
        _row(
            "aangifte_omzetbelasting.pdf",
            1,
            "Btw-overzicht 2025 Q1 2025 88000 Q2 2025 96000 Q3 2025 90000 Q4 2025 83000 "
            "Q1 2025 88000 Q2 2025 96000 Q3 2025 90000 Q4 2025 83000",
        ),
    ]
    answer = service._build_workflow_answer("inconsistency_check", rows, question="Controleer inconsistenties 2025")
    assert "Btw-omzet totaal (afgeleid: Q1+Q2+Q3+Q4):" in answer
    assert "357.000" in answer
