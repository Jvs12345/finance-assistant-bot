from src.services.formula_registry import get_formula_registry
from src.services.llama_service import LlamaService
from src.services.financial_value_extractor import FinancialValueExtractor


class _DummyES:
    def search(self, **kwargs):
        return []


def _build_service():
    service = LlamaService.__new__(LlamaService)
    service.model = "llama3.2"
    service.formula_registry = get_formula_registry()
    service.value_extractor = FinancialValueExtractor(service.formula_registry)
    service.es_client = _DummyES()
    return service


def test_formula_revenue_growth_uses_matching_company_only():
    service = _build_service()
    rows = [
        {
            "document_id": "1",
            "filename": "alphabet-2026.pdf",
            "title": "Alphabet Inc",
            "content": "Quarter Ended March 31, 2025 2026 Revenues 90,234 109,896",
            "page_number": 5,
            "score": 1.2,
            "category": "annual_report",
            "file_type": "pdf",
        },
        {
            "document_id": "2",
            "filename": "palantir-2026.pdf",
            "title": "Palantir",
            "content": "Year Ended December 31, 2025 2026 Revenue 2,225 2,866",
            "page_number": 4,
            "score": 1.0,
            "category": "annual_report",
            "file_type": "pdf",
        },
    ]

    out, _, _ = service._enforce_source_consistency("Calculate Palantir revenue growth for 2026", rows)
    calc = service._try_formula_registry_calculation(
        "Calculate Palantir revenue growth for 2026",
        out,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert calc is not None
    assert "Revenue Growth %" in calc["answer"]
    assert "palantir-2026.pdf" in calc["answer"].lower()
    assert all("alphabet" not in s.get("filename", "").lower() for s in calc["sources"])


def test_formula_compare_two_companies_keeps_sources_separate():
    service = _build_service()
    rows = [
        {
            "document_id": "1",
            "filename": "alphabet-2026.pdf",
            "title": "Alphabet Inc",
            "content": "Quarter Ended March 31, 2025 2026 Revenues 90,234 109,896",
            "page_number": 5,
            "score": 1.2,
            "category": "annual_report",
            "file_type": "pdf",
        },
        {
            "document_id": "2",
            "filename": "palantir-2026.pdf",
            "title": "Palantir",
            "content": "Year Ended December 31, 2025 2026 Revenue 2,225 2,866",
            "page_number": 4,
            "score": 1.0,
            "category": "annual_report",
            "file_type": "pdf",
        },
    ]
    calc = service._try_formula_registry_calculation(
        "Compare Palantir and Alphabet revenue growth for 2026",
        rows,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert calc is not None
    answer = calc["answer"].lower()
    assert "alphabet:" in answer
    assert "palantir:" in answer
    assert "alphabet-2026.pdf" in answer
    assert "palantir-2026.pdf" in answer


def test_formula_cogs_can_be_derived():
    service = _build_service()
    rows = [
        {
            "document_id": "2",
            "filename": "palantir-2026.pdf",
            "title": "Palantir",
            "content": "Year Ended 2025 2026 Revenue 2,225 2,866 Gross profit 1,800 2,300",
            "page_number": 4,
            "score": 1.0,
            "category": "annual_report",
            "file_type": "pdf",
        }
    ]
    calc = service._try_formula_registry_calculation(
        "Calculate Palantir COGS for 2026",
        rows,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert calc is not None
    answer = calc["answer"].lower()
    assert "cost of revenue / cogs" in answer
    assert "derived values" in answer
    assert "revenue - gross_profit" in answer


def test_formula_cogs_does_not_use_deferred_revenue():
    service = _build_service()
    rows = [
        {
            "document_id": "2",
            "filename": "pltr-2026.pdf",
            "title": "Palantir",
            "content": (
                "CONSOLIDATED BALANCE SHEETS\nDeferred revenue 408,963\n"
                "CONSOLIDATED STATEMENTS OF OPERATIONS\nYear Ended 2025 2026 Revenue 2,225 2,866 Gross profit 1,800 2,300"
            ),
            "page_number": 4,
            "score": 1.0,
            "category": "annual_report",
            "file_type": "pdf",
        }
    ]
    calc = service._try_formula_registry_calculation(
        "Calculate Palantir COGS for 2026",
        rows,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert calc is not None
    answer = calc["answer"].lower()
    assert "deferred revenue" not in answer
    assert "derived" in answer


def test_formula_revenue_growth_does_not_use_deferred_revenue():
    service = _build_service()
    rows = [
        {
            "document_id": "2",
            "filename": "pltr-2026.pdf",
            "title": "Palantir",
            "content": (
                "CONSOLIDATED BALANCE SHEETS\nDeferred revenue 408,963\n"
                "CONSOLIDATED STATEMENTS OF OPERATIONS\nYear Ended December 31, 2025 2026 Revenue 2,225 2,866"
            ),
            "page_number": 4,
            "score": 1.0,
            "category": "annual_report",
            "file_type": "pdf",
        }
    ]
    calc = service._try_formula_registry_calculation(
        "Calculate Palantir revenue growth for 2026",
        rows,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert calc is not None
    answer = calc["answer"].lower()
    assert "deferred revenue" not in answer
    assert "revenue growth %" in answer
