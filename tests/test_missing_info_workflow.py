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


def test_missing_info_workflow_structured_output_and_phrase():
    service = _build_service()
    rows = [
        {**_row("checklist_test.pdf", 1, "jaarrekening checklist balans winst-en-verliesrekening btw bankafschrift"), "corpus_type": "existing"},
        {**_row("btw_aangifte_test.pdf", 2, "btw-overzicht omzetbelasting aansluiting"), "corpus_type": "uploaded"},
        {**_row("contract_test.pdf", 3, "contract overeenkomst payment terms"), "corpus_type": "uploaded"},
    ]
    answer = service._build_missing_info_answer(rows)
    assert "Op basis van de beschikbare documenten lijken meerdere jaarrekening-onderdelen nog onduidelijk of vragen ze extra onderbouwing." in answer
    assert "Gevonden informatie:" in answer
    assert "Ontbrekende of onduidelijke informatie:" in answer
    assert "Bronnen:" in answer


def test_missing_info_dutch_output_has_no_english_helper_labels():
    service = _build_service()
    rows = [
        {**_row("balans_test.pdf", 1, "bankafschrift factuur voorraad afschrijving lening btw contract polis"), "corpus_type": "uploaded"},
        {**_row("checklist_test.pdf", 1, "jaarrekening checklist bankafschrift factuur voorraad"), "corpus_type": "existing"},
    ]
    answer = service._build_missing_info_answer(rows).lower()
    assert "bank statement support" not in answer
    assert "invoice support" not in answer
    assert "inventory valuation" not in answer


def test_missing_info_workflow_marks_missing_items_without_contradiction():
    service = _build_service()
    rows = [
        {**_row("balans_test.pdf", 1, "winst-en-verliesrekening omzet kosten"), "corpus_type": "uploaded"},
        {**_row("jaarrekening_checklist_test.pdf", 1, "vereist: bankafschriften verkoopfacturen inkoopfacturen"), "corpus_type": "existing"},
    ]
    answer = service._build_missing_info_answer(rows)
    assert "aandachtspunt op basis van checklist" in answer
    assert "Geen expliciet ontbrekende punten gevonden" not in answer
