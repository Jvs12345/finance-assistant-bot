from pathlib import Path

from src.services.document_indexing_service import DocumentIndexingService


def _prepare(monkeypatch, tmp_path, filename, page_text):
    svc = DocumentIndexingService()
    monkeypatch.setattr(svc.processor, "extract_pages_from_pdf", lambda _: [{"page": 1, "content": page_text}])
    pdf_path = tmp_path / filename
    pdf_path.write_bytes(b"%PDF-1.4\n%test\n")
    return svc.prepare_pdf_document(
        file_path=pdf_path,
        document_id="doc-1",
        source_filename=pdf_path.name,
        category="other",
        metadata={},
    )


def test_dutch_profit_loss_not_inferred_as_invoice(monkeypatch, tmp_path):
    prepared = _prepare(
        monkeypatch,
        tmp_path,
        "balans_test.pdf",
        "winst-en-verliesrekening totale omzet 372000 kosten 250000 btw-overzicht",
    )
    assert prepared["metadata"]["document_type"] != "invoice"
    assert prepared["metadata"]["document_type_detail"] == "profit_loss"
    assert prepared["metadata"]["jurisdiction"] == "Netherlands"


def test_vat_overview_and_dutch_jurisdiction(monkeypatch, tmp_path):
    prepared = _prepare(
        monkeypatch,
        tmp_path,
        "btw_aangifte_test.pdf",
        "btw-overzicht belastbare omzet omzet hoog tarief verschuldigde btw voorbelasting",
    )
    assert prepared["metadata"]["document_type_detail"] == "vat_overview"
    assert prepared["metadata"]["jurisdiction"] == "Netherlands"


def test_insurance_document_inference(monkeypatch, tmp_path):
    prepared = _prepare(
        monkeypatch,
        tmp_path,
        "verzekering_test.pdf",
        "polisoverzicht verzekering dekking aansprakelijkheid verzekerde som",
    )
    assert prepared["metadata"]["document_type_detail"] == "insurance_document"
    assert prepared["metadata"]["jurisdiction"] == "Netherlands"


def test_us_jurisdiction_requires_clear_us_indicators(monkeypatch, tmp_path):
    prepared = _prepare(
        monkeypatch,
        tmp_path,
        "annual_report_test.pdf",
        "SEC Form 10-K Nasdaq United States GAAP revenue",
    )
    assert prepared["metadata"]["jurisdiction"] == "United States"
