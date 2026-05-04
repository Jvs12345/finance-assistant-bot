from pathlib import Path

from src.services.document_indexing_service import DocumentIndexingService


def test_document_indexing_infers_filters(monkeypatch, tmp_path):
    svc = DocumentIndexingService()

    sample_pages = [
        {
            "page": 1,
            "content": (
                "Alphabet Inc annual report 2026. "
                "Consolidated statements. Cost of revenues 36,361 41,271. "
                "United States filing."
            ),
        }
    ]

    monkeypatch.setattr(svc.processor, "extract_pages_from_pdf", lambda _: sample_pages)

    pdf_path = tmp_path / "2026q1-alphabet-annual-report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

    prepared = svc.prepare_pdf_document(
        file_path=pdf_path,
        document_id="doc-1",
        source_filename=pdf_path.name,
        category="other",
        metadata={},
    )

    assert prepared["metadata"]["tax_year"] == 2026
    assert prepared["metadata"]["document_type"] == "annual_report"
    assert prepared["metadata"]["jurisdiction"] in ("United States", None)
