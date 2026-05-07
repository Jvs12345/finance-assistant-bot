from src.services.verification_service import AnswerVerificationService


def _row(filename: str, page: int, content: str):
    return {
        "filename": filename,
        "page_number": page,
        "content": content,
        "snippet": content,
    }


def test_verification_fails_on_wrong_arithmetic():
    svc = AnswerVerificationService()
    rows = [_row("doc.pdf", 1, "totale omzet 372000 belastbare omzet 357000")]
    answer = (
        "Mogelijke inconsistenties:\n"
        "1. Test\n"
        "- Bewijs A: Totale omzet: 372000, bron: doc.pdf, page 1\n"
        "- Bewijs B: Btw-omzet: 357000, bron: doc.pdf, page 1\n"
        "- Verschil: 372000 - 357000 = 9999\n"
    )
    res = svc.verify("Controleer inconsistenties", answer, [], rows)
    assert res.status == "fail"
    assert any("Rekencontrole mislukt" in i for i in res.issues)


def test_verification_fails_on_missing_source_reference():
    svc = AnswerVerificationService()
    rows = [_row("doc.pdf", 1, "totale omzet 372000")]
    answer = "- Bewijs A: Totale omzet: 372000, bron: ander.pdf, page 1"
    res = svc.verify("Controleer inconsistenties", answer, [], rows)
    assert res.status == "fail"
    assert any("Bronverwijzing niet gevonden" in i for i in res.issues)


def test_verification_passes_on_valid_lines():
    svc = AnswerVerificationService()
    rows = [_row("doc.pdf", 1, "totale omzet 372000 belastbare omzet 357000")]
    answer = (
        "- Bewijs A: Totale omzet: 372000, bron: doc.pdf, page 1\n"
        "- Bewijs B: Btw-omzet: 357000, bron: doc.pdf, page 1\n"
        "- Verschil: 372000 - 357000 = 15000\n"
    )
    res = svc.verify("Controleer inconsistenties", answer, [], rows)
    assert res.status == "pass"


def test_verification_prefers_full_content_over_short_snippet():
    svc = AnswerVerificationService()
    search_rows = [_row("doc.pdf", 1, "totale omzet 372000 belastbare omzet 357000")]
    source_cards = [{
        "filename": "doc.pdf",
        "page": 1,
        "snippet": "korte snippet zonder bedragen",
    }]
    answer = (
        "- Bewijs A: Totale omzet: 372000, bron: doc.pdf, page 1\n"
        "- Bewijs B: Btw-omzet: 357000, bron: doc.pdf, page 1\n"
        "- Verschil: 372000 - 357000 = 15000\n"
    )
    res = svc.verify("Controleer inconsistenties", answer, source_cards, search_rows)
    assert res.status == "pass"
