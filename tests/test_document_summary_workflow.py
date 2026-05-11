from src.services.advisor_intents import detect_advisor_intent
from src.services.llama_service import LlamaService
from src.services.formula_registry import get_formula_registry


def _build_service():
    service = LlamaService.__new__(LlamaService)
    service.formula_registry = get_formula_registry()
    service.max_context_chars = 7500
    service.max_chars_per_chunk = 1500
    service.final_context_chunks = 8
    return service


def _row(filename: str, page: int, title: str, content: str, score: float = 1.0):
    return {
        "document_id": f"{filename}-{page}",
        "filename": filename,
        "title": title,
        "summary": "",
        "snippet": "",
        "content": content,
        "category": "other",
        "file_type": "pdf",
        "page_number": page,
        "chunk_index": page,
        "score": score,
        "corpus_type": "uploaded",
    }


def test_document_summary_intent_detection_examples():
    samples = [
        "Vat bestand samen",
        "Vat het document samen",
        "Geef een samenvatting van dit document",
        "Waar gaat dit document over?",
        "summarise this document",
    ]
    for question in samples:
        assert detect_advisor_intent(question) == "document_summary"


def test_document_summary_detection_not_single_phrase_bound():
    variants = [
        "Maak een korte samenvatting",
        "Wat staat er in dit document",
        "Summarize this document",
        "Give a document summary",
        "What is this document about?",
    ]
    for question in variants:
        assert detect_advisor_intent(question) == "document_summary"


def test_ocr_cleanup_reduces_broken_fragments_in_summary_context():
    service = _build_service()
    rows = [
        _row(
            "generic_contract.pdf",
            1,
            "Intro",
            "vanoersannen Veiligheids- en Gezondheidsplan =< Tae ~~ ~~ page 1 1 1"
            " De contractor moet een risicoanalyse uitvoeren en PBM dragen.",
        ),
        _row(
            "generic_contract.pdf",
            2,
            "Verplichtingen",
            "De aannemer dient incidenten te melden en toegangsinstructies te volgen.",
        ),
    ]
    context, _noise = service._build_document_summary_context(rows)
    assert "=< Tae" not in context
    assert "~~ ~~" not in context
    assert "contractor moet een risicoanalyse uitvoeren" in context.lower()


def test_document_summary_output_structure_dutch():
    service = _build_service()
    block = service._workflow_output_format("document_summary", question="Vat dit document samen")
    assert "Kort antwoord:" in block
    assert "Belangrijkste onderdelen:" in block
    assert "Belangrijke verplichtingen of aandachtspunten:" in block
    assert "Mogelijke acties voor de gebruiker:" in block


def test_document_summary_answer_removes_trailing_sources_section():
    service = _build_service()
    answer = (
        "Kort antwoord:\n"
        "Dit document beschrijft veiligheidsvereisten.\n\n"
        "Belangrijkste onderdelen:\n"
        "- Introductie\n\n"
        "Belangrijke verplichtingen of aandachtspunten:\n"
        "- Meld incidenten\n\n"
        "Mogelijke acties voor de gebruiker:\n"
        "- Controleer PBM-lijst\n\n"
        "Bronnen:\n"
        "- generic_contract.pdf, page 2"
    )
    cleaned = service._post_process_document_summary_answer(
        answer=answer,
        question="Vat dit document samen",
        noisy_context=False,
    )
    assert "Bronnen:" not in cleaned


def test_document_summary_generalizes_across_document_types():
    service = _build_service()
    rows = [
        _row("generic_policy.pdf", 1, "Introduction", "Purpose and scope of this policy document.", 3.0),
        _row("generic_policy.pdf", 14, "Risk Controls", "Mandatory controls and mitigation actions.", 2.8),
        _row("generic_policy.pdf", 28, "Conclusion", "Final responsibilities and review cycle.", 2.6),
        _row("generic_manual.pdf", 3, "Requirements", "Operators must follow checklist before start.", 2.2),
    ]
    selected = service._select_representative_summary_rows(rows, retrieval_limit=8)
    selected_names = {r["filename"] for r in selected}
    assert "generic_policy.pdf" in selected_names
    assert len(selected) >= 3


def test_document_summary_pbm_in_safety_context_not_preventief_beheer():
    service = _build_service()
    answer = (
        "Kort antwoord:\n"
        "Dit document gaat over veiligheid op de werkvloer.\n\n"
        "Belangrijkste onderdelen:\n"
        "- PBM (Preventief Beheer)\n"
        "- Incidentmelding\n"
    )
    cleaned = service._post_process_document_summary_answer(
        answer=answer,
        question="Vat dit veiligheidsdocument samen",
        noisy_context=False,
    )
    assert "Preventief Beheer" not in cleaned
    assert "PBM (persoonlijke beschermingsmiddelen)" in cleaned


def test_document_summary_fixes_common_dutch_character_issues():
    service = _build_service()
    answer = (
        "Kort antwoord:\n"
        "Het systeem is geÄ«mplementeerd en moet jaarlijks worden gecontroleerd."
    )
    cleaned = service._post_process_document_summary_answer(
        answer=answer,
        question="Vat dit document samen",
        noisy_context=False,
    )
    assert "geÄ«mplementeerd" not in cleaned
    assert "geïmplementeerd" in cleaned


