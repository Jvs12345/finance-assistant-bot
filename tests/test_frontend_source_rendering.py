from pathlib import Path


def test_frontend_strips_trailing_sources_section():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert "function stripTrailingSourceSection" in html
    assert "Bronnen|Sources" in html


def test_frontend_uses_dutch_sources_heading():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert "Gebruikte bronnen" in html


def test_frontend_keeps_source_cards_with_source_payload():
    html = Path("static/index.html").read_text(encoding="utf-8")
    assert "renderSourceCards(sources, visibleText)" in html
