from src.services.formula_registry import get_formula_registry


def test_formula_registry_loads():
    registry = get_formula_registry()
    formulas = registry.list_formulas()
    assert len(formulas) >= 20


def test_formula_registry_matches_question():
    registry = get_formula_registry()
    matches = registry.find_by_question("What is the gross margin and COGS?")
    ids = [m.id for m in matches]
    assert any(i in ids for i in ["gross_margin_pct", "gross_profit"])
