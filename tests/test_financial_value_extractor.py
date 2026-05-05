from src.services.formula_registry import get_formula_registry
from src.services.financial_value_extractor import FinancialValueExtractor
import pytest


def _formula(formula_id: str):
    registry = get_formula_registry()
    for f in registry.list_formulas():
        if f.id == formula_id:
            return f
    raise AssertionError(f"Formula not found: {formula_id}")


def _row(content: str, score: float = 1.0, filename: str = "doc.pdf", page: int = 1):
    return {
        "document_id": "d1",
        "filename": filename,
        "title": filename,
        "content": content,
        "summary": "",
        "snippet": "",
        "page_number": page,
        "score": score,
        "category": "annual_report",
        "file_type": "pdf",
    }


def test_revenue_rejects_deferred_revenue():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("gross_profit")
    rows = [
        _row("CONSOLIDATED BALANCE SHEETS\nDeferred revenue 408,963", score=5.0, filename="pltr.pdf"),
        _row("CONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue 2,866", score=1.0, filename="pltr.pdf"),
    ]
    val = extractor.resolve_best_value("revenue", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "deferred revenue" not in val.line_text.lower()
    assert "revenue 2,866".replace(",", "") in val.line_text.lower().replace(",", "")


def test_operating_income_rejects_operating_expenses():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("operating_margin_pct")
    rows = [
        _row("CONSOLIDATED STATEMENTS OF INCOME\nOperating expenses 800", score=4.0),
        _row("CONSOLIDATED STATEMENTS OF INCOME\nOperating income 1,200", score=1.0),
    ]
    val = extractor.resolve_best_value("operating_income", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "operating income" in val.line_text.lower()
    assert "expenses" not in val.line_text.lower()


def test_total_assets_rejects_total_current_assets():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("debt_ratio")
    rows = [
        _row("CONSOLIDATED BALANCE SHEETS\nTotal current assets 5,000", score=3.0),
        _row("CONSOLIDATED BALANCE SHEETS\nTotal assets 8,000", score=1.0),
    ]
    val = extractor.resolve_best_value("total_assets", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "total assets" in val.line_text.lower()
    assert "current assets" not in val.line_text.lower()


def test_total_liabilities_rejects_current_and_deferred():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("debt_ratio")
    rows = [
        _row("CONSOLIDATED BALANCE SHEETS\nTotal current liabilities 1,000", score=4.0),
        _row("CONSOLIDATED BALANCE SHEETS\nDeferred revenue 400", score=3.5),
        _row("CONSOLIDATED BALANCE SHEETS\nTotal liabilities 3,000", score=1.0),
    ]
    val = extractor.resolve_best_value("total_liabilities", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "total liabilities" in val.line_text.lower()
    assert "current liabilities" not in val.line_text.lower()
    assert "deferred revenue" not in val.line_text.lower()


def test_cash_flow_from_operations_rejects_cash_equivalents():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("free_cash_flow")
    rows = [
        _row("CONSOLIDATED BALANCE SHEETS\nCash and cash equivalents 2,000", score=4.0),
        _row("CONSOLIDATED STATEMENTS OF CASH FLOWS\nNet cash provided by operating activities 1,000", score=1.0),
    ]
    val = extractor.resolve_best_value("operating_cash_flow", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "operating activities" in val.line_text.lower()


@pytest.mark.parametrize(
    "variable,formula_id,accepted_line,rejected_line",
    [
        ("revenue", "gross_profit", "Revenue 2,866", "Accounts receivable 500"),
        ("revenue", "gross_profit", "Total revenue 2,866", "Remaining performance obligations 1,200"),
        ("cogs", "gross_profit", "Cost of revenue 567", "Deferred revenue 408,963"),
        ("cogs", "gross_profit", "Cost of sales 567", "Revenue 2,866"),
        ("gross_profit", "gross_margin_pct", "Gross profit 2,299", "Operating income 1,200"),
        ("operating_income", "operating_margin_pct", "Income from operations 1,200", "Operating lease liabilities 300"),
        ("net_income", "net_margin_pct", "Net income 900", "Comprehensive income 950"),
        ("net_income", "net_margin_pct", "Net earnings 900", "Accumulated deficit 2,000"),
        ("total_equity", "debt_to_equity", "Total stockholders equity 5,000", "Additional paid in capital 2,000"),
        ("operating_cash_flow", "free_cash_flow", "Net cash provided by operating activities 1,000", "Operating income 900"),
    ],
)
def test_variable_rejects_false_positive_labels(variable, formula_id, accepted_line, rejected_line):
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula(formula_id)
    rows = [
        _row(f"CONSOLIDATED STATEMENTS OF OPERATIONS\n{rejected_line}", score=5.0),
        _row(f"CONSOLIDATED STATEMENTS OF OPERATIONS\n{accepted_line}", score=1.0),
    ]
    if variable in {"total_equity"}:
        rows = [
            _row(f"CONSOLIDATED BALANCE SHEETS\n{rejected_line}", score=5.0),
            _row(f"CONSOLIDATED BALANCE SHEETS\n{accepted_line}", score=1.0),
        ]
    if variable in {"operating_cash_flow"}:
        rows = [
            _row(f"CONSOLIDATED STATEMENTS OF INCOME\n{rejected_line}", score=5.0),
            _row(f"CONSOLIDATED STATEMENTS OF CASH FLOWS\n{accepted_line}", score=1.0),
        ]
    val = extractor.resolve_best_value(variable, formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert accepted_line.lower().split()[0] in val.line_text.lower()
    assert rejected_line.lower() not in val.line_text.lower()


def test_section_preference_income_metric_rejects_balance_sheet_when_better_income_exists():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("gross_profit")
    rows = [
        _row("CONSOLIDATED BALANCE SHEETS\nRevenue 2,866", score=5.0),
        _row("CONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue 2,866", score=1.0),
    ]
    val = extractor.resolve_best_value("revenue", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "operations" in rows[1]["content"].lower()
    assert "operations" in val.source.get("content", "").lower()


def test_section_preference_balance_metric_rejects_income_statement():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("debt_ratio")
    rows = [
        _row("CONSOLIDATED STATEMENTS OF INCOME\nTotal assets 8,000", score=5.0),
        _row("CONSOLIDATED BALANCE SHEETS\nTotal assets 8,000", score=1.0),
    ]
    val = extractor.resolve_best_value("total_assets", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "balance sheets" in val.source.get("content", "").lower()


def test_section_preference_cash_flow_metric_rejects_income_and_balance():
    registry = get_formula_registry()
    extractor = FinancialValueExtractor(registry)
    formula = _formula("free_cash_flow")
    rows = [
        _row("CONSOLIDATED STATEMENTS OF INCOME\nNet cash provided by operating activities 1,000", score=4.0),
        _row("CONSOLIDATED BALANCE SHEETS\nNet cash provided by operating activities 1,000", score=3.0),
        _row("CONSOLIDATED STATEMENTS OF CASH FLOWS\nNet cash provided by operating activities 1,000", score=1.0),
    ]
    val = extractor.resolve_best_value("operating_cash_flow", formula, rows, desired_year=None, company_target=None)
    assert val is not None
    assert "cash flows" in val.source.get("content", "").lower()
