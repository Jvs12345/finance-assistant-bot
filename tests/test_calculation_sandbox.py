from src.services.calculation_sandbox import CalculationSandbox, CalculationSandboxError


def test_calculation_sandbox_basic_expression():
    value = CalculationSandbox.evaluate(
        expression="(revenue_2026 - revenue_2025) / revenue_2025 * 100",
        variables={"revenue_2026": 109896, "revenue_2025": 90234},
    )
    assert round(value, 4) == round(((109896 - 90234) / 90234) * 100, 4)


def test_calculation_sandbox_blocks_unsafe_syntax():
    try:
        CalculationSandbox.evaluate(
            expression="__import__('os').system('echo bad')",
            variables={},
        )
    except CalculationSandboxError:
        return
    raise AssertionError("Unsafe expression was not blocked")
