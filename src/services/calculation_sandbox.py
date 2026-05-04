"""
Lightweight local calculation sandbox.

Evaluates arithmetic expressions safely using Python AST with a strict allowlist.
No file, network, imports, or function calls are permitted.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict


class CalculationSandboxError(ValueError):
    """Raised when expression or variables are invalid for sandbox execution."""


@dataclass
class CalculationResult:
    expression: str
    variables: Dict[str, float]
    value: float
    label: str | None = None
    unit: str | None = None


class CalculationSandbox:
    """
    Safe arithmetic evaluator.

    Allowed:
    - literals (int/float)
    - variable names
    - +, -, *, /, //, %, **
    - unary + and -
    """

    _ALLOWED_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Name,
        ast.Load,
        ast.Constant,
    }

    @classmethod
    def evaluate(cls, expression: str, variables: Dict[str, Any]) -> float:
        if not isinstance(expression, str) or not expression.strip():
            raise CalculationSandboxError("Expression must be a non-empty string.")

        safe_vars: Dict[str, float] = {}
        for key, value in (variables or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise CalculationSandboxError("Variable names must be non-empty strings.")
            if not isinstance(value, (int, float)):
                raise CalculationSandboxError(f"Variable '{key}' must be numeric.")
            safe_vars[key] = float(value)

        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise CalculationSandboxError(f"Invalid expression syntax: {exc}") from exc

        for node in ast.walk(parsed):
            if type(node) not in cls._ALLOWED_NODES:
                raise CalculationSandboxError(
                    f"Disallowed syntax in expression: {type(node).__name__}"
                )
            if isinstance(node, ast.Name) and node.id not in safe_vars:
                raise CalculationSandboxError(f"Unknown variable in expression: {node.id}")

        try:
            value = eval(  # noqa: S307 - protected by strict AST validation
                compile(parsed, "<calc_sandbox>", "eval"),
                {"__builtins__": {}},
                safe_vars,
            )
        except ZeroDivisionError as exc:
            raise CalculationSandboxError("Division by zero in calculation.") from exc
        except Exception as exc:
            raise CalculationSandboxError(f"Failed to evaluate expression: {exc}") from exc

        if not isinstance(value, (int, float)):
            raise CalculationSandboxError("Expression did not produce a numeric result.")
        return float(value)
