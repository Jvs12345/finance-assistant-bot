"""
Local financial formula registry loader and matcher.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FormulaDefinition:
    id: str
    name: str
    category: str
    expression: str
    unit: str
    variables: List[str]
    variable_labels: Dict[str, List[str]]
    aliases: List[str]
    derived_inputs: Dict[str, Dict[str, object]]
    output_type: str
    explanation: str
    variable_rules: Dict[str, Dict[str, Any]]


class FormulaRegistry:
    def __init__(self, path: Optional[Path] = None):
        base_path = Path(__file__).resolve().parent.parent
        self.path = path or (base_path / "data" / "financial_formulas.json")
        self._formulas: List[FormulaDefinition] = []
        self._variable_concepts: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            logger.warning(f"Formula registry file not found: {self.path}")
            self._formulas = []
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        formulas = []
        self._variable_concepts = payload.get("variable_concepts", {}) or {}
        for item in payload.get("formulas", []):
            formulas.append(
                FormulaDefinition(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    category=item.get("category", "other"),
                    expression=item.get("expression", ""),
                    unit=item.get("unit", ""),
                    variables=item.get("variables", []),
                    variable_labels=item.get("variable_labels", {}),
                    aliases=item.get("aliases", []),
                    derived_inputs=item.get("derived_inputs", {}),
                    output_type=item.get("output_type", item.get("unit", "amount")),
                    explanation=item.get("explanation", ""),
                    variable_rules=item.get("variable_rules", {}),
                )
            )
        self._formulas = formulas

    def list_formulas(self) -> List[FormulaDefinition]:
        return self._formulas

    def get_variable_rule(
        self,
        variable: str,
        formula: Optional[FormulaDefinition] = None,
    ) -> Dict[str, Any]:
        """Resolve variable extraction rule with formula-level override."""
        rule: Dict[str, Any] = {}
        lookup_keys = [variable]
        if variable.endswith("_current") or variable.endswith("_previous"):
            lookup_keys.append(variable.rsplit("_", 1)[0])
        for key in lookup_keys:
            if key in self._variable_concepts:
                rule.update(self._variable_concepts[key] or {})
                break
        if formula and variable in (formula.variable_rules or {}):
            merged = dict(rule)
            merged.update(formula.variable_rules.get(variable) or {})
            return merged
        return rule

    def find_by_question(self, question: str, max_results: int = 6) -> List[FormulaDefinition]:
        q = (question or "").lower()
        if not q:
            return []

        scored = []
        for formula in self._formulas:
            score = 0
            for alias in formula.aliases:
                a = alias.lower()
                if a in q:
                    score += len(a.split()) + 3
            for token in re.findall(r"[a-zA-Z]{3,}", formula.name.lower()):
                if token in q:
                    score += 1
            if score > 0:
                scored.append((score, formula))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:max_results]]

    def render_prompt_hints(self, question: str, max_results: int = 5) -> str:
        matches = self.find_by_question(question, max_results=max_results)
        if not matches:
            return ""

        lines = ["Formula library hints (local deterministic definitions):"]
        for f in matches:
            lines.append(
                f"- {f.id}: {f.name} = {f.expression} | variables: {', '.join(f.variables)}"
            )
        return "\n".join(lines)


_registry: Optional[FormulaRegistry] = None


def get_formula_registry() -> FormulaRegistry:
    global _registry
    if _registry is None:
        _registry = FormulaRegistry()
    return _registry
