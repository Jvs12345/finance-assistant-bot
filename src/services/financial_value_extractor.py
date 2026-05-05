"""Label-aware financial value extraction for formula inputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from src.services.formula_registry import FormulaRegistry, FormulaDefinition


@dataclass
class ExtractedValue:
    variable: str
    display_label: str
    raw_value: float
    normalized_value: float
    value_text: str
    unit_label: str
    year: Optional[int]
    source: Dict[str, Any]
    source_label: str
    line_text: str
    match_score: float
    derived: bool = False


class FinancialValueExtractor:
    """Resolve variable values from retrieved chunks with strict label/section rules."""

    _SECTION_HINTS = {
        "income_statement": [
            "consolidated statements of operations",
            "consolidated statements of income",
            "income statement",
            "statement of operations",
        ],
        "balance_sheet": [
            "consolidated balance sheets",
            "balance sheet",
            "financial position",
        ],
        "cash_flow": [
            "consolidated statements of cash flows",
            "cash flow statement",
            "cash provided by operating activities",
        ],
        "notes": [
            "notes to consolidated financial statements",
            "note ",
        ],
    }

    _SEGMENT_HINTS = [
        "segment",
        "google services",
        "google cloud",
        "other bets",
        "region",
        "product revenue",
    ]

    def __init__(self, registry: FormulaRegistry):
        self.registry = registry

    def resolve_best_value(
        self,
        variable: str,
        formula: FormulaDefinition,
        rows: List[Dict[str, Any]],
        desired_year: Optional[int],
        company_target: Optional[str] = None,
    ) -> Optional[ExtractedValue]:
        rule = self.registry.get_variable_rule(variable, formula)
        accepted = self._accepted_labels(variable, formula, rule)
        rejected = [self._normalize_text(x) for x in rule.get("rejected_labels", []) if x]
        preferred_sections = [self._normalize_text(x) for x in rule.get("preferred_sections", []) if x]
        rejected_sections = [self._normalize_text(x) for x in rule.get("rejected_sections", []) if x]
        requires_exact = bool(rule.get("requires_exact_label", False))
        allow_contains = bool(rule.get("contains_match", not requires_exact))
        allow_segment = bool(rule.get("allow_segment_values", False))
        statement_type = str(rule.get("statement_type") or "").strip().lower() or None

        best: Optional[ExtractedValue] = None
        for row in rows:
            if company_target and not self._row_mentions_entity(row, company_target):
                continue

            section_blob = self._normalize_text(
                " ".join(
                    [
                        str(row.get("title", "")),
                        str(row.get("filename", "")),
                        str(row.get("content", ""))[:1200],
                    ]
                )
            )
            section_type = self._detect_section_type(section_blob)

            lines = self._candidate_lines(row)
            for idx, line in enumerate(lines):
                lower_line = line.lower()
                matched_label = self._match_label(lower_line, accepted, requires_exact, allow_contains)
                if not matched_label:
                    continue

                label_start = lower_line.find(matched_label)
                label_segment = line[label_start:] if label_start >= 0 else line
                label_text = re.split(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?", label_segment, maxsplit=1)[0].strip()
                label_norm = self._normalize_text(label_text or matched_label)
                matched_norm = self._normalize_text(matched_label)

                if self._is_rejected_label(label_norm, rejected, matched_norm):
                    continue
                if any(self._normalize_text(seg) in label_norm for seg in self._SEGMENT_HINTS) and not allow_segment:
                    continue

                window = " ".join(lines[max(0, idx - 2): min(len(lines), idx + 3)])
                window_norm = self._normalize_text(window)
                window_section_type = self._detect_section_type(window_norm) or section_type
                if statement_type and window_section_type and window_section_type != statement_type:
                    continue
                if any(rs in window_norm for rs in rejected_sections):
                    continue
                tokens = self._extract_numeric_tokens(label_segment)
                if not tokens:
                    continue

                value_token, token_year = self._select_token_for_year(tokens, window, desired_year)
                if value_token is None:
                    continue
                value = self._parse_numeric_token(value_token)
                if value is None:
                    continue

                unit_multiplier, unit_label = self._detect_unit_multiplier(window)
                normalized = value * unit_multiplier
                year = token_year or self._extract_year_from_window(window) or row.get("tax_year")
                if year is not None:
                    try:
                        year = int(year)
                    except Exception:
                        year = None

                score = float(row.get("score", 0.0) or 0.0)
                score += 4.0 if self._normalize_text(matched_label) == label_norm else 2.0
                if desired_year and year:
                    score += 3.0 if int(year) == int(desired_year) else -2.0
                if preferred_sections and any(ps in window_norm for ps in preferred_sections):
                    score += 2.0
                if statement_type and window_section_type == statement_type:
                    score += 2.0
                if window_section_type == "notes":
                    score -= 0.3

                candidate = ExtractedValue(
                    variable=variable,
                    display_label=label_text or matched_label,
                    raw_value=value,
                    normalized_value=normalized,
                    value_text=value_token,
                    unit_label=unit_label,
                    year=year,
                    source=row,
                    source_label=f"{row.get('filename', 'Unknown')}, page {row.get('page_number') or 'unknown'}",
                    line_text=line,
                    match_score=score,
                )
                if best is None or candidate.match_score > best.match_score:
                    best = candidate

        return best

    def _is_rejected_label(
        self,
        label_norm: str,
        rejected_labels: List[str],
        matched_norm: str,
    ) -> bool:
        for rejected in rejected_labels:
            if rejected not in label_norm:
                continue
            # Keep specific accepted phrase matches such as "cost of revenue"
            # when rejected token is generic (for example "revenue").
            if matched_norm and len(matched_norm.split()) > 1 and rejected in matched_norm:
                continue
            return True
        return False

    def _accepted_labels(
        self,
        variable: str,
        formula: FormulaDefinition,
        rule: Dict[str, Any],
    ) -> List[str]:
        labels = []
        explicit_rule_labels = list(rule.get("accepted_labels", []) or [])
        labels.extend(explicit_rule_labels)
        if not explicit_rule_labels:
            labels.extend(formula.variable_labels.get(variable, []) or [])
        labels.append(variable.replace("_", " "))
        if variable.endswith("_current") or variable.endswith("_previous"):
            root = variable.rsplit("_", 1)[0]
            labels.extend(rule.get("aliases", []) or [])
            if not explicit_rule_labels:
                labels.extend(formula.variable_labels.get(root, []) or [])
            labels.append(root.replace("_", " "))
        dedup: List[str] = []
        seen = set()
        for label in labels:
            l = (label or "").strip().lower()
            if not l or l in seen:
                continue
            seen.add(l)
            dedup.append(l)
        return dedup

    def _candidate_lines(self, row: Dict[str, Any]) -> List[str]:
        text = str(row.get("content") or row.get("summary") or row.get("snippet") or "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            return lines
        parts = re.split(r"[.;]\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _match_label(
        self,
        lower_line: str,
        accepted_labels: List[str],
        requires_exact: bool,
        allow_contains: bool,
    ) -> Optional[str]:
        line_norm = self._normalize_text(lower_line)
        for label in accepted_labels:
            norm_label = self._normalize_text(label)
            if not norm_label:
                continue
            if line_norm == norm_label:
                return label
            if requires_exact and (line_norm.startswith(norm_label + " ") or f" {norm_label} " in f" {line_norm} "):
                return label
        if requires_exact:
            return None
        if allow_contains:
            for label in accepted_labels:
                if label in lower_line:
                    return label
                norm_label = self._normalize_text(label)
                if norm_label and norm_label in line_norm:
                    return label
        return None

    def _extract_numeric_tokens(self, text: str) -> List[str]:
        tokens = re.findall(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?", text or "")
        out = []
        for token in tokens:
            parsed = self._parse_numeric_token(token)
            if parsed is None:
                continue
            # Drop obvious year tokens when there are several numeric tokens.
            if 1900 <= abs(parsed) <= 2100 and len(tokens) > 2:
                continue
            out.append(token)
        return out

    def _select_token_for_year(
        self,
        tokens: List[str],
        context: str,
        desired_year: Optional[int],
    ) -> Tuple[Optional[str], Optional[int]]:
        if not tokens:
            return None, None
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", context or "")]
        if not desired_year:
            return tokens[-1], (years[-1] if years else None)
        if years and desired_year in years:
            mapped = list(tokens)
            if len(mapped) > len(years):
                mapped = mapped[-len(years):]
            idx = years.index(desired_year)
            if idx < len(mapped):
                return mapped[idx], desired_year
        return tokens[-1], (years[-1] if years else desired_year)

    def _detect_section_type(self, text: str) -> Optional[str]:
        t = text or ""
        best = None
        best_hits = 0
        for section_name, patterns in self._SECTION_HINTS.items():
            hits = sum(1 for p in patterns if p in t)
            if hits > best_hits:
                best = section_name
                best_hits = hits
        return best

    def _detect_unit_multiplier(self, text: str) -> Tuple[float, str]:
        t = (text or "").lower()
        if "in billions" in t or "in billion" in t:
            return 1_000_000_000.0, "billions"
        if "in millions" in t or "in million" in t:
            return 1_000_000.0, "millions"
        if "in thousands" in t or "in thousand" in t:
            return 1_000.0, "thousands"
        return 1.0, "units"

    def _extract_year_from_window(self, text: str) -> Optional[int]:
        years = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", text or "")
        if not years:
            return None
        return max(int(y) for y in years)

    def _parse_numeric_token(self, token: str) -> Optional[float]:
        if not token:
            return None
        t = token.strip().replace("$", "").replace(",", "")
        negative = False
        if t.startswith("(") and t.endswith(")"):
            negative = True
            t = t[1:-1]
        try:
            value = float(t)
            return -value if negative else value
        except ValueError:
            return None

    def _row_mentions_entity(self, row: Dict[str, Any], entity: str) -> bool:
        searchable = " ".join(
            [
                str(row.get("filename", "")),
                str(row.get("title", "")),
                str(row.get("source_name", "")),
                str(row.get("content", ""))[:1600],
            ]
        ).lower()
        return entity.lower() in searchable

    def _normalize_text(self, text: str) -> str:
        t = (text or "").lower()
        t = t.replace("’", "'").replace("`", "'")
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
