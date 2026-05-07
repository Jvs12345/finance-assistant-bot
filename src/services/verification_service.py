"""Local answer verification for source links and arithmetic consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class VerificationResult:
    status: str
    issues: List[str]
    safe_answer: Optional[str] = None


class AnswerVerificationService:
    """Verify generated answers against retrieved sources using deterministic checks."""

    _DUTCH_HINTS = (
        "welke", "controleer", "jaarrekening", "btw", "bron", "omzet", "verzekeringsrisico",
    )

    def verify(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> VerificationResult:
        if not answer:
            return VerificationResult(status="pass", issues=[])

        issues: List[str] = []
        source_index = self._build_source_index(search_results, sources)

        source_issues = self._verify_source_references(answer, source_index)
        issues.extend(source_issues)

        calc_issues = self._verify_difference_lines(answer)
        issues.extend(calc_issues)

        if issues:
            return VerificationResult(
                status="fail",
                issues=issues,
                safe_answer=self._build_safe_answer(question, issues),
            )
        return VerificationResult(status="pass", issues=[])

    def _build_source_index(
        self,
        search_results: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], str]:
        idx: Dict[Tuple[str, str], str] = {}
        combined = list(search_results) + list(sources)
        for row in combined:
            filename = str(row.get("filename") or "").strip().lower()
            page = str(row.get("page_number") or row.get("page") or "").strip()
            if not filename or not page:
                continue
            text = " ".join(
                [
                    str(row.get("content", "")),
                    str(row.get("snippet", "")),
                    str(row.get("summary", "")),
                    str(row.get("title", "")),
                ]
            ).lower()
            key = (filename, page)
            existing = idx.get(key)
            if existing is None or len(text) > len(existing):
                idx[key] = text
        return idx

    def _verify_source_references(
        self,
        answer: str,
        source_index: Dict[Tuple[str, str], str],
    ) -> List[str]:
        issues: List[str] = []
        ref_pattern = re.compile(
            r"(?:bron|source)\s*:\s*([^,\n]+),\s*page\s*(\d+)",
            flags=re.IGNORECASE,
        )
        number_pattern = re.compile(r"(?<!\d)(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?!\d)")

        for line in answer.splitlines():
            match = ref_pattern.search(line)
            if not match:
                continue
            filename = match.group(1).strip().lower()
            page = match.group(2).strip()
            source_text = source_index.get((filename, page))
            if source_text is None:
                issues.append(f"Bronverwijzing niet gevonden in opgehaalde bronnen: {match.group(1).strip()}, page {page}.")
                continue

            claim_text = line[:match.start()]
            nums = number_pattern.findall(claim_text)
            if not nums:
                continue
            row_vals = self._extract_numeric_values(source_text)
            for token in nums:
                val = self._parse_number(token)
                if val is None:
                    continue
                # Ignore likely year mentions in evidence lines.
                if 1900 <= val <= 2100:
                    continue
                if not any(abs(val - rv) <= max(1.0, abs(val) * 0.001) for rv in row_vals):
                    issues.append(
                        f"Waarde {token} komt niet overeen met gevonden numerieke waarden in {match.group(1).strip()}, page {page}."
                    )
                    break
        return issues

    def _verify_difference_lines(self, answer: str) -> List[str]:
        issues: List[str] = []
        diff_pattern = re.compile(
            r"Verschil:\s*([0-9\.,-]+)\s*-\s*([0-9\.,-]+)\s*=\s*([0-9\.,-]+)",
            flags=re.IGNORECASE,
        )
        for line in answer.splitlines():
            m = diff_pattern.search(line)
            if not m:
                continue
            left = self._parse_number(m.group(1))
            right = self._parse_number(m.group(2))
            shown = self._parse_number(m.group(3))
            if left is None or right is None or shown is None:
                continue
            expected = left - right
            if abs(expected - shown) > max(1.0, abs(expected) * 0.001):
                issues.append(
                    f"Rekencontrole mislukt: {m.group(1)} - {m.group(2)} hoort {expected:,.2f} te zijn, niet {m.group(3)}."
                )
        return issues

    def _extract_numeric_values(self, text: str) -> List[float]:
        values: List[float] = []
        for token in re.findall(r"(?<!\d)(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?!\d)", text):
            val = self._parse_number(token)
            if val is not None:
                values.append(val)
        return values

    def _parse_number(self, token: str) -> Optional[float]:
        if not token:
            return None
        t = token.strip().replace(" ", "")
        # EU thousand format 372.000
        if "." in t and "," not in t and re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", t):
            t = t.replace(".", "")
        # US thousand format 714,000
        elif "," in t and "." not in t and re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
        try:
            return float(t)
        except ValueError:
            return None

    def _is_dutch(self, question: str) -> bool:
        q = (question or "").lower()
        return any(h in q for h in self._DUTCH_HINTS)

    def _build_safe_answer(self, question: str, issues: List[str]) -> str:
        if self._is_dutch(question):
            issue_lines = "\n".join(f"- {i}" for i in issues[:4])
            return (
                "Kort antwoord:\n"
                "Ik kan dit antwoord nog niet betrouwbaar bevestigen op basis van de opgehaalde bronnen.\n\n"
                "Problemen bij automatische controle:\n"
                f"{issue_lines}\n\n"
                "Volgende stap:\n"
                "Controleer bronkoppeling en berekeningen opnieuw, of verfijn de vraag op document/jaar/periode."
            )

        issue_lines = "\n".join(f"- {i}" for i in issues[:4])
        return (
            "Direct answer:\n"
            "I cannot reliably confirm this answer from the retrieved sources yet.\n\n"
            "Verification issues:\n"
            f"{issue_lines}\n\n"
            "Suggested next step:\n"
            "Re-check source mapping and calculations, or narrow the question by document/year/period."
        )
