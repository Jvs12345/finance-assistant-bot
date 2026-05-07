from typing import Dict, List
import re


INTENT_PRIORITY: List[str] = [
    "missing_info_check",
    "inconsistency_check",
    "advisory_points",
    "insurance_risk_check",
    "client_file_summary",
]

INTENT_TRIGGERS: Dict[str, List[str]] = {
    "missing_info_check": [
        "welke informatie ontbreekt",
        "wat ontbreekt",
        "wat mist er",
        "voordat de jaarrekening kan worden opgesteld",
        "jaarrekening kan worden opgesteld",
        "jaarrekening opgesteld",
        "jaarrekening opstellen",
        "missing information",
        "before preparing the annual accounts",
    ],
    "inconsistency_check": [
        "inconsistenties",
        "verschillen",
        "komt dit overeen",
        "controleer of",
        "check consistency",
        "winst-en-verliesrekening",
        "btw-overzicht",
        "klantnotities",
        "between p&l and vat",
    ],
    "advisory_points": [
        "adviespunten",
        "bespreken met klant",
        "mkb-klant",
        "mkb klant",
        "discussion points",
        "client advice",
        "advise this client",
    ],
    "insurance_risk_check": [
        "verzekeringsrisico",
        "verzekering",
        "risico's",
        "risicos",
        "bedrijfsactiviteiten",
        "activa",
        "contractinformatie",
        "liability",
        "underinsurance",
    ],
    "client_file_summary": [
        "vat klantdossier samen",
        "samenvatting klantdossier",
        "belangrijkste punten",
        "summarise client file",
        "client overview",
    ],
}


def _normalize_text(text: str) -> str:
    value = (text or "").lower()
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"[^\w\s&\-/']", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_advisor_intent(question: str, is_calculation: bool = False) -> str:
    if is_calculation:
        return "financial_calculation"

    normalized_question = _normalize_text(question)
    if not normalized_question:
        return "normal_qna"

    best_intent = "normal_qna"
    best_score = 0

    for intent in INTENT_PRIORITY:
        score = 0
        for trigger in INTENT_TRIGGERS.get(intent, []):
            if _normalize_text(trigger) in normalized_question:
                score += max(1, len(trigger.split()))
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent if best_score > 0 else "normal_qna"
