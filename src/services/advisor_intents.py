from typing import Dict, List
import re


INTENT_PRIORITY: List[str] = [
    "client_file_summary",
    "document_summary",
    "ce_compliance_gap_check",
    "document_governance_check",
    "technical_requirements_check",
    "risk_attention_check",
    "quotation_preparation",
    "ai_use_case_identification",
    "use_case_prioritization",
    "pilot_project_translation",
    "local_privacy_explanation",
    "missing_info_check",
    "inconsistency_check",
    "advisory_points",
    "insurance_risk_check",
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
    "document_summary": [
        "vat bestand samen",
        "vat het bestand samen",
        "vat document samen",
        "vat dit document samen",
        "geef een samenvatting",
        "maak een korte samenvatting",
        "waar gaat dit document over",
        "wat staat er in dit document",
        "summarise this document",
        "summarize this document",
        "give a document summary",
        "what is this document about",
    ],
    "technical_requirements_check": [
        "technische eisen",
        "technical requirements",
        "welke eisen",
        "welke verplichtingen",
        "onderdelen of systemen",
        "components or systems",
        "aannemer of leverancier",
        "contractor responsibilities",
        "stappen voordat werkzaamheden",
        "before work starts",
    ],
    "risk_attention_check": [
        "risico",
        "risico's",
        "risicos",
        "aandachtspunten",
        "hazards",
        "risk points",
        "safety concerns",
    ],
    "ce_compliance_gap_check": [
        "ce-documentatie",
        "ce documentatie",
        "ce compliance",
        "conformiteitseisen",
        "conformity requirements",
        "technisch dossier",
        "verklaring van overeenstemming",
        "declaration of conformity",
        "welke informatie ontbreekt nog voordat ce",
    ],
    "document_governance_check": [
        "metadata",
        "revisie",
        "revision",
        "versie",
        "version",
        "status",
        "eigenaar",
        "owner",
        "audit-proof",
        "audit proof",
        "tegenstrijdige instructies",
        "dubbele instructies",
        "normen zonder versie",
        "ocr-ruis",
        "ocr ruis",
        "vertrouwelijk",
        "confidential",
        "wijzigingsvoorstellen",
        "prioriteit voor de volgende revisie",
        "documentbeheer",
        "document governance",
        "versiebeheer",
        "wijzigingslog",
        "audit trail",
        "goedkeuringsstatus",
        "review cycle",
        "beheerder",
        "document owner",
    ],
    "quotation_preparation": [
        "offerte",
        "quotation",
        "quote",
        "eerste offerte",
        "offerte op te stellen",
        "offerte betrouwbaar",
        "conceptstructuur voor een offerte",
        "prijsindicatie",
        "scope en randvoorwaarden",
    ],
    "ai_use_case_identification": [
        "ai-use-cases",
        "ai use cases",
        "welke use-cases",
        "repeterende taken",
        "documentintensief",
        "waar kan ai tijd besparen",
        "interne kennisassistent",
    ],
    "use_case_prioritization": [
        "scoor deze use-cases",
        "impact haalbaarheid databeschikbaarheid",
        "prioriteer use-cases",
        "meeste waarde voor het bedrijf",
        "snel te testen met beperkte data",
    ],
    "pilot_project_translation": [
        "werk de beste use-case uit als pilotproject",
        "pilotproject voor studenten",
        "wat is het probleem",
        "welke data is nodig",
        "wie is betrokken",
        "wat kan een prototype opleveren",
        "succes van de pilot",
    ],
    "local_privacy_explanation": [
        "lokale ai-assistent",
        "waarom lokaal relevant",
        "welke data blijft lokaal",
        "verschil met chatgpt",
        "verschil met claude",
        "data security",
        "privacy",
    ],
}

INTENT_MIN_SCORE: Dict[str, int] = {
    "document_summary": 2,
    "ce_compliance_gap_check": 2,
    "document_governance_check": 2,
    "technical_requirements_check": 2,
    "risk_attention_check": 2,
    "quotation_preparation": 2,
    "ai_use_case_identification": 2,
    "use_case_prioritization": 2,
    "pilot_project_translation": 2,
    "local_privacy_explanation": 2,
    "missing_info_check": 2,
    "inconsistency_check": 2,
    "advisory_points": 2,
    "insurance_risk_check": 2,
    "client_file_summary": 2,
}


def _normalize_text(text: str) -> str:
    value = (text or "").lower()
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"[^\w\s&\-/']", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_document_summary_request(normalized_question: str) -> bool:
    if not normalized_question:
        return False

    summary_terms = [
        "samenvatting",
        "samenvat",
        "samenvatten",
        "samenvat",
        "summary",
        "summarise",
        "summarize",
        "overzicht",
        "overview",
    ]
    doc_terms = [
        "document",
        "bestand",
        "file",
        "pdf",
        "dossier",
    ]

    asks_summary = any(t in normalized_question for t in summary_terms)
    mentions_doc = any(t in normalized_question for t in doc_terms)
    has_about_pattern = bool(
        re.search(r"\b(waar gaat|what is)\b.*\b(document|bestand|file|pdf)\b.*\b(over|about)\b", normalized_question)
        or re.search(r"\bwat staat er in\b.*\b(document|bestand|file|pdf)\b", normalized_question)
    )
    has_short_summary_pattern = bool(
        re.search(r"\b(geef|maak|give|make)\b.*\b(korte|short)?\s*(samenvatting|summary)\b", normalized_question)
    )
    has_verb_document_pattern = bool(
        re.search(r"\bvat\b.*\b(document|bestand|file|pdf)\b.*\bsamen\b", normalized_question)
        or re.search(r"\b(summarise|summarize)\b.*\b(document|file|pdf)\b", normalized_question)
    )

    return (
        has_about_pattern
        or has_verb_document_pattern
        or (asks_summary and (mentions_doc or has_short_summary_pattern))
    )


def _is_ce_compliance_request(normalized_question: str) -> bool:
    ce_terms = [
        "ce", "ce-documentatie", "ce documentatie", "ce compliance", "conformiteit",
        "conformiteitseis", "technisch dossier", "declaration of conformity",
        "verklaring van overeenstemming",
    ]
    gap_terms = ["ontbreekt", "miss", "missing", "compleet", "complete", "onderbouwd"]
    has_ce = any(term in normalized_question for term in ce_terms)
    has_gap = any(term in normalized_question for term in gap_terms)
    return has_ce and (has_gap or "welke informatie" in normalized_question)


def _is_technical_requirements_request(normalized_question: str) -> bool:
    tech_terms = [
        "technische eisen", "technical requirements", "onderdelen", "systemen", "components",
        "verplichtingen", "aannemer", "leverancier", "werkzaamheden mogen starten",
        "before work starts",
    ]
    return any(term in normalized_question for term in tech_terms)


def _is_risk_attention_request(normalized_question: str) -> bool:
    return any(
        term in normalized_question
        for term in ["risico", "risico's", "risicos", "aandachtspunten", "hazard", "safety concern"]
    )


def _is_insurance_risk_request(normalized_question: str) -> bool:
    insurance_terms = [
        "verzekeringsrisico",
        "verzekeringsrisico s",
        "verzekering",
        "polis",
        "dekking",
        "underinsurance",
        "liability coverage",
        "insurance risk",
    ]
    return any(term in normalized_question for term in insurance_terms)


def _is_quotation_preparation_request(normalized_question: str) -> bool:
    quote_terms = [
        "offerte", "quotation", "quote", "prijs", "doorlooptijd", "scope", "randvoorwaarden",
    ]
    return any(term in normalized_question for term in quote_terms)


def _is_use_case_identification_request(normalized_question: str) -> bool:
    use_case_terms = [
        "use-case", "use case", "ai-use-cases", "ai use cases", "repeterende taken",
        "documentintensief", "waar kan ai tijd besparen",
    ]
    return any(term in normalized_question for term in use_case_terms)


def _is_use_case_prioritization_request(normalized_question: str) -> bool:
    return (
        "impact" in normalized_question
        and ("haalbaarheid" in normalized_question or "feasibility" in normalized_question)
        and ("databeschikbaarheid" in normalized_question or "data availability" in normalized_question)
    ) or "scoor deze use-cases" in normalized_question


def _is_pilot_translation_request(normalized_question: str) -> bool:
    pilot_terms = ["pilotproject", "pilot project", "prototype", "student", "20 weken", "20 weeks"]
    return any(term in normalized_question for term in pilot_terms) and (
        "probleem" in normalized_question
        or "data is nodig" in normalized_question
        or "wie is betrokken" in normalized_question
        or "use-case uit" in normalized_question
    )


def _is_local_privacy_request(normalized_question: str) -> bool:
    privacy_terms = [
        "lokale ai", "local ai", "blijft lokaal", "data privacy", "data security",
        "chatgpt", "claude", "on premise", "on-premise",
    ]
    return any(term in normalized_question for term in privacy_terms)


def _is_document_governance_request(normalized_question: str) -> bool:
    if not normalized_question:
        return False

    strong_phrases = [
        "documentbeheer",
        "document governance",
        "versiebeheer",
        "wijzigingslog",
        "audit trail",
        "audit-proof",
        "audit proof",
        "goedkeuringsstatus",
        "tegenstrijdige instructies",
        "dubbele instructies",
        "normen zonder versie",
        "ocr-ruis",
        "ocr ruis",
        "document owner",
        "wie is eigenaar van dit document",
        "welke versie is geldig",
    ]
    if any(term in normalized_question for term in strong_phrases):
        return True

    weak_terms = [
        "metadata", "revisie", "revision", "versie", "version", "status", "eigenaar", "owner",
        "tegenstrijdig", "normen", "richtlijn", "vertrouwelijk", "confidential", "wijzigingsvoorstel",
        "beheerder", "review cycle",
    ]
    matches = sum(1 for term in weak_terms if term in normalized_question)
    return matches >= 2


def detect_advisor_intent(question: str, is_calculation: bool = False) -> str:
    if is_calculation:
        return "financial_calculation"

    normalized_question = _normalize_text(question)
    if not normalized_question:
        return "normal_qna"

    if _is_ce_compliance_request(normalized_question):
        return "ce_compliance_gap_check"

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

    min_score = INTENT_MIN_SCORE.get(best_intent, 1)
    if best_score >= min_score:
        return best_intent

    if _is_document_summary_request(normalized_question):
        return "document_summary"
    if _is_document_governance_request(normalized_question):
        return "document_governance_check"
    if _is_use_case_prioritization_request(normalized_question):
        return "use_case_prioritization"
    if _is_pilot_translation_request(normalized_question):
        return "pilot_project_translation"
    if _is_quotation_preparation_request(normalized_question):
        return "quotation_preparation"
    if _is_use_case_identification_request(normalized_question):
        return "ai_use_case_identification"
    if _is_technical_requirements_request(normalized_question):
        return "technical_requirements_check"
    if _is_insurance_risk_request(normalized_question):
        return "insurance_risk_check"
    if _is_risk_attention_request(normalized_question):
        return "risk_attention_check"
    if _is_local_privacy_request(normalized_question):
        return "local_privacy_explanation"
    return "normal_qna"
