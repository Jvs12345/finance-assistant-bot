#!/usr/bin/env python3
"""
Verify XAF extraction and retrieval behavior.

Checks:
1. XAF files are present.
2. Elasticsearch returns XAF hits for XAF-specific queries.
3. Llama retrieval strategy returns XAF-only sources for XAF-focused questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from src.db.elasticsearch_client import get_elasticsearch_client
from src.services.llama_service import get_llama_service


def find_xaf_files() -> List[Path]:
    source_dir = Path("Source_files")
    if not source_dir.exists():
        return []
    return sorted([p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xaf"])


def check_elasticsearch_queries() -> Tuple[bool, List[str]]:
    client = get_elasticsearch_client()
    checks = [
        "btw nummer",
        "vat registration number",
        "fiscal year",
        "company name",
    ]
    failures: List[str] = []
    for query in checks:
        hits = client.search(query=query, limit=5, enable_fuzzy=True, file_type="xaf", use_vector=False)
        if not hits:
            failures.append(f"No XAF hits for query: {query}")
            continue
        if any((h.get("file_type") or "").lower() != "xaf" for h in hits):
            failures.append(f"Non-XAF hit returned for query: {query}")
    return len(failures) == 0, failures


def check_llama_xaf_focus() -> Tuple[bool, List[str]]:
    service = get_llama_service()
    questions = [
        "Wat is het btw nummer in de xaf auditfile?",
        "Noem de company name uit de XAF.",
        "Wat is het fiscal year volgens de auditfile?",
    ]
    failures: List[str] = []

    for question in questions:
        rows = service._retrieve_with_intent_strategy(
            question=question,
            detected_intent="normal_qna",
            retrieval_limit=6,
            system_context=None,
            jurisdiction="Netherlands",
            tax_year=None,
            entity_type=None,
            client_name=None,
            document_type=None,
            corpus_type=None,
        )
        if not rows:
            failures.append(f"No retrieval results for question: {question}")
            continue
        non_xaf = [r for r in rows if (r.get("file_type") or "").lower() != "xaf"]
        if non_xaf:
            failures.append(
                f"Non-XAF results returned for XAF-focused question: {question}"
            )

    return len(failures) == 0, failures


def main() -> int:
    xaf_files = find_xaf_files()
    print(f"XAF files found: {len(xaf_files)}")
    for path in xaf_files:
        print(f" - {path.name}")

    if not xaf_files:
        print("FAIL: No XAF files found in Source_files.")
        return 1

    ok_es, es_failures = check_elasticsearch_queries()
    ok_llama, llama_failures = check_llama_xaf_focus()

    if ok_es:
        print("PASS: Elasticsearch XAF queries returned valid XAF hits.")
    else:
        print("FAIL: Elasticsearch XAF checks failed:")
        for item in es_failures:
            print(f" - {item}")

    if ok_llama:
        print("PASS: Llama retrieval strategy is XAF-focused for XAF questions.")
    else:
        print("FAIL: Llama XAF-focus checks failed:")
        for item in llama_failures:
            print(f" - {item}")

    all_ok = ok_es and ok_llama
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

