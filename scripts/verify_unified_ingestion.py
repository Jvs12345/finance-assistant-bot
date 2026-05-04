#!/usr/bin/env python3
"""
Verification script for the unified PDF upload -> index -> query -> delete flow.

Usage:
  python scripts/verify_unified_ingestion.py --base-url http://localhost:8100 --api-key dummy-api-key --pdf Source_files\\sample.pdf --query "What is this document about?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def import_check() -> None:
    from src.db.elasticsearch_client import get_elasticsearch_client  # noqa: F401


def run(base_url: str, api_key: str, pdf_path: Path, query: str, do_delete: bool) -> int:
    import_check()
    print("[OK] Import check: src.db.elasticsearch_client")

    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return 1

    auth_header = {"Authorization": f"Bearer {api_key}"}

    with open(pdf_path, "rb") as fh:
        upload_resp = requests.post(
            f"{base_url}/api/v1/documents/upload",
            headers=auth_header,
            files={"file": (pdf_path.name, fh, "application/pdf")},
            data={
                "category": "tax_law",
                "jurisdiction": "Netherlands",
                "tax_year": "2025",
                "entity_type": "bv",
                "client_name": "Verification Client",
                "source_name": pdf_path.name,
            },
            timeout=300,
        )

    print(f"[INFO] Upload status: {upload_resp.status_code}")
    if upload_resp.status_code >= 400:
        print(upload_resp.text)
        return 2

    upload_data = upload_resp.json()
    print(json.dumps(upload_data, indent=2))
    document_id = upload_data.get("document_id")
    if not document_id:
        print("[ERROR] No document_id returned from upload")
        return 3

    ask_resp = requests.post(
        f"{base_url}/api/v1/llama/ask",
        headers={"Content-Type": "application/json"},
        json={
            "question": query,
            "model": "llama3.2",
            "jurisdiction": "Netherlands",
            "tax_year": 2025,
            "entity_type": "bv",
            "client_name": "Verification Client",
        },
        timeout=180,
    )

    print(f"[INFO] Query status: {ask_resp.status_code}")
    if ask_resp.status_code >= 400:
        print(ask_resp.text)
        return 4

    ask_data = ask_resp.json()
    print(json.dumps({"answer_preview": ask_data.get("answer", "")[:500], "sources": ask_data.get("sources", [])}, indent=2))

    if do_delete:
        delete_resp = requests.delete(
            f"{base_url}/api/v1/documents/{document_id}",
            headers=auth_header,
            timeout=120,
        )
        print(f"[INFO] Delete status: {delete_resp.status_code}")
        if delete_resp.status_code not in (200, 202, 204):
            print(delete_resp.text)
            return 5

    print("[OK] Unified ingestion verification completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify unified ingestion/query flow")
    parser.add_argument("--base-url", default="http://localhost:8100", help="API base URL")
    parser.add_argument("--api-key", required=True, help="Bearer API key")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to test PDF")
    parser.add_argument("--query", default="Summarize this uploaded document.", help="Question to ask after upload")
    parser.add_argument("--no-delete", action="store_true", help="Keep uploaded document after verification")
    args = parser.parse_args()

    return run(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        pdf_path=args.pdf,
        query=args.query,
        do_delete=not args.no_delete,
    )


if __name__ == "__main__":
    raise SystemExit(main())
