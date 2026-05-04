#!/usr/bin/env python3
"""
Verification script for semantic search setup.

This script checks that all components are configured correctly:
- Environment variables
- Elasticsearch connection
- Embedding service
- Semantic index
- Document ingestion pipeline
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)


def check_env_vars():
    """Check required environment variables."""
    print("\n" + "="*60)
    print("1. Checking Environment Variables")
    print("="*60)

    required = {
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        "EMBEDDING_API_KEY": os.getenv("EMBEDDING_API_KEY"),
        "EMBEDDING_DIM": os.getenv("EMBEDDING_DIM"),
        "SEARCH_MODE": os.getenv("SEARCH_MODE"),
        "ELASTICSEARCH_URL": os.getenv("ELASTICSEARCH_URL"),
    }

    all_set = True
    for key, value in required.items():
        if not value:
            print(f"  ✗ {key}: NOT SET")
            all_set = False
        elif value.startswith("your-") or value == "sk-your-openai-api-key-here":
            print(f"  ⚠️  {key}: Placeholder value detected")
            all_set = False
        else:
            # Mask API keys
            display_value = value
            if "KEY" in key and len(value) > 10:
                display_value = value[:8] + "..." + value[-4:]
            print(f"  ✓ {key}: {display_value}")

    if all_set:
        print("\n✓ All environment variables configured")
    else:
        print("\n✗ Some environment variables missing or have placeholder values")
        print("  Update .env file with your actual API keys")

    return all_set


def check_elasticsearch():
    """Check Elasticsearch connection."""
    print("\n" + "="*60)
    print("2. Checking Elasticsearch Connection")
    print("="*60)

    try:
        from elasticsearch import Elasticsearch
        from src.config import settings

        es = Elasticsearch([settings.elasticsearch_url], request_timeout=5)

        if es.ping():
            info = es.info()
            print(f"  ✓ Connected to Elasticsearch")
            print(f"    Version: {info['version']['number']}")
            print(f"    Cluster: {info['cluster_name']}")
            return True
        else:
            print(f"  ✗ Cannot ping Elasticsearch at {settings.elasticsearch_url}")
            return False

    except Exception as e:
        print(f"  ✗ Elasticsearch connection failed: {e}")
        print(f"    Make sure Elasticsearch is running")
        return False


def check_embedding_service():
    """Check embedding service."""
    print("\n" + "="*60)
    print("3. Checking Embedding Service")
    print("="*60)

    try:
        from src.services.embedding_service import get_embedding_service

        service = get_embedding_service()

        print(f"  Provider: {service.provider}")
        print(f"  Model: {service.model}")
        print(f"  Dimension: {service.dimension}")

        # Test embedding generation
        print("\n  Testing embedding generation...")
        test_text = "This is a test document about semantic search."
        embedding = service.get_embedding(test_text)

        if len(embedding) == service.dimension:
            print(f"  ✓ Generated {len(embedding)}-dimensional embedding")
            print(f"    Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
            return True
        else:
            print(f"  ✗ Embedding dimension mismatch: got {len(embedding)}, expected {service.dimension}")
            return False

    except Exception as e:
        print(f"  ✗ Embedding service failed: {e}")
        print(f"    Check your EMBEDDING_API_KEY and provider settings")
        return False


def check_semantic_index():
    """Check semantic search index."""
    print("\n" + "="*60)
    print("4. Checking Semantic Search Index")
    print("="*60)

    try:
        from src.db.semantic_index import get_semantic_search_client

        client = get_semantic_search_client()

        print(f"  ✓ Semantic index client initialized")
        print(f"    Index name: {client.index_name}")
        print(f"    Search mode: {client.search_mode}")

        # Check document count
        count = client.get_document_count()
        print(f"    Indexed documents: {count}")

        if count == 0:
            print(f"\n  ⚠️  No documents in index")
            print(f"     Run: python scripts/ingest_documents.py --directory data/Source_files")
        else:
            print(f"  ✓ Index contains {count} documents")

        return True

    except Exception as e:
        print(f"  ✗ Semantic index check failed: {e}")
        return False


def check_ingestion_pipeline():
    """Check document ingestion pipeline."""
    print("\n" + "="*60)
    print("5. Checking Document Ingestion Pipeline")
    print("="*60)

    try:
        from src.services.document_ingestion import get_ingestion_service

        service = get_ingestion_service()
        print(f"  ✓ Ingestion service initialized")

        # Check summary generation capability
        print(f"\n  Testing summary generation...")
        test_text = "This is a test document. " * 50
        summary_short, summary_long = service.generate_summaries(test_text, "Test Document")

        print(f"  ✓ Generated summaries:")
        print(f"    Short ({len(summary_short)} chars): {summary_short[:80]}...")
        print(f"    Long ({len(summary_long)} chars): {summary_long[:80]}...")

        return True

    except Exception as e:
        print(f"  ✗ Ingestion pipeline check failed: {e}")
        return False


def check_ai_service():
    """Check AI service integration."""
    print("\n" + "="*60)
    print("6. Checking AI Service Integration")
    print("="*60)

    try:
        from src.services.ai_service import get_ai_service

        service = get_ai_service()
        print(f"  ✓ AI service initialized")
        print(f"    Model: {service.model}")
        print(f"    Service version: {service._service_version}")

        # Check search client
        print(f"    Search client: {type(service.search_client).__name__}")

        if "Semantic" in type(service.search_client).__name__:
            print(f"  ✓ Using semantic search client")
            return True
        else:
            print(f"  ⚠️  Not using semantic search client")
            return False

    except Exception as e:
        print(f"  ✗ AI service check failed: {e}")
        return False


def run_end_to_end_test():
    """Run a basic end-to-end test."""
    print("\n" + "="*60)
    print("7. Running End-to-End Test")
    print("="*60)

    try:
        from src.db.semantic_index import get_semantic_search_client

        client = get_semantic_search_client()

        # Check if we have documents
        count = client.get_document_count()
        if count == 0:
            print("  ⚠️  Skipping (no documents ingested)")
            return True

        # Try a simple search
        print("\n  Testing semantic search...")
        results = client.search(
            query="installation instructions",
            limit=3
        )

        if results:
            print(f"  ✓ Search returned {len(results)} results:")
            for i, result in enumerate(results[:3], 1):
                print(f"    {i}. {result['title']} (score: {result['score']:.3f})")
            return True
        else:
            print(f"  ⚠️  Search returned no results")
            return True

    except Exception as e:
        print(f"  ✗ End-to-end test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("SEMANTIC SEARCH VERIFICATION")
    print("="*60)
    print("\nThis script verifies your semantic search setup.")

    results = {
        "Environment Variables": check_env_vars(),
        "Elasticsearch": check_elasticsearch(),
        "Embedding Service": check_embedding_service(),
        "Semantic Index": check_semantic_index(),
        "Ingestion Pipeline": check_ingestion_pipeline(),
        "AI Service": check_ai_service(),
        "End-to-End Test": run_end_to_end_test(),
    }

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for check, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {check}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n✓ All checks passed! Semantic search is ready to use.")
        print("\nNext steps:")
        print("  1. Ingest documents: python scripts/ingest_documents.py --directory data")
        print("  2. Try examples: python examples/semantic_search_example.py")
        print("  3. Start app: uvicorn src.main:app --reload")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Set EMBEDDING_API_KEY in .env")
        print("  - Start Elasticsearch: docker-compose up -d")
        print("  - Check .env configuration")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVerification interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        sys.exit(1)
