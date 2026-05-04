#!/usr/bin/env python3
"""
Test script to validate the reindexing process on a small sample.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.confluence_parser import parse_confluence_export
from src.db.elasticsearch_client import get_elasticsearch_client
from src.utils.logging import setup_logging, get_logger
from src.config import settings

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def test_sample_processing():
    """Test processing on a small sample of documents."""
    print("=" * 70)
    print("  Testing Confluence Parser on Sample Documents")
    print("=" * 70)
    print()

    export_dir = Path("Source_files")

    if not export_dir.exists():
        print(f"ERROR: Source_files directory not found")
        return 1

    print(f"Export directory: {export_dir.absolute()}")
    print()

    # Parse only first 5 pages for testing
    try:
        from src.services.confluence_parser import parse_entities_xml, process_all_pages

        print("Parsing entities.xml...")
        pages_data = parse_entities_xml(export_dir)

        # Take only first 5 pages
        sample_pages = list(pages_data.items())[:5]
        print(f"Processing {len(sample_pages)} sample pages...")
        print()

        documents = []
        for page_id, page_info in sample_pages:
            print(f"Processing: {page_info.get('title', 'Unknown')}")
            page_docs = process_all_pages(export_dir, {page_id: page_info})
            documents.extend(page_docs)
            print(f"  Generated {len(page_docs)} documents")

        print()
        print(f"Successfully processed {len(documents)} documents from {len(sample_pages)} pages")

        # Show sample document
        if documents:
            print()
            print("Sample document:")
            doc = documents[0]
            print(f"  ID: {doc.get('document_id')}")
            print(f"  Title: {doc.get('title')}")
            print(f"  Content length: {len(doc.get('full_text', ''))} chars")
            print(f"  Summary length: {len(doc.get('summary_long', ''))} chars")

        print()
        print("=" * 70)
        print("TEST SUCCESSFUL!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_sample_processing())
