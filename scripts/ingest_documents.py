#!/usr/bin/env python3
"""
Document ingestion script for semantic search.

This script processes documents and indexes them with:
- Full text extraction
- Summary generation (short + long)
- Embedding computation
- Semantic search indexing

Usage:
    python scripts/ingest_documents.py --directory data/Source_files
    python scripts/ingest_documents.py --file document.pdf --category tax_law
    python scripts/ingest_documents.py --directory data --force
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.services.document_ingestion import get_ingestion_service
from src.utils.logging import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)


def ingest_file(
    file_path: Path,
    category: Optional[str] = None,
    metadata: Optional[dict] = None
):
    """
    Ingest a single file.

    Args:
        file_path: Path to file
        category: Optional category
        metadata: Optional metadata dict
    """
    ingestion = get_ingestion_service()

    logger.info(f"Processing: {file_path.name}")

    result = ingestion.ingest_document(
        file_path=file_path,
        category=category,
        metadata=metadata or {}
    )

    if result["status"] == "success":
        logger.info(f"✓ Successfully ingested: {file_path.name}")
        logger.info(f"  Title: {result['title']}")
        logger.info(f"  Summary: {result['summary_short'][:100]}...")
    else:
        logger.error(f"✗ Failed to ingest: {file_path.name}")
        logger.error(f"  Error: {result['error']}")


def ingest_directory(
    directory: Path,
    category: Optional[str] = None,
    recursive: bool = False
):
    """
    Ingest all supported files in a directory.

    Args:
        directory: Directory path
        category: Optional category for all files
        recursive: Whether to search subdirectories
    """
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return

    # Supported file extensions
    extensions = [
        '*.pdf', '*.csv', '*.xml', '*.html', '*.htm',
        '*.txt', '*.md', '*.json'
    ]

    # Find all files
    file_paths = []
    for ext in extensions:
        if recursive:
            file_paths.extend(directory.rglob(ext))
        else:
            file_paths.extend(directory.glob(ext))

    if not file_paths:
        logger.warning(f"No supported files found in {directory}")
        return

    logger.info(f"Found {len(file_paths)} files to process")

    # Bulk ingest
    ingestion = get_ingestion_service()
    results = ingestion.bulk_ingest(file_paths=file_paths)

    # Report results
    logger.info("=" * 60)
    logger.info("Ingestion Complete")
    logger.info("=" * 60)
    logger.info(f"Total files: {results['total']}")
    logger.info(f"Successful: {results['success']}")
    logger.info(f"Failed: {results['failed']}")

    if results['errors']:
        logger.error("\nErrors:")
        for error in results['errors']:
            logger.error(f"  - {error['filename']}: {error['error']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents for semantic search"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=Path,
        help="Single file to ingest"
    )
    group.add_argument(
        "--directory",
        type=Path,
        help="Directory of files to ingest"
    )

    parser.add_argument(
        "--category",
        type=str,
        help="Document category (tax_law, regulation, annual_report, etc.)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively"
    )
    parser.add_argument("--jurisdiction", type=str, help="Jurisdiction (e.g., Netherlands, EU)")
    parser.add_argument("--tax-year", type=int, help="Tax year (e.g., 2025)")
    parser.add_argument("--client-name", type=str, help="Client/entity name")
    parser.add_argument("--entity-type", type=str, help="Entity type (individual, bv, nv, etc.)")
    parser.add_argument("--source-name", type=str, help="Source name override for citations")

    args = parser.parse_args()

    # Build metadata
    metadata = {}
    if args.jurisdiction:
        metadata["jurisdiction"] = args.jurisdiction
    if args.tax_year:
        metadata["tax_year"] = args.tax_year
    if args.client_name:
        metadata["client_name"] = args.client_name
    if args.entity_type:
        metadata["entity_type"] = args.entity_type
    if args.source_name:
        metadata["source_name"] = args.source_name

    # Process
    try:
        if args.file:
            if not args.file.exists():
                logger.error(f"File not found: {args.file}")
                sys.exit(1)

            ingest_file(
                file_path=args.file,
                category=args.category,
                metadata=metadata if metadata else None
            )

        elif args.directory:
            ingest_directory(
                directory=args.directory,
                category=args.category,
                recursive=args.recursive
            )

        logger.info("\n✓ Ingestion complete!")

    except KeyboardInterrupt:
        logger.warning("\n\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
