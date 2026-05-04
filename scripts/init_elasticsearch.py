#!/usr/bin/env python
"""
Initialize Elasticsearch indices for Document Search & Retrieval System.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.elasticsearch import get_elasticsearch_client
from src.db.index_schemas import create_documents_index
from src.utils.logging import setup_logging, get_logger
from src.config import settings

# Setup logging
setup_logging(log_level=settings.log_level)
logger = get_logger(__name__)


def main():
    """Initialize Elasticsearch indices."""
    try:
        logger.info("Starting Elasticsearch index initialization...")

        # Get Elasticsearch client
        es_client = get_elasticsearch_client()

        # Check Elasticsearch health
        logger.info("Checking Elasticsearch health...")
        health = es_client.health_check()
        logger.info(f"Elasticsearch cluster status: {health['status']}")

        # Create documents index
        logger.info("Creating documents index...")
        created = create_documents_index(es_client)

        if created:
            logger.info("✅ Documents index created successfully")
        else:
            logger.info("ℹ️  Documents index already exists")

        logger.info("✨ Elasticsearch initialization complete!")
        return 0

    except Exception as e:
        logger.error(f"❌ Failed to initialize Elasticsearch: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
