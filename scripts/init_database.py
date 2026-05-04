#!/usr/bin/env python
"""
Initialize PostgreSQL database tables for Document Search & Retrieval System.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.postgres import get_postgres_client
from src.utils.logging import setup_logging, get_logger
from src.config import settings

# Setup logging
setup_logging(log_level=settings.log_level)
logger = get_logger(__name__)


def main():
    """Initialize PostgreSQL database tables."""
    try:
        logger.info("Starting PostgreSQL database initialization...")

        # Get PostgreSQL client
        db_client = get_postgres_client()

        # Create tables
        logger.info("Creating database tables...")
        db_client.create_tables()

        logger.info("✅ Database tables created successfully")
        logger.info("✨ PostgreSQL initialization complete!")
        return 0

    except Exception as e:
        logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
