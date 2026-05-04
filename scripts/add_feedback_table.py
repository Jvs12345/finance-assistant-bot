"""
Database migration script to add the search_feedback table.

This script creates the feedback table for tracking user ratings on search results.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.postgres import PostgreSQLClient, Base
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def main():
    """Add search_feedback table to the database."""
    logger.info("Starting migration: Adding search_feedback table...")

    pg_client = PostgreSQLClient()

    try:
        # Create all tables (this will create search_feedback if it doesn't exist)
        # The documents table will remain unchanged
        Base.metadata.create_all(pg_client._engine)

        logger.info("✅ search_feedback table created/verified successfully")
        logger.info("Migration completed successfully!")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
