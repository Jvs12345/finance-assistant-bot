#!/usr/bin/env python3
"""
Migration script to add original_filename column to documents table.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.db.postgres import PostgreSQLClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Add original_filename column to documents table."""
    logger.info("Starting migration: Adding original_filename column...")

    # Initialize PostgreSQL client
    pg_client = PostgreSQLClient()

    # Add column using raw SQL
    try:
        with pg_client._engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='documents' AND column_name='original_filename';
            """))

            if result.fetchone():
                logger.info("✅ Column 'original_filename' already exists")
                return

            # Add the column
            conn.execute(text("""
                ALTER TABLE documents
                ADD COLUMN original_filename VARCHAR(255) NOT NULL DEFAULT 'unknown.pdf';
            """))
            conn.commit()

            logger.info("✅ Successfully added 'original_filename' column")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
