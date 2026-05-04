"""
Initialize SQLite database for the Document Search & Retrieval System.
"""

from src.db.postgres import Base, get_postgres_client
from src.utils.logging import get_logger

logger = get_logger(__name__)

def init_database():
    """Initialize SQLite database tables."""
    try:
        # Get database client
        db_client = get_postgres_client()
        
        # Create tables
        db_client.create_tables()
        
        logger.info("SQLite database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    init_database()