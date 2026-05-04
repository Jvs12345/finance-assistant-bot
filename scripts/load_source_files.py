"""
Script to load files from Source_files directory into the search system.
"""
import os
import uuid
from datetime import datetime
from src.db.postgres import get_postgres_client
from src.db.simple_whoosh import get_search_client
from enum import Enum

class ProcessingStatus(str, Enum):
    """Document processing status enumeration."""
    UPLOADED = "uploaded"
    INDEXED = "indexed"
    FAILED = "failed"

class ProcessingStatus(str, Enum):
    """Document processing status enumeration."""
    UPLOADED = "uploaded"
    INDEXED = "indexed"
    FAILED = "failed"
from src.services.simple_document_processor import process_document_content
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_source_files():
    """Load files from Source_files directory into the search system."""
    source_dir = "Source_files"
    logger.info(f"Starting to load files from {source_dir}")
    db_client = get_postgres_client()
    search_client = get_search_client()
    logger.info("Database and search clients initialized")

    # Create data directories if they don't exist
    os.makedirs("data/pdfs", exist_ok=True)

    # Process all files in the Source_files directory
    files = os.listdir(source_dir)
    logger.info(f"Found {len(files)} files in source directory: {files}")
    
    for filename in files:
        file_path = os.path.join(source_dir, filename)
        logger.info(f"Processing file: {filename}")
        
        try:
            # Generate a unique ID for the document
            document_id = str(uuid.uuid4())
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Create document record in database
            doc = db_client.create_document(
                document_id=document_id,
                filename=filename,  # Use original filename
                original_filename=filename,
                file_path=file_path,
                file_size=file_size
            )
            
            # Process document content
            logger.info(f"Processing content of {filename}")
            pages = process_document_content(file_path)
            logger.info(f"Extracted {len(pages)} pages from {filename}")
            
            # Index each page in Whoosh
            logger.info("Starting to index pages in Whoosh")
            for page_num, content in enumerate(pages, 1):
                search_client.index_document(
                    document_id=document_id,
                    content=content,
                    page_number=page_num,
                    filename=filename
                )
            
            # Update document status
            db_client.update_document_status(
                document_id=document_id,
                status=ProcessingStatus.INDEXED,
                total_pages=len(pages),
                indexed_at=datetime.utcnow()
            )
            
            logger.info(f"Successfully processed and indexed {filename}")
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            # Update document status if it was created
            if 'document_id' in locals():
                db_client.update_document_status(
                    document_id=document_id,
                    status=ProcessingStatus.FAILED,
                    error_message=str(e)
                )

if __name__ == "__main__":
    load_source_files()