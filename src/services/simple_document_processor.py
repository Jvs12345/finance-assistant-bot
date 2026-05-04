"""
Simple document processing utilities for text extraction from PDFs.
"""

import pypdf
from typing import List
from src.utils.logging import get_logger

logger = get_logger(__name__)

def process_document_content(file_path: str) -> List[str]:
    """
    Extract text content from a PDF file, page by page.

    Args:
        file_path: Path to the PDF file

    Returns:
        List[str]: List of text content for each page
    """
    try:
        logger.info(f"Opening PDF file: {file_path}")
        pages = []
        
        with open(file_path, 'rb') as file:
            pdf = pypdf.PdfReader(file)
            logger.info(f"PDF has {len(pdf.pages)} pages")
            
            for i, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        pages.append(text)
                    logger.info(f"Extracted page {i}")
                except Exception as e:
                    logger.error(f"Error extracting page {i}: {e}")
                    pages.append("")  # Add empty string for failed pages
        
        return pages
    
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {e}")
        raise