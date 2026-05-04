"""
PDF utility functions for page counting and limiting.

Temporary solution to handle LandingAI's 50-page limit.
"""

from pathlib import Path
from typing import Tuple
from pypdf import PdfReader, PdfWriter

from src.utils.logging import get_logger

logger = get_logger(__name__)

# LandingAI's hard limit for PDF pages
LANDINGAI_MAX_PAGES = 50


def get_pdf_page_count(file_path: Path) -> int:
    """
    Get the number of pages in a PDF file.

    Args:
        file_path: Path to PDF file

    Returns:
        int: Number of pages in the PDF

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If PDF cannot be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        reader = PdfReader(file_path)
        page_count = len(reader.pages)

        logger.debug(f"PDF {file_path.name} has {page_count} pages")
        return page_count

    except Exception as e:
        logger.error(f"Failed to read PDF {file_path.name}: {e}")
        raise


def limit_pdf_to_max_pages(
    file_path: Path,
    max_pages: int = LANDINGAI_MAX_PAGES
) -> Tuple[Path, int, bool]:
    """
    Create a limited version of PDF if it exceeds max pages.

    If PDF has <= max_pages, returns original file.
    If PDF has > max_pages, creates a new PDF with only the first max_pages.

    Args:
        file_path: Path to original PDF file
        max_pages: Maximum number of pages to keep (default: 50)

    Returns:
        Tuple[Path, int, bool]: (path_to_use, original_page_count, was_truncated)
            - path_to_use: Path to the PDF to use (original or limited)
            - original_page_count: Total pages in original PDF
            - was_truncated: True if PDF was truncated

    Raises:
        Exception: If PDF reading or writing fails
    """
    # Get original page count
    original_page_count = get_pdf_page_count(file_path)

    # If within limit, return original
    if original_page_count <= max_pages:
        logger.info(
            f"PDF {file_path.name} has {original_page_count} pages "
            f"(within {max_pages} page limit)"
        )
        return file_path, original_page_count, False

    # PDF exceeds limit, create truncated version
    logger.warning(
        f"PDF {file_path.name} has {original_page_count} pages, "
        f"exceeding {max_pages} page limit. Creating truncated version..."
    )

    try:
        reader = PdfReader(file_path)
        writer = PdfWriter()

        # Add first max_pages to new PDF
        for i in range(min(max_pages, original_page_count)):
            writer.add_page(reader.pages[i])

        # Create limited PDF with "_limited" suffix
        limited_path = file_path.parent / f"{file_path.stem}_limited{file_path.suffix}"

        with open(limited_path, 'wb') as output_file:
            writer.write(output_file)

        limited_size = limited_path.stat().st_size
        original_size = file_path.stat().st_size

        logger.warning(
            f"Created limited PDF: {limited_path.name} "
            f"({max_pages} pages, {limited_size:,} bytes) "
            f"from {file_path.name} ({original_page_count} pages, {original_size:,} bytes)"
        )

        return limited_path, original_page_count, True

    except Exception as e:
        logger.error(f"Failed to create limited PDF for {file_path.name}: {e}")
        raise


def cleanup_limited_pdf(file_path: Path) -> None:
    """
    Remove a limited PDF file if it exists.

    Args:
        file_path: Path to limited PDF file
    """
    if file_path.exists() and "_limited" in file_path.stem:
        try:
            file_path.unlink()
            logger.debug(f"Cleaned up limited PDF: {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup limited PDF {file_path.name}: {e}")


def check_pdf_compatibility(file_path: Path) -> dict:
    """
    Check if PDF is compatible with LandingAI processing.

    Returns diagnostic information about the PDF.

    Args:
        file_path: Path to PDF file

    Returns:
        dict: Compatibility information
            - page_count: Number of pages
            - within_limit: True if <= 50 pages
            - needs_truncation: True if > 50 pages
            - file_size_mb: File size in MB
            - recommendation: String recommendation
    """
    page_count = get_pdf_page_count(file_path)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    within_limit = page_count <= LANDINGAI_MAX_PAGES
    needs_truncation = page_count > LANDINGAI_MAX_PAGES

    if within_limit:
        recommendation = "PDF is compatible with LandingAI processing"
    else:
        pages_to_remove = page_count - LANDINGAI_MAX_PAGES
        recommendation = (
            f"PDF exceeds {LANDINGAI_MAX_PAGES}-page limit. "
            f"Will truncate last {pages_to_remove} pages for processing."
        )

    return {
        "page_count": page_count,
        "within_limit": within_limit,
        "needs_truncation": needs_truncation,
        "file_size_mb": round(file_size_mb, 2),
        "recommendation": recommendation,
        "max_pages_allowed": LANDINGAI_MAX_PAGES
    }
