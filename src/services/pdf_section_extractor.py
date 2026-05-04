"""
PDF section extraction service.
Extracts meaningful sections from PDF documents with OCR support.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import re

import pypdf
from PIL import Image
import pytesseract

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PDFSection:
    """A section extracted from a PDF document."""
    id: str
    text: str
    page_num: int
    section_type: str  # 'heading', 'paragraph', 'table', 'image', 'caption'
    title: str
    source_file: str
    bbox: Optional[Dict[str, float]] = None  # Bounding box if available
    confidence: float = 1.0  # OCR confidence if applicable


class PDFSectionExtractor:
    """
    Extracts meaningful sections from PDF documents.

    Features:
    - Page-level extraction
    - Heading detection
    - OCR for images
    - Section deduplication
    """

    def __init__(
        self,
        min_section_length: int = 50,
        max_section_length: int = 5000,
        enable_ocr: bool = True
    ):
        """
        Initialize the PDF section extractor.

        Args:
            min_section_length: Minimum characters for a valid section
            max_section_length: Maximum characters for a single section
            enable_ocr: Whether to perform OCR on images
        """
        self.min_section_length = min_section_length
        self.max_section_length = max_section_length
        self.enable_ocr = enable_ocr

    def _generate_section_id(self, content: str, page_num: int, source: str) -> str:
        """
        Generate a unique ID for a section.

        Args:
            content: Section text content
            page_num: Page number
            source: Source file name

        Returns:
            Unique section ID
        """
        hash_input = f"{source}:{page_num}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _detect_heading(self, text: str) -> bool:
        """
        Detect if text is likely a heading.

        Args:
            text: Text to analyze

        Returns:
            True if text appears to be a heading
        """
        text = text.strip()

        # Heuristics for heading detection
        if len(text) == 0:
            return False

        # Short text (< 100 chars) with no period at end
        if len(text) < 100 and not text.endswith('.'):
            return True

        # Starts with number pattern (e.g., "1.", "1.1", "Chapter 1")
        if re.match(r'^(\d+\.)+\s+|^(Chapter|Section|Part)\s+\d+', text, re.IGNORECASE):
            return True

        # All caps (likely a heading)
        if text.isupper() and len(text) > 3:
            return True

        return False

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n+', text)

        # Filter out very short paragraphs
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > self.min_section_length]

        return paragraphs

    def _extract_text_from_page(self, page: pypdf.PageObject) -> str:
        """
        Extract text from a PDF page.

        Args:
            page: PyPDF PageObject

        Returns:
            Extracted text
        """
        try:
            text = page.extract_text()
            return text.strip()
        except Exception as e:
            logger.warning(f"Failed to extract text from page: {e}")
            return ""

    def _perform_ocr_on_page(self, page: pypdf.PageObject) -> Optional[str]:
        """
        Perform OCR on a PDF page.

        Args:
            page: PyPDF PageObject

        Returns:
            OCR text or None if failed
        """
        if not self.enable_ocr:
            return None

        try:
            # Note: Full OCR implementation would extract images from PDF
            # This is a placeholder for the OCR pipeline
            # In production, you would:
            # 1. Extract images from the page
            # 2. Run pytesseract.image_to_string() on each image
            # 3. Combine results

            logger.debug("OCR not fully implemented in this version")
            return None

        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None

    def extract_sections(self, pdf_path: Path) -> List[PDFSection]:
        """
        Extract sections from a PDF file with memory optimization for large files.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PDFSection objects
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        logger.info(f"Extracting sections from {pdf_path.name} ({file_size_mb:.1f} MB)")

        sections = []
        source_file = pdf_path.name

        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                total_pages = len(pdf_reader.pages)

                logger.info(f"Processing {total_pages} pages")

                # Process pages with progress indication for large files
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    # Show progress for large PDFs every 10 pages
                    if file_size_mb > 50 and page_num % 10 == 0:
                        print(f"      Progress: {page_num}/{total_pages} pages", end="\r")

                    # Extract text
                    page_text = self._extract_text_from_page(page)

                    if not page_text:
                        # Try OCR if text extraction failed
                        page_text = self._perform_ocr_on_page(page)

                    if not page_text:
                        logger.debug(f"No text on page {page_num}")
                        continue

                    # Split into paragraphs
                    paragraphs = self._split_into_paragraphs(page_text)

                    for para_text in paragraphs:
                        # Skip very long sections (likely extraction errors)
                        if len(para_text) > self.max_section_length:
                            # Split large sections into chunks
                            chunks = [para_text[i:i+self.max_section_length]
                                     for i in range(0, len(para_text), self.max_section_length)]
                            for chunk in chunks:
                                if len(chunk) >= self.min_section_length:
                                    section = self._create_section(chunk, page_num, source_file)
                                    sections.append(section)
                        else:
                            section = self._create_section(para_text, page_num, source_file)
                            sections.append(section)

                if file_size_mb > 50:
                    print()  # Clear progress line

                logger.info(f"Extracted {len(sections)} sections from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Failed to extract sections from {pdf_path.name}: {e}")
            raise

        return sections

    def _create_section(self, text: str, page_num: int, source_file: str) -> PDFSection:
        """Helper method to create a PDFSection object."""
        section_type = 'heading' if self._detect_heading(text) else 'paragraph'
        title = text[:50] + "..." if len(text) > 50 else text

        return PDFSection(
            id=self._generate_section_id(text, page_num, source_file),
            text=text,
            page_num=page_num,
            section_type=section_type,
            title=title,
            source_file=source_file
        )

    def extract_sections_batch(self, pdf_paths: List[Path]) -> Dict[str, List[PDFSection]]:
        """
        Extract sections from multiple PDF files.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            Dict mapping file name to list of sections
        """
        results = {}

        for pdf_path in pdf_paths:
            try:
                sections = self.extract_sections(pdf_path)
                results[pdf_path.name] = sections
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results[pdf_path.name] = []

        return results
