"""
Markdown chunking service for splitting documents by page.
"""

import re
from typing import List, Dict, Any
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MarkdownChunker:
    """Chunk markdown content by page boundaries."""

    # Pattern to match page markers in LandingAI markdown output
    # Examples: "Page: 1 of 51", "Page:   2 of 51"
    PAGE_PATTERN = re.compile(
        r'<tr><td[^>]*>Page:</td><td[^>]*>(\d+)\s+of\s+(\d+)</td></tr>',
        re.IGNORECASE
    )

    def chunk_by_page(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Split markdown content into page-level chunks.

        Args:
            markdown_content: Full markdown content from PDF parser

        Returns:
            list: List of page chunks with metadata

        Example:
            [
                {
                    "page": 1,
                    "content": "...",
                    "total_pages": 51
                },
                ...
            ]
        """
        logger.info("Chunking markdown content by page")

        # Find all page markers
        page_markers = list(self.PAGE_PATTERN.finditer(markdown_content))

        if not page_markers:
            logger.warning("No page markers found, returning content as single chunk")
            return [{
                "page": 1,
                "content": markdown_content.strip(),
                "total_pages": 1
            }]

        total_pages = int(page_markers[0].group(2))
        chunks = []

        for i, match in enumerate(page_markers):
            page_num = int(match.group(1))

            # Determine start and end positions
            start_pos = match.end()

            if i < len(page_markers) - 1:
                # Content ends at the next page marker
                end_pos = page_markers[i + 1].start()
            else:
                # Last page goes to end of document
                end_pos = len(markdown_content)

            # Extract page content
            page_content = markdown_content[start_pos:end_pos].strip()

            # Skip empty pages
            if not page_content:
                logger.debug(f"Skipping empty page {page_num}")
                continue

            chunks.append({
                "page": page_num,
                "content": page_content,
                "total_pages": total_pages
            })

        logger.info(f"Created {len(chunks)} page chunks from {total_pages} total pages")

        return chunks

    def extract_metadata(self, page_content: str) -> Dict[str, Any]:
        """
        Extract metadata from page content (tables, headers, etc.).

        Args:
            page_content: Content of a single page

        Returns:
            dict: Extracted metadata (part numbers, headers, etc.)
        """
        metadata = {
            "part_numbers": [],
            "headers": [],
            "has_tables": False,
            "has_images": False
        }

        # Extract headers
        header_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)
        headers = header_pattern.findall(page_content)
        if headers:
            metadata["headers"] = headers

        # Check for tables
        if '<table' in page_content.lower() or '|' in page_content:
            metadata["has_tables"] = True

        # Check for images
        if '![' in page_content or '<img' in page_content.lower():
            metadata["has_images"] = True

        # Extract potential part numbers (alphanumeric patterns)
        # Common patterns: ABC-123, 12345-6789, P/N: ABC123
        part_number_patterns = [
            r'\b([A-Z]{2,}-\d{2,})\b',  # ABC-123
            r'\b(\d{4,}-\d{2,})\b',  # 12345-67
            r'P/N:?\s*([A-Z0-9-]+)',  # P/N: ABC123
            r'Part\s+(?:Number|No\.?):?\s*([A-Z0-9-]+)',  # Part Number: ABC123
        ]

        part_numbers = set()
        for pattern in part_number_patterns:
            matches = re.findall(pattern, page_content, re.IGNORECASE)
            part_numbers.update(matches)

        if part_numbers:
            metadata["part_numbers"] = sorted(list(part_numbers))

        return metadata


# Global markdown chunker instance
_markdown_chunker: MarkdownChunker = None


def get_markdown_chunker() -> MarkdownChunker:
    """
    Get the global markdown chunker instance (singleton pattern).

    Returns:
        MarkdownChunker: The markdown chunker instance
    """
    global _markdown_chunker
    if _markdown_chunker is None:
        _markdown_chunker = MarkdownChunker()
    return _markdown_chunker
