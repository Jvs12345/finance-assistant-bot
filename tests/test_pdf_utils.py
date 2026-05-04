"""
Tests for PDF utility functions.
"""

import pytest
from pathlib import Path
from pypdf import PdfReader, PdfWriter

from src.utils.pdf_utils import (
    get_pdf_page_count,
    limit_pdf_to_max_pages,
    cleanup_limited_pdf,
    check_pdf_compatibility,
    LANDINGAI_MAX_PAGES
)


def create_test_pdf(path: Path, num_pages: int) -> Path:
    """
    Create a test PDF with specified number of pages.

    Args:
        path: Path where PDF should be created
        num_pages: Number of pages to create

    Returns:
        Path: Path to created PDF
    """
    writer = PdfWriter()

    # Add pages
    for i in range(num_pages):
        writer.add_blank_page(width=612, height=792)  # Standard letter size

    with open(path, 'wb') as f:
        writer.write(f)

    return path


@pytest.fixture
def temp_pdf_dir(tmp_path):
    """Create temporary directory for test PDFs."""
    pdf_dir = tmp_path / "test_pdfs"
    pdf_dir.mkdir()
    return pdf_dir


class TestGetPDFPageCount:
    """Test PDF page counting."""

    def test_count_single_page_pdf(self, temp_pdf_dir):
        """Test counting pages in a 1-page PDF."""
        pdf_path = create_test_pdf(temp_pdf_dir / "single.pdf", 1)
        count = get_pdf_page_count(pdf_path)
        assert count == 1

    def test_count_multi_page_pdf(self, temp_pdf_dir):
        """Test counting pages in a multi-page PDF."""
        pdf_path = create_test_pdf(temp_pdf_dir / "multi.pdf", 25)
        count = get_pdf_page_count(pdf_path)
        assert count == 25

    def test_count_exactly_50_pages(self, temp_pdf_dir):
        """Test counting pages in a 50-page PDF."""
        pdf_path = create_test_pdf(temp_pdf_dir / "fifty.pdf", 50)
        count = get_pdf_page_count(pdf_path)
        assert count == 50

    def test_count_over_50_pages(self, temp_pdf_dir):
        """Test counting pages in a 75-page PDF."""
        pdf_path = create_test_pdf(temp_pdf_dir / "large.pdf", 75)
        count = get_pdf_page_count(pdf_path)
        assert count == 75

    def test_count_nonexistent_file(self):
        """Test counting pages in non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_pdf_page_count(Path("/nonexistent/file.pdf"))


class TestLimitPDFToMaxPages:
    """Test PDF page limiting."""

    def test_no_limit_needed_small_pdf(self, temp_pdf_dir):
        """Test PDF with fewer pages than limit (no change)."""
        pdf_path = create_test_pdf(temp_pdf_dir / "small.pdf", 10)

        result_path, page_count, was_truncated = limit_pdf_to_max_pages(pdf_path)

        assert result_path == pdf_path  # Same file returned
        assert page_count == 10
        assert was_truncated is False

    def test_no_limit_needed_exactly_50(self, temp_pdf_dir):
        """Test PDF with exactly 50 pages (no change)."""
        pdf_path = create_test_pdf(temp_pdf_dir / "exact.pdf", 50)

        result_path, page_count, was_truncated = limit_pdf_to_max_pages(pdf_path)

        assert result_path == pdf_path  # Same file returned
        assert page_count == 50
        assert was_truncated is False

    def test_limit_applied_over_50(self, temp_pdf_dir):
        """Test PDF with more than 50 pages (truncated)."""
        pdf_path = create_test_pdf(temp_pdf_dir / "large.pdf", 75)

        result_path, page_count, was_truncated = limit_pdf_to_max_pages(pdf_path)

        # Should return different path
        assert result_path != pdf_path
        assert "_limited" in result_path.stem
        assert page_count == 75  # Original page count
        assert was_truncated is True

        # Verify limited PDF has exactly 50 pages
        limited_count = get_pdf_page_count(result_path)
        assert limited_count == 50

        # Verify limited file exists
        assert result_path.exists()

    def test_limit_applied_100_pages(self, temp_pdf_dir):
        """Test PDF with 100 pages (truncated to 50)."""
        pdf_path = create_test_pdf(temp_pdf_dir / "huge.pdf", 100)

        result_path, page_count, was_truncated = limit_pdf_to_max_pages(pdf_path)

        assert was_truncated is True
        assert page_count == 100

        # Verify limited PDF
        limited_count = get_pdf_page_count(result_path)
        assert limited_count == 50

    def test_custom_max_pages(self, temp_pdf_dir):
        """Test with custom max_pages parameter."""
        pdf_path = create_test_pdf(temp_pdf_dir / "custom.pdf", 30)

        result_path, page_count, was_truncated = limit_pdf_to_max_pages(
            pdf_path, max_pages=20
        )

        assert was_truncated is True
        assert page_count == 30

        # Verify limited PDF has exactly 20 pages
        limited_count = get_pdf_page_count(result_path)
        assert limited_count == 20

    def test_limited_filename_format(self, temp_pdf_dir):
        """Test that limited PDF has correct filename format."""
        pdf_path = create_test_pdf(temp_pdf_dir / "document.pdf", 60)

        result_path, _, was_truncated = limit_pdf_to_max_pages(pdf_path)

        assert was_truncated is True
        assert result_path.name == "document_limited.pdf"
        assert result_path.parent == pdf_path.parent


class TestCleanupLimitedPDF:
    """Test cleanup of limited PDFs."""

    def test_cleanup_limited_pdf(self, temp_pdf_dir):
        """Test cleanup removes limited PDF."""
        # Create a limited PDF
        original = create_test_pdf(temp_pdf_dir / "doc.pdf", 60)
        limited_path, _, _ = limit_pdf_to_max_pages(original)

        assert limited_path.exists()

        # Cleanup
        cleanup_limited_pdf(limited_path)

        # Verify removed
        assert not limited_path.exists()

    def test_cleanup_nonexistent_file(self, temp_pdf_dir):
        """Test cleanup handles non-existent file gracefully."""
        nonexistent = temp_pdf_dir / "nonexistent_limited.pdf"

        # Should not raise error
        cleanup_limited_pdf(nonexistent)

    def test_cleanup_ignores_non_limited_files(self, temp_pdf_dir):
        """Test cleanup only removes files with '_limited' in name."""
        normal_pdf = create_test_pdf(temp_pdf_dir / "normal.pdf", 10)

        # Try to cleanup normal PDF (should be ignored by name check)
        cleanup_limited_pdf(normal_pdf)

        # Normal PDF should not be deleted (no _limited in name)
        assert normal_pdf.exists()


class TestCheckPDFCompatibility:
    """Test PDF compatibility checking."""

    def test_compatible_small_pdf(self, temp_pdf_dir):
        """Test compatibility check for small PDF."""
        pdf_path = create_test_pdf(temp_pdf_dir / "small.pdf", 10)

        result = check_pdf_compatibility(pdf_path)

        assert result["page_count"] == 10
        assert result["within_limit"] is True
        assert result["needs_truncation"] is False
        assert result["max_pages_allowed"] == LANDINGAI_MAX_PAGES
        assert "compatible" in result["recommendation"].lower()

    def test_compatible_exactly_50(self, temp_pdf_dir):
        """Test compatibility check for exactly 50-page PDF."""
        pdf_path = create_test_pdf(temp_pdf_dir / "exact.pdf", 50)

        result = check_pdf_compatibility(pdf_path)

        assert result["page_count"] == 50
        assert result["within_limit"] is True
        assert result["needs_truncation"] is False

    def test_incompatible_large_pdf(self, temp_pdf_dir):
        """Test compatibility check for large PDF."""
        pdf_path = create_test_pdf(temp_pdf_dir / "large.pdf", 75)

        result = check_pdf_compatibility(pdf_path)

        assert result["page_count"] == 75
        assert result["within_limit"] is False
        assert result["needs_truncation"] is True
        assert "truncate" in result["recommendation"].lower()
        assert "25 pages" in result["recommendation"]  # 75 - 50 = 25

    def test_file_size_included(self, temp_pdf_dir):
        """Test that file size is included in compatibility check."""
        pdf_path = create_test_pdf(temp_pdf_dir / "test.pdf", 10)

        result = check_pdf_compatibility(pdf_path)

        assert "file_size_mb" in result
        assert isinstance(result["file_size_mb"], (int, float))
        assert result["file_size_mb"] >= 0  # File size can be very small (0.0x MB)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_full_workflow_small_pdf(self, temp_pdf_dir):
        """Test complete workflow with small PDF."""
        # Create small PDF
        pdf_path = create_test_pdf(temp_pdf_dir / "workflow.pdf", 25)

        # Check compatibility
        compat = check_pdf_compatibility(pdf_path)
        assert compat["within_limit"] is True

        # Limit (should return same file)
        result_path, count, truncated = limit_pdf_to_max_pages(pdf_path)
        assert result_path == pdf_path
        assert not truncated

        # No cleanup needed
        assert pdf_path.exists()

    def test_full_workflow_large_pdf(self, temp_pdf_dir):
        """Test complete workflow with large PDF."""
        # Create large PDF
        pdf_path = create_test_pdf(temp_pdf_dir / "workflow_large.pdf", 80)

        # Check compatibility
        compat = check_pdf_compatibility(pdf_path)
        assert compat["within_limit"] is False
        assert compat["needs_truncation"] is True

        # Limit (should create new file)
        result_path, count, truncated = limit_pdf_to_max_pages(pdf_path)
        assert result_path != pdf_path
        assert truncated
        assert count == 80

        # Verify limited PDF
        limited_count = get_pdf_page_count(result_path)
        assert limited_count == 50

        # Cleanup
        cleanup_limited_pdf(result_path)
        assert not result_path.exists()

        # Original still exists
        assert pdf_path.exists()

    def test_multiple_pdfs_same_directory(self, temp_pdf_dir):
        """Test handling multiple PDFs in same directory."""
        # Create multiple PDFs
        pdf1 = create_test_pdf(temp_pdf_dir / "doc1.pdf", 60)
        pdf2 = create_test_pdf(temp_pdf_dir / "doc2.pdf", 70)

        # Limit both
        limited1, _, _ = limit_pdf_to_max_pages(pdf1)
        limited2, _, _ = limit_pdf_to_max_pages(pdf2)

        # Verify unique limited files
        assert limited1 != limited2
        assert limited1.exists()
        assert limited2.exists()

        # Cleanup
        cleanup_limited_pdf(limited1)
        cleanup_limited_pdf(limited2)

        # Verify cleanup
        assert not limited1.exists()
        assert not limited2.exists()

        # Originals still exist
        assert pdf1.exists()
        assert pdf2.exists()
