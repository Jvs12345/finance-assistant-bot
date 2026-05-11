"""
XAF conversion service.

Converts .xaf files into normalized text artifacts for indexing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.services.multi_format_processor import MultiFormatProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class XafConversionService:
    """Convert XAF files into plain-text artifacts."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("data") / "xaf_converted"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = MultiFormatProcessor()

    def convert_to_text_file(self, xaf_path: Path) -> Path:
        text = self.processor.extract_text_from_xaf(xaf_path)
        out_path = self.output_dir / f"{xaf_path.stem}.xaf.txt"
        out_path.write_text(text, encoding="utf-8")
        return out_path

    def convert_to_pdf_file(self, xaf_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Convert XAF to a readable PDF preview.
        Returns None if conversion cannot be performed.
        """
        try:
            from xaf_to_pdf import build_pdf
        except Exception as exc:
            logger.warning(f"XAF PDF conversion unavailable (xaf_to_pdf import failed): {exc}")
            return None

        try:
            out_path = output_path or xaf_path.with_suffix(".pdf")
            build_pdf(xaf_path, out_path)
            return out_path
        except Exception as exc:
            logger.warning(f"Failed to build XAF PDF preview for {xaf_path.name}: {exc}")
            return None


_xaf_converter: XafConversionService | None = None


def get_xaf_conversion_service() -> XafConversionService:
    global _xaf_converter
    if _xaf_converter is None:
        _xaf_converter = XafConversionService()
    return _xaf_converter
