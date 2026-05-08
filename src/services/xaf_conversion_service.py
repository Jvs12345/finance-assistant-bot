"""
XAF conversion service.

Converts .xaf files into normalized text artifacts for indexing.
"""

from __future__ import annotations

from pathlib import Path

from src.services.multi_format_processor import MultiFormatProcessor


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


_xaf_converter: XafConversionService | None = None


def get_xaf_conversion_service() -> XafConversionService:
    global _xaf_converter
    if _xaf_converter is None:
        _xaf_converter = XafConversionService()
    return _xaf_converter

