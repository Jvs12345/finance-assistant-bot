"""
Tests for document processing pipeline components.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.services.pdf_parser import PDFParser
from src.services.markdown_chunker import MarkdownChunker
from src.services.summarizer import Summarizer
from src.models.document import DocumentCategory, ProcessingStatus


class TestMarkdownChunker:
    """Test markdown chunking functionality."""

    @pytest.fixture
    def chunker(self):
        """Get markdown chunker instance."""
        return MarkdownChunker()

    @pytest.mark.unit
    def test_chunk_by_page_with_markers(self, chunker):
        """Test chunking markdown with page markers."""
        markdown = """
<tr><td>Page:</td><td>1 of 3</td></tr>
This is content for page 1.
<tr><td>Page:</td><td>2 of 3</td></tr>
This is content for page 2.
<tr><td>Page:</td><td>3 of 3</td></tr>
This is content for page 3.
"""
        chunks = chunker.chunk_by_page(markdown)

        assert len(chunks) == 3
        assert chunks[0]["page"] == 1
        assert chunks[0]["total_pages"] == 3
        assert "page 1" in chunks[0]["content"]
        assert chunks[1]["page"] == 2
        assert "page 2" in chunks[1]["content"]
        assert chunks[2]["page"] == 3
        assert "page 3" in chunks[2]["content"]

    @pytest.mark.unit
    def test_chunk_no_markers(self, chunker):
        """Test chunking markdown without page markers."""
        markdown = "This is a single page document with no markers."
        chunks = chunker.chunk_by_page(markdown)

        assert len(chunks) == 1
        assert chunks[0]["page"] == 1
        assert chunks[0]["total_pages"] == 1
        assert chunks[0]["content"] == markdown.strip()

    @pytest.mark.unit
    def test_extract_metadata_headers(self, chunker):
        """Test extracting headers from markdown."""
        content = """
# Main Title
## Subsection
Some content here
### Sub-subsection
More content
"""
        metadata = chunker.extract_metadata(content)

        assert len(metadata["headers"]) == 3
        assert "Main Title" in metadata["headers"]
        assert "Subsection" in metadata["headers"]

    @pytest.mark.unit
    def test_extract_metadata_tables(self, chunker):
        """Test detecting tables in markdown."""
        content_with_table = """
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
"""
        metadata = chunker.extract_metadata(content_with_table)
        assert metadata["has_tables"] is True

        content_without_table = "Just plain text"
        metadata = chunker.extract_metadata(content_without_table)
        assert metadata["has_tables"] is False

    @pytest.mark.unit
    def test_extract_part_numbers(self, chunker):
        """Test extracting part numbers from content."""
        content = """
Part Number: ABC-123
P/N: XYZ-456
Reference: 12345-67
"""
        metadata = chunker.extract_metadata(content)
        part_numbers = metadata["part_numbers"]

        assert len(part_numbers) > 0
        assert any("ABC-123" in pn for pn in part_numbers)


class TestPDFParser:
    """Test PDF parsing functionality."""

    @pytest.mark.unit
    def test_pdf_parser_initialization(self):
        """Test PDF parser initialization."""
        with patch('src.services.pdf_parser.LandingAIADE') as mock_client:
            parser = PDFParser()
            assert parser is not None
            mock_client.assert_called_once()

    @pytest.mark.unit
    def test_parse_pdf_file_not_found(self):
        """Test parsing non-existent PDF file."""
        with patch('src.services.pdf_parser.LandingAIADE'):
            parser = PDFParser()
            non_existent_file = Path("/nonexistent/file.pdf")

            with pytest.raises(FileNotFoundError):
                parser.parse_pdf(non_existent_file)

    @pytest.mark.unit
    def test_parse_pdf_invalid_file_type(self):
        """Test parsing non-PDF file."""
        with patch('src.services.pdf_parser.LandingAIADE'):
            parser = PDFParser()

            # Create a temporary non-PDF file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                txt_file = Path(f.name)
                f.write(b"Not a PDF")

            try:
                with pytest.raises(ValueError, match="not a PDF"):
                    parser.parse_pdf(txt_file)
            finally:
                txt_file.unlink()


class TestSummarizer:
    """Test summarization functionality."""

    @pytest.mark.unit
    def test_summarizer_initialization(self):
        """Test summarizer initialization."""
        with patch('src.services.summarizer.Anthropic') as mock_client:
            summarizer = Summarizer()
            assert summarizer is not None
            mock_client.assert_called_once()

    @pytest.mark.unit
    def test_summarize_text_too_short(self):
        """Test summarizing text that's too short."""
        with patch('src.services.summarizer.Anthropic'):
            summarizer = Summarizer()

            with pytest.raises(ValueError, match="too short"):
                summarizer.summarize_text("Short")

    @pytest.mark.unit
    def test_summarize_text_success(self):
        """Test successful text summarization."""
        with patch('src.services.summarizer.Anthropic') as mock_anthropic:
            # Mock the response
            mock_message = Mock()
            mock_content = Mock()
            mock_content.text = "This is a summary of the technical document."
            mock_message.content = [mock_content]
            mock_message.usage.input_tokens = 100
            mock_message.usage.output_tokens = 50

            mock_client = Mock()
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            summarizer = Summarizer()
            content = "This is a long technical document " * 50  # Make it long enough

            summary = summarizer.summarize_text(content)

            assert summary == "This is a summary of the technical document."
            mock_client.messages.create.assert_called_once()

    @pytest.mark.unit
    def test_batch_summarize(self):
        """Test batch summarization."""
        with patch('src.services.summarizer.Anthropic') as mock_anthropic:
            # Mock responses
            mock_message = Mock()
            mock_content = Mock()
            mock_content.text = "Summary"
            mock_message.content = [mock_content]
            mock_message.usage.input_tokens = 100
            mock_message.usage.output_tokens = 20

            mock_client = Mock()
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            summarizer = Summarizer()
            contents = [
                "Long technical content " * 50,
                "Another long document " * 50
            ]

            summaries = summarizer.batch_summarize(contents)

            assert len(summaries) == 2
            assert all(s == "Summary" for s in summaries)


class TestPostgreSQLClient:
    """Test PostgreSQL client functionality."""

    @pytest.mark.unit
    def test_document_model_to_dict(self):
        """Test converting document model to dictionary."""
        from src.db.postgres import Document

        doc = Document(
            id="test-123",
            filename="uuid-filename.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            category=DocumentCategory.MAINTENANCE,
            processing_status=ProcessingStatus.UPLOADED
        )

        doc_dict = doc.to_dict()

        assert doc_dict["document_id"] == "test-123"
        assert doc_dict["filename"] == "test.pdf"
        assert doc_dict["category"] == "maintenance"
        assert doc_dict["processing_status"] == "uploaded"


class TestDocumentProcessor:
    """Test document processing pipeline."""

    @pytest.mark.unit
    def test_processor_initialization(self):
        """Test document processor initialization."""
        with patch('src.services.document_processor.get_pdf_parser'), \
             patch('src.services.document_processor.get_markdown_chunker'), \
             patch('src.services.document_processor.get_summarizer'), \
             patch('src.services.document_processor.get_elasticsearch_client'):

            from src.services.document_processor import DocumentProcessor
            processor = DocumentProcessor()

            assert processor is not None
            assert processor.pdf_parser is not None
            assert processor.chunker is not None
            assert processor.summarizer is not None
            assert processor.es_client is not None
