"""
PDF parsing service using LandingAI ADE SDK.
"""

from pathlib import Path
from typing import Optional
from landingai_ade import LandingAIADE

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PDFParser:
    """PDF parser using LandingAI's Document Processing Engine."""

    def __init__(self):
        """Initialize LandingAI client."""
        self._client: Optional[LandingAIADE] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Create LandingAI client with API key."""
        try:
            self._client = LandingAIADE(
                apikey=settings.vision_agent_api_key
            )
            logger.info("LandingAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LandingAI client: {e}")
            raise

    @property
    def client(self) -> LandingAIADE:
        """Get the LandingAI client instance."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def parse_pdf(
        self,
        file_path: Path,
        model: str = "dpt-2-latest"
    ) -> str:
        """
        Parse a PDF file to markdown format.

        Args:
            file_path: Path to the PDF file
            model: LandingAI model to use (default: dpt-2-latest)

        Returns:
            str: Markdown content extracted from PDF

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If parsing fails or no markdown content returned
            Exception: For other parsing errors
        """
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")

        logger.info(f"Parsing PDF: {file_path.name} (size: {file_path.stat().st_size} bytes)")

        try:
            # Parse PDF with LandingAI
            parse_response = self.client.parse(
                document=file_path,
                model=model
            )

            # Extract markdown content
            if not hasattr(parse_response, 'markdown') or not parse_response.markdown:
                raise ValueError(
                    f"No markdown content returned from LandingAI for {file_path.name}"
                )

            markdown_content = parse_response.markdown

            logger.info(
                f"Successfully parsed {file_path.name}: "
                f"{len(markdown_content)} characters of markdown"
            )

            return markdown_content

        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path.name}: {e}")
            raise

    def parse_pdf_with_retry(
        self,
        file_path: Path,
        model: str = "dpt-2-latest",
        max_retries: int = 3
    ) -> str:
        """
        Parse PDF with retry logic for transient failures.

        Args:
            file_path: Path to the PDF file
            model: LandingAI model to use
            max_retries: Maximum number of retry attempts

        Returns:
            str: Markdown content

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Parse attempt {attempt}/{max_retries} for {file_path.name}")
                return self.parse_pdf(file_path, model=model)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Parse attempt {attempt}/{max_retries} failed for {file_path.name}: {e}"
                )

                if attempt == max_retries:
                    logger.error(
                        f"All {max_retries} parse attempts failed for {file_path.name}"
                    )
                    raise last_error

                # Wait before retry (exponential backoff could be added here)
                import time
                time.sleep(2 * attempt)

        # Should never reach here, but for type safety
        raise last_error


# Global PDF parser instance
_pdf_parser: Optional[PDFParser] = None


def get_pdf_parser() -> PDFParser:
    """
    Get the global PDF parser instance (singleton pattern).

    Returns:
        PDFParser: The PDF parser instance
    """
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = PDFParser()
    return _pdf_parser
