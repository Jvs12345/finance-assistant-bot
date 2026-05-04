"""
Confluence export parser for processing Confluence space exports.
Handles entities.xml, attachments, and CSV data.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.services.multi_format_processor import get_multi_format_processor
from src.services.recursive_document_scanner import RecursiveDocumentScanner
from src.models.document import DocumentCategory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConfluencePage:
    """Represents a Confluence page."""

    def __init__(self, page_id: str, title: str, content: str, creator: str = None,
                 created_date: str = None, modified_date: str = None):
        self.page_id = page_id
        self.title = title
        self.content = content
        self.creator = creator
        self.created_date = created_date
        self.modified_date = modified_date
        self.attachments: List[Path] = []


class ConfluenceParser:
    """Parser for Confluence space exports."""

    def __init__(self, export_dir: Path):
        """
        Initialize Confluence parser.

        Args:
            export_dir: Path to Confluence export directory
        """
        self.export_dir = Path(export_dir)
        self.entities_file = self.export_dir / "entities.xml"
        self.attachments_dir = self.export_dir / "attachments"
        self.processor = get_multi_format_processor()
        self.pages: Dict[str, ConfluencePage] = {}

    def parse_entities_xml(self) -> List[ConfluencePage]:
        """
        Parse entities.xml file to extract pages and metadata.

        This does a two-pass parsing:
        1. First pass: Parse all BodyContent objects and build ID-to-content mapping
        2. Second pass: Parse Page objects and link them to their body content

        Returns:
            List[ConfluencePage]: List of Confluence pages
        """
        if not self.entities_file.exists():
            logger.error(f"entities.xml not found at {self.entities_file}")
            return []

        logger.info(f"Parsing entities.xml from {self.export_dir}")

        try:
            tree = ET.parse(self.entities_file)
            root = tree.getroot()

            # FIRST PASS: Build BodyContent ID-to-content mapping
            logger.info("First pass: Parsing BodyContent objects...")
            body_content_map = {}

            for obj in root.findall(".//object[@class='BodyContent']"):
                try:
                    body_id_elem = obj.find(".//id[@name='id']")
                    body_text_elem = obj.find(".//property[@name='body']")

                    if body_id_elem is not None and body_text_elem is not None:
                        body_id = body_id_elem.text
                        body_text = body_text_elem.text or ""
                        body_content_map[body_id] = body_text

                except Exception as e:
                    logger.debug(f"Error parsing BodyContent object: {e}")
                    continue

            logger.info(f"Found {len(body_content_map)} BodyContent objects")

            # SECOND PASS: Parse all Page objects and link to body content
            logger.info("Second pass: Parsing Page objects...")
            pages = []

            for obj in root.findall(".//object[@class='Page']"):
                try:
                    page_id_elem = obj.find(".//id[@name='id']")
                    title_elem = obj.find(".//property[@name='title']")

                    if page_id_elem is None or title_elem is None:
                        continue

                    page_id = page_id_elem.text
                    title = title_elem.text

                    # Extract creation date
                    created_date = None
                    created_elem = obj.find(".//property[@name='creationDate']")
                    if created_elem is not None:
                        created_date = created_elem.text

                    # Extract modification date
                    modified_date = None
                    modified_elem = obj.find(".//property[@name='lastModificationDate']")
                    if modified_elem is not None:
                        modified_date = modified_elem.text

                    # Extract body content by looking up BodyContent IDs
                    content = ""
                    body_contents_collection = obj.find(".//collection[@name='bodyContents']")

                    if body_contents_collection is not None:
                        for element in body_contents_collection.findall(".//element[@class='BodyContent']"):
                            body_content_id_elem = element.find(".//id[@name='id']")
                            if body_content_id_elem is not None:
                                body_content_id = body_content_id_elem.text
                                if body_content_id in body_content_map:
                                    content = body_content_map[body_content_id]
                                    break  # Use first body content found

                    page = ConfluencePage(
                        page_id=page_id,
                        title=title,
                        content=content,
                        created_date=created_date,
                        modified_date=modified_date
                    )

                    pages.append(page)
                    self.pages[page_id] = page

                    if content:
                        logger.debug(f"Parsed page with content: {title} (ID: {page_id}, content length: {len(content)})")
                    else:
                        logger.debug(f"Parsed page without content: {title} (ID: {page_id})")

                except Exception as e:
                    logger.warning(f"Error parsing page object: {e}")
                    continue

            logger.info(f"Parsed {len(pages)} pages from entities.xml")

            # Log statistics
            pages_with_content = sum(1 for p in pages if p.content)
            logger.info(f"Pages with body content: {pages_with_content} / {len(pages)}")

            return pages

        except Exception as e:
            logger.error(f"Error parsing entities.xml: {e}")
            return []

    def find_attachments_for_page(self, page_id: str) -> List[Path]:
        """
        Find all attachments for a given page.

        Args:
            page_id: Confluence page ID

        Returns:
            List[Path]: List of attachment file paths
        """
        if not self.attachments_dir.exists():
            return []

        page_attachment_dir = self.attachments_dir / page_id
        if not page_attachment_dir.exists():
            return []

        attachments = []

        # Recursively find all files
        for file_path in page_attachment_dir.rglob("*"):
            if file_path.is_file():
                attachments.append(file_path)

        return attachments

    def process_all_pages(self) -> List[Dict[str, Any]]:
        """
        Process all pages and their attachments.

        Returns:
            List[Dict]: List of processed documents
        """
        pages = self.parse_entities_xml()

        if not pages:
            logger.warning("No pages found in Confluence export")
            return []

        all_documents = []

        for page in pages:
            logger.info(f"Processing page: {page.title}")

            # Create document from page content
            page_doc = {
                "document_id": f"confluence-page-{page.page_id}",
                "filename": f"{page.title}.txt",
                "file_type": "confluence_page",
                "title": page.title,
                "content": f"Title: {page.title}\n\n{page.content}",
                "category": DocumentCategory.DOCUMENTATION.value,
                "metadata": {
                    "source": "confluence",
                    "page_id": page.page_id,
                    "created_date": page.created_date,
                    "modified_date": page.modified_date
                }
            }

            # Generate summary for page
            if page.content:
                try:
                    summary = self.processor.generate_summary(
                        page_doc["content"],
                        page.title
                    )
                    page_doc["summary"] = summary
                except Exception as e:
                    logger.warning(f"Failed to generate summary for {page.title}: {e}")
                    page_doc["summary"] = f"{page.title}\n{page.content[:200]}"

            all_documents.append(page_doc)

            # Process attachments
            attachments = self.find_attachments_for_page(page.page_id)

            for i, attachment_path in enumerate(attachments, 1):
                try:
                    logger.info(f"  Processing attachment {i}/{len(attachments)}: {attachment_path.name}")

                    attachment_doc = self.processor.process_document(
                        file_path=attachment_path,
                        document_id=f"confluence-attachment-{page.page_id}-{i}",
                        category=DocumentCategory.ATTACHMENT,
                        metadata={
                            "source": "confluence",
                            "parent_page_id": page.page_id,
                            "parent_page_title": page.title
                        }
                    )

                    all_documents.append(attachment_doc)

                except Exception as e:
                    logger.error(f"Error processing attachment {attachment_path}: {e}")
                    continue

        logger.info(f"Processed {len(all_documents)} total documents from Confluence export")
        return all_documents

    def parse_csv_files(self) -> List[Dict[str, Any]]:
        """
        Parse all CSV files in the export directory.

        Returns:
            List[Dict]: List of processed CSV documents
        """
        csv_files = list(self.export_dir.glob("*.csv"))

        if not csv_files:
            logger.info("No CSV files found in export directory")
            return []

        documents = []

        for csv_file in csv_files:
            try:
                logger.info(f"Processing CSV file: {csv_file.name}")

                doc = self.processor.process_document(
                    file_path=csv_file,
                    document_id=f"confluence-csv-{csv_file.stem}",
                    category=DocumentCategory.DATA,
                    metadata={
                        "source": "confluence",
                        "file_type": "csv"
                    }
                )

                documents.append(doc)

            except Exception as e:
                logger.error(f"Error processing CSV {csv_file}: {e}")
                continue

        logger.info(f"Processed {len(documents)} CSV files")
        return documents

    def process_all_files_recursive(self) -> List[Dict[str, Any]]:
        """
        Recursively process ALL files in the export directory.
        This ensures no file is missed, regardless of whether it's part of
        the Confluence export structure.

        Returns:
            List[Dict]: List of processed documents from all files
        """
        logger.info(f"Starting recursive file scan of {self.export_dir}")

        scanner = RecursiveDocumentScanner(self.export_dir)
        documents = scanner.scan_and_process_all(recursive=True)

        logger.info(f"Recursive scan found {len(documents)} documents")
        return documents


def parse_confluence_export(export_dir: Path, use_recursive_scan: bool = False) -> List[Dict[str, Any]]:
    """
    Parse a complete Confluence export directory.

    Args:
        export_dir: Path to Confluence export directory
        use_recursive_scan: If True, recursively scan all files instead of using Confluence structure

    Returns:
        List[Dict]: All processed documents
    """
    parser = ConfluenceParser(export_dir)

    all_documents = []

    if use_recursive_scan:
        # Use recursive scanner to process ALL files in ALL folders
        logger.info("Using recursive scan mode - processing all files in all subdirectories")
        all_documents.extend(parser.process_all_files_recursive())
    else:
        # Use traditional Confluence export structure
        logger.info("Using Confluence export structure")

        # Parse pages and attachments
        all_documents.extend(parser.process_all_pages())

        # Parse CSV files
        all_documents.extend(parser.parse_csv_files())

    return all_documents
