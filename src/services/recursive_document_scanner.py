"""
Recursive document scanner that processes all files in all folders.
Handles any file type and ensures complete indexing of document repositories.
"""

from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime

from pypdf import PdfReader

from src.services.multi_format_processor import get_multi_format_processor
from src.models.document import DocumentCategory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RecursiveDocumentScanner:
    """Scan and process all documents in a directory tree."""

    # File extensions to skip (binary files that can't be meaningfully processed)
    SKIP_EXTENSIONS = {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        '.wav', '.flac', '.ogg'
    }

    # Directory names to skip
    SKIP_DIRECTORIES = {
        '__pycache__', '.git', '.svn', '.hg', 'node_modules',
        '.venv', 'venv', 'env', '.env', '.idea', '.vscode',
        'build', 'dist', 'target'
    }

    def __init__(self, base_directory: Path):
        """
        Initialize the scanner.

        Args:
            base_directory: Root directory to scan
        """
        self.base_directory = Path(base_directory)
        self.processor = get_multi_format_processor()
        self.processed_files: Set[str] = set()
        self.skipped_files: Set[str] = set()
        self.error_files: Set[str] = set()

    def should_skip_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be skipped.

        Args:
            file_path: Path to file

        Returns:
            bool: True if file should be skipped
        """
        # Skip by extension
        if file_path.suffix.lower() in self.SKIP_EXTENSIONS:
            return True

        # Skip hidden files
        if file_path.name.startswith('.'):
            return True

        # Skip very large files to avoid memory issues
        # PDFs can be larger (up to 1GB) since we process them page-by-page
        # Other files limited to 100MB
        try:
            file_size = file_path.stat().st_size
            if file_path.suffix.lower() == '.pdf':
                # Allow PDFs up to 1GB (1024MB)
                if file_size > 1024 * 1024 * 1024:
                    logger.warning(f"Skipping very large PDF (>1GB): {file_path.name}")
                    return True
            else:
                # Other files limited to 100MB
                if file_size > 100 * 1024 * 1024:
                    logger.warning(f"Skipping large file (>100MB): {file_path.name}")
                    return True
        except Exception:
            pass

        return False

    def should_skip_directory(self, dir_path: Path) -> bool:
        """
        Determine if a directory should be skipped.

        Args:
            dir_path: Path to directory

        Returns:
            bool: True if directory should be skipped
        """
        dir_name = dir_path.name.lower()

        # Skip common development/system directories
        if dir_name in self.SKIP_DIRECTORIES:
            return True

        # Skip hidden directories
        if dir_name.startswith('.'):
            return True

        return False

    def scan_directory(self, directory: Path, recursive: bool = True) -> List[Path]:
        """
        Scan a directory for all processable files.

        Args:
            directory: Directory to scan
            recursive: If True, scan subdirectories recursively

        Returns:
            List[Path]: List of file paths to process
        """
        files_to_process = []

        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return files_to_process

        if not directory.is_dir():
            logger.error(f"Path is not a directory: {directory}")
            return files_to_process

        logger.info(f"Scanning directory: {directory}")

        try:
            if recursive:
                # Recursively walk through all subdirectories
                for item in directory.rglob("*"):
                    if item.is_file():
                        # Check if parent directory should be skipped
                        skip_dir = False
                        for parent in item.parents:
                            if parent == directory:
                                break
                            if self.should_skip_directory(parent):
                                skip_dir = True
                                break

                        if skip_dir:
                            continue

                        if not self.should_skip_file(item):
                            files_to_process.append(item)
                        else:
                            self.skipped_files.add(str(item.relative_to(self.base_directory)))
            else:
                # Only process files in the current directory
                for item in directory.iterdir():
                    if item.is_file() and not self.should_skip_file(item):
                        files_to_process.append(item)
                    elif item.is_file():
                        self.skipped_files.add(str(item.relative_to(self.base_directory)))

            logger.info(f"Found {len(files_to_process)} files to process in {directory}")

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")

        return files_to_process

    def categorize_file(self, file_path: Path) -> DocumentCategory:
        """
        Determine the document category based on file type and location.

        Args:
            file_path: Path to file

        Returns:
            DocumentCategory: Category for the document
        """
        file_type = self.processor.detect_file_type(file_path)

        if file_type == 'pdf':
            return DocumentCategory.DOCUMENTATION
        elif file_type == 'csv':
            return DocumentCategory.DATA
        elif file_type in ['png', 'jpg', 'jpeg', 'gif', 'image']:
            return DocumentCategory.ATTACHMENT
        elif file_type in ['html', 'xml']:
            return DocumentCategory.DOCUMENTATION
        else:
            return DocumentCategory.OTHER

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single file and extract its content.

        Args:
            file_path: Path to file

        Returns:
            dict: Processed document data
        """
        try:
            logger.info(f"Processing: {file_path.relative_to(self.base_directory)}")

            # Generate document ID based on file path
            relative_path = file_path.relative_to(self.base_directory)
            path_str = str(relative_path).replace('\\', '/').replace('/', '-')
            document_id = f"file-{path_str}"

            # Determine category
            category = self.categorize_file(file_path)

            # Build metadata
            metadata = {
                "source": "file_system",
                "relative_path": str(relative_path),
                "absolute_path": str(file_path),
                "directory": str(file_path.parent.relative_to(self.base_directory)),
                "file_size": file_path.stat().st_size,
                "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }

            # PDFs: split per page for better navigation (page_number populated)
            if file_path.suffix.lower() == ".pdf":
                results = self.process_pdf_file(
                    file_path=file_path,
                    base_document_id=document_id,
                    category=category,
                    metadata=metadata
                )
            else:
                # Process the document as a single item
                single_result = self.processor.process_document(
                    file_path=file_path,
                    document_id=document_id,
                    category=category,
                    metadata=metadata
                )

                # Add title from filename if not set
                if "title" not in single_result or not single_result["title"]:
                    single_result["title"] = file_path.stem

                # Add relative path for reference
                single_result["relative_path"] = str(relative_path)
                results = [single_result]

            self.processed_files.add(str(relative_path))
            return results

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.error_files.add(str(file_path.relative_to(self.base_directory)))
            return None

    def process_pdf_file(
        self,
        file_path: Path,
        base_document_id: str,
        category: DocumentCategory,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split a PDF into per-page documents with page_number populated.
        """
        documents: List[Dict[str, Any]] = []
        try:
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            return documents

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Failed to extract text from {file_path.name} page {page_num}: {e}")
                text = ""

            doc = {
                "document_id": f"{base_document_id}-p{page_num:04d}",
                "filename": file_path.name,
                "file_type": "pdf",
                "title": f"{file_path.stem} - Page {page_num}",
                "content": text,
                "summary": text[:500] + "..." if len(text) > 500 else text,
                "category": category.value if hasattr(category, "value") else str(category),
                "page_number": page_num,
                "metadata": {
                    **metadata,
                    "page_number": page_num,
                    "page_count": total_pages,
                },
                "relative_path": metadata.get("relative_path", str(file_path.name)),
            }
            documents.append(doc)

        logger.info(f"Split PDF into {len(documents)} pages: {file_path.name}")
        return documents

    def scan_and_process_all(self, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Scan and process all files in the base directory.

        Args:
            recursive: If True, scan subdirectories recursively

        Returns:
            List[Dict]: List of processed documents
        """
        logger.info("=" * 70)
        logger.info(f"Starting recursive document scan of: {self.base_directory}")
        logger.info("=" * 70)

        start_time = datetime.now()

        # Scan for all files
        files_to_process = self.scan_directory(self.base_directory, recursive=recursive)

        if not files_to_process:
            logger.warning("No files found to process")
            return []

        logger.info(f"Total files to process: {len(files_to_process)}")
        logger.info("")

        # Process all files
        documents: List[Dict[str, Any]] = []

        for i, file_path in enumerate(files_to_process, 1):
            logger.info(f"[{i}/{len(files_to_process)}] Processing: {file_path.name}")

            result = self.process_file(file_path)

            if isinstance(result, list):
                documents.extend(result)
            elif result:
                documents.append(result)

            # Progress update every 10 files
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(files_to_process)} files processed")

        # Summary
        elapsed = datetime.now() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed.total_seconds():.1f} seconds")
        logger.info(f"Successfully processed: {len(self.processed_files)} files")
        logger.info(f"Skipped: {len(self.skipped_files)} files")
        logger.info(f"Errors: {len(self.error_files)} files")
        logger.info(f"Documents ready for indexing: {len(documents)}")
        logger.info("")

        return documents

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            dict: Statistics about the scan
        """
        return {
            "base_directory": str(self.base_directory),
            "processed_count": len(self.processed_files),
            "skipped_count": len(self.skipped_files),
            "error_count": len(self.error_files),
            "processed_files": sorted(list(self.processed_files)),
            "skipped_files": sorted(list(self.skipped_files)),
            "error_files": sorted(list(self.error_files))
        }


def scan_and_process_directory(directory: Path, recursive: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to scan and process a directory.

    Args:
        directory: Directory to scan
        recursive: If True, scan subdirectories recursively

    Returns:
        List[Dict]: List of processed documents
    """
    scanner = RecursiveDocumentScanner(directory)
    return scanner.scan_and_process_all(recursive=recursive)
