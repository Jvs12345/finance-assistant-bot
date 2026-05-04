"""
Semantic indexer for PDF sections.
Builds and maintains a searchable index with automatic summary and issue tracking.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from src.services.pdf_section_extractor import PDFSection, PDFSectionExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticIndexer:
    """
    Indexes PDF sections for semantic search.

    Features:
    - Section-level indexing
    - Automatic summary generation
    - Issue tracking (missing data, OCR failures, etc.)
    - Incremental updates
    """

    def __init__(self, index_dir: Path):
        """
        Initialize the semantic indexer.

        Args:
            index_dir: Directory to store index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_dir / "index.json"
        self.summary_file = self.index_dir / "summary.json"
        self.issues_file = self.index_dir / "issues.json"

        # In-memory index
        self.index: Dict[str, Any] = {
            'documents': [],
            'metadata': {
                'created_at': None,
                'updated_at': None,
                'total_documents': 0,
                'total_sections': 0
            }
        }

        self.summary: Dict[str, Any] = {}
        self.issues: List[Dict[str, Any]] = []

        # Load existing index if available
        self._load_index()

    def _load_index(self) -> None:
        """Load existing index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                logger.info(f"Loaded existing index with {len(self.index['documents'])} documents")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")

        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    self.summary = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load summary: {e}")

        if self.issues_file.exists():
            try:
                with open(self.issues_file, 'r', encoding='utf-8') as f:
                    self.issues = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load issues: {e}")

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved index to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def _save_summary(self) -> None:
        """Save summary to disk."""
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    def _save_issues(self) -> None:
        """Save issues to disk."""
        try:
            with open(self.issues_file, 'w', encoding='utf-8') as f:
                json.dump(self.issues, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save issues: {e}")

    def _log_issue(self, issue_type: str, message: str, details: Optional[Dict] = None) -> None:
        """
        Log an issue during indexing.

        Args:
            issue_type: Type of issue (e.g., 'ocr_failure', 'parse_error')
            message: Issue description
            details: Additional details
        """
        issue = {
            'timestamp': datetime.now().isoformat(),
            'type': issue_type,
            'message': message,
            'details': details or {}
        }

        self.issues.append(issue)
        logger.warning(f"Issue logged: {issue_type} - {message}")

    def _update_summary(self) -> None:
        """Update summary statistics."""
        self.summary = {
            'total_documents': len(set(doc['source'] for doc in self.index['documents'])),
            'total_sections': len(self.index['documents']),
            'section_types': {},
            'sources': {},
            'last_updated': datetime.now().isoformat(),
            'index_path': str(self.index_dir)
        }

        # Count section types
        for doc in self.index['documents']:
            section_type = doc.get('section_type', 'unknown')
            self.summary['section_types'][section_type] = \
                self.summary['section_types'].get(section_type, 0) + 1

            # Count by source
            source = doc.get('source', 'unknown')
            self.summary['sources'][source] = \
                self.summary['sources'].get(source, 0) + 1

    def index_sections(self, sections: List[PDFSection], source_file: str, save_immediately: bool = True) -> None:
        """
        Index PDF sections.

        Args:
            sections: List of PDFSection objects
            source_file: Source PDF file name
            save_immediately: If True, save to disk immediately; if False, defer saving for bulk operations
        """
        logger.info(f"Indexing {len(sections)} sections from {source_file}")

        # Remove existing sections from this source
        self.index['documents'] = [
            doc for doc in self.index['documents']
            if doc.get('source') != source_file
        ]

        # Add new sections
        for section in sections:
            doc = {
                'id': section.id,
                'text': section.text,
                'title': section.title,
                'source': source_file,
                'page_num': section.page_num,
                'section_type': section.section_type,
                'confidence': section.confidence,
                'indexed_at': datetime.now().isoformat()
            }

            self.index['documents'].append(doc)

            # Log issues
            if section.confidence < 0.8:
                self._log_issue(
                    'low_confidence',
                    f"Low OCR confidence on {source_file} page {section.page_num}",
                    {'confidence': section.confidence, 'section_id': section.id}
                )

        # Update metadata
        self.index['metadata']['updated_at'] = datetime.now().isoformat()
        if self.index['metadata']['created_at'] is None:
            self.index['metadata']['created_at'] = datetime.now().isoformat()

        # Save to disk (only if requested)
        if save_immediately:
            self._update_summary()
            self._save_index()
            self._save_summary()
            self._save_issues()

        logger.info(f"Successfully indexed {len(sections)} sections from {source_file}")

    def index_pdf_file(self, pdf_path: Path) -> None:
        """
        Extract and index a single PDF file.

        Args:
            pdf_path: Path to PDF file
        """
        try:
            extractor = PDFSectionExtractor()
            sections = extractor.extract_sections(pdf_path)

            if not sections:
                self._log_issue(
                    'no_sections',
                    f"No sections extracted from {pdf_path.name}",
                    {'file': str(pdf_path)}
                )
                return

            self.index_sections(sections, pdf_path.name)

        except Exception as e:
            self._log_issue(
                'parse_error',
                f"Failed to parse {pdf_path.name}: {str(e)}",
                {'file': str(pdf_path), 'error': str(e)}
            )
            raise

    def index_pdf_directory(self, pdf_dir: Path, pattern: str = "*.pdf", batch_size: int = 10) -> None:
        """
        Index all PDF files in a directory with bulk optimization.

        Args:
            pdf_dir: Directory containing PDF files
            pattern: File pattern to match (default: "*.pdf")
            batch_size: Number of files to process before saving (default: 10)
        """
        pdf_dir = Path(pdf_dir)

        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")

        pdf_files = list(pdf_dir.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return

        # Calculate total size
        total_size_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)

        logger.info(f"Found {len(pdf_files)} PDF files to index (batch size: {batch_size}, total size: {total_size_mb:.1f} MB)")
        print(f"\n[INFO] Processing {len(pdf_files)} PDF files ({total_size_mb:.1f} MB) in batches of {batch_size}...")

        successful = 0
        failed = 0
        import time
        start_time = time.time()

        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
                print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name} ({file_size_mb:.1f} MB)", end="")

                # Extract sections
                extractor = PDFSectionExtractor()
                sections = extractor.extract_sections(pdf_file)

                if not sections:
                    self._log_issue(
                        'no_sections',
                        f"No sections extracted from {pdf_file.name}",
                        {'file': str(pdf_file)}
                    )
                    print(" - [WARN] No sections found")
                    continue

                # Index sections without saving (bulk mode)
                is_last_file = (i == len(pdf_files))
                should_save = (i % batch_size == 0) or is_last_file

                self.index_sections(sections, pdf_file.name, save_immediately=should_save)

                successful += 1
                section_count = len(sections)

                if should_save:
                    print(f" - [OK] {section_count} sections (saved batch)")
                else:
                    print(f" - [OK] {section_count} sections")

            except Exception as e:
                failed += 1
                logger.error(f"Failed to index {pdf_file.name}: {e}")
                print(f" - [ERROR] {e}")
                self._log_issue(
                    'parse_error',
                    f"Failed to parse {pdf_file.name}: {str(e)}",
                    {'file': str(pdf_file), 'error': str(e)}
                )
                continue

        # Final save and summary update
        print(f"\n[INFO] Finalizing index...")
        self._update_summary()
        self._save_index()
        self._save_summary()
        self._save_issues()

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)

        print(f"\n[INFO] Indexing complete!")
        print(f"  Total sections indexed: {len(self.index['documents'])}")
        print(f"  Successful files: {successful}/{len(pdf_files)}")
        print(f"  Failed files: {failed}")
        print(f"  Total time: {minutes}m {seconds}s")
        print(f"  Average: {elapsed_time/len(pdf_files):.1f}s per file")

        logger.info(f"Indexing complete. Total sections: {len(self.index['documents'])}, Success: {successful}, Failed: {failed}, Time: {elapsed_time:.1f}s")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get index summary.

        Returns:
            Summary dictionary
        """
        return self.summary

    def get_issues(self) -> List[Dict[str, Any]]:
        """
        Get logged issues.

        Returns:
            List of issue dictionaries
        """
        return self.issues

    def clear_issues(self) -> None:
        """Clear all logged issues."""
        self.issues = []
        self._save_issues()
        logger.info("Cleared all issues")
