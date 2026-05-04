#!/usr/bin/env python3
"""
Optimized PDF indexer with Elasticsearch performance tuning.
Based on Elasticsearch bulk indexing best practices.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import multiprocessing
import time

from elasticsearch import Elasticsearch, helpers
from src.services.document_indexing_service import DocumentIndexingService
from src.services.embedding_service import EmbeddingService, EmbeddingProvider
from src.utils.logging import setup_logging, get_logger
from src.config import settings

setup_logging(log_level=settings.log_level)
logger = get_logger(__name__)


class OptimizedElasticsearchIndexer:
    """Optimized Elasticsearch indexer with performance tuning."""

    def __init__(self, es_url: str, index_name: str):
        """Initialize with optimized settings."""
        
        # Try primary URL first
        try:
            self.es = Elasticsearch(
                [es_url],
                request_timeout=120,
                max_retries=5,
                retry_on_timeout=True
            )
            if not self.es.ping():
                raise ConnectionError("Ping failed")
        except Exception:
            # Fallback to host-mapped port used by this stack
            alt_url = "http://localhost:39200"
            logger.warning(f"Could not connect to {es_url}, trying {alt_url}...")
            self.es = Elasticsearch(
                [alt_url],
                request_timeout=120,
                max_retries=5,
                retry_on_timeout=True
            )
            
        self.index_name = index_name
        
        # Initialize embedding service (local by default for reindexing)
        logger.info("Initializing embedding service...")
        self.embedding_service = EmbeddingService(
            provider=EmbeddingProvider.LOCAL,
            model="all-MiniLM-L6-v2"
        )
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

    def optimize_for_bulk_indexing(self):
        """
        Apply Elasticsearch optimizations for bulk indexing.
        Based on: https://stackoverflow.com/questions/48590502/
        """
        try:
            # 1. Disable refresh during bulk indexing (HUGE performance gain)
            self.es.indices.put_settings(
                index=self.index_name,
                body={"index": {"refresh_interval": "-1"}}
            )
            logger.info("✓ Disabled index refresh (will re-enable after indexing)")

            # 2. Set replicas to 0 during indexing (faster writes)
            self.es.indices.put_settings(
                index=self.index_name,
                body={"index": {"number_of_replicas": 0}}
            )
            logger.info("[OK] Set replicas to 0 (faster indexing)")

            # 3. Increase bulk thread pool
            # Note: This requires cluster settings permission
            try:
                self.es.cluster.put_settings(
                    body={
                        "transient": {
                            "thread_pool.write.queue_size": 1000
                        }
                    }
                )
                logger.info("✓ Increased write thread pool size")
            except Exception as e:
                logger.warning(f"Could not modify cluster settings (may not have permission): {e}")

        except Exception as e:
            logger.warning(f"Some optimizations failed (continuing anyway): {e}")

    def restore_normal_settings(self):
        """Restore normal settings after bulk indexing."""
        try:
            # Re-enable refresh (30s is default)
            self.es.indices.put_settings(
                index=self.index_name,
                body={"index": {"refresh_interval": "30s"}}
            )
            logger.info("[OK] Re-enabled index refresh")

            # Force a refresh to make all documents searchable
            self.es.indices.refresh(index=self.index_name)
            logger.info("[OK] Forced index refresh")

            # Optionally restore replicas if needed
            # self.es.indices.put_settings(
            #     index=self.index_name,
            #     body={"index": {"number_of_replicas": 1}}
            # )

        except Exception as e:
            logger.error(f"Error restoring settings: {e}")

    def bulk_index_optimized(self, documents: List[Dict[str, Any]], chunk_size: int = 500):
        """
        Bulk index with optimal chunk size and settings.

        Args:
            documents: List of documents to index
            chunk_size: Number of docs per bulk request (default 500, optimal for most cases)

        Returns:
            dict: Success and error counts
        """
        actions = []
        for doc in documents:
            doc_id = doc.get("id") or doc.get("document_id") or f"doc-{hash(str(doc))}"

            action = {
                "_index": self.index_name,
                "_id": doc_id,
                "_source": {
                    "document_id": doc.get("document_id", doc_id),
                    "filename": doc.get("file_name", doc.get("filename", "Unknown")),
                    "source_filename": doc.get("source_filename", doc.get("file_name", doc.get("filename", "Unknown"))),
                    "title": doc.get("title", doc.get("file_name", "Unknown")),
                    "content": doc.get("content", "")[:1000000],  # Limit to 1MB of text
                    "excerpt": doc.get("excerpt", ""),
                    "summary": doc.get("summary"),
                    "category": doc.get("category", "document"),
                    "file_type": doc.get("file_type", "pdf"),
                    "chunk_id": doc.get("chunk_id", doc_id),
                    "jurisdiction": doc.get("jurisdiction"),
                    "tax_year": doc.get("tax_year"),
                    "client_name": doc.get("client_name"),
                    "entity_type": doc.get("entity_type"),
                    "source_name": doc.get("source_name"),
                    "section_reference": doc.get("section_reference"),
                    "file_path": doc.get("file_path", ""),
                    "file_size": doc.get("file_size", 0),
                    "page_number": doc.get("page_number"),
                    "chunk_index": doc.get("chunk_index"),
                    "total_chunks": doc.get("total_chunks"),
                    "upload_date": doc.get("upload_date"),
                    "metadata": doc.get("metadata", {}),
                    "indexed_at": datetime.utcnow().isoformat()
                }
            }
            
            # Add embedding if available
            if "embedding" in doc:
                action["_source"]["embedding"] = doc["embedding"]
                
            actions.append(action)

        # Use helpers.bulk with optimized settings
        try:
            success, errors = helpers.bulk(
                self.es,
                actions,
                chunk_size=chunk_size,  # Optimal chunk size
                request_timeout=120,
                raise_on_error=False,
                raise_on_exception=False,
                max_retries=3,
                initial_backoff=2,
                max_backoff=60
            )

            error_count = len(errors) if isinstance(errors, list) else 0
            return {
                "success_count": success,
                "error_count": error_count,
                "errors": errors if error_count > 0 else []
            }

        except Exception as e:
            logger.error(f"Bulk indexing error: {e}")
            return {
                "success_count": 0,
                "error_count": len(documents),
                "errors": [str(e)]
            }


def process_pdf_parallel(
    pdf_path: Path,
    index: int,
    total: int,
    text_chunk_size: int,
    no_chunks: bool,
) -> List[Dict[str, Any]]:
    """Process a single PDF page-by-page (parallel-safe)."""
    try:
        print(f"[{index}/{total}] Processing: {pdf_path.name} ({pdf_path.stat().st_size / (1024*1024):.1f} MB)")
        start = time.time()

        chunk_size = text_chunk_size if not no_chunks else 10_000_000
        indexing_service = DocumentIndexingService(chunk_size=chunk_size, overlap=500)
        prepared = indexing_service.prepare_pdf_document(
            file_path=pdf_path,
            document_id=f"pdf-{pdf_path.stem}",
            source_filename=pdf_path.name,
            category="other",
            metadata={"source_name": pdf_path.name},
        )
        elapsed = time.time() - start

        for doc in prepared["documents"]:
            doc["processing_time_seconds"] = elapsed

        print(
            f"[{index}/{total}] [OK] {pdf_path.name} "
            f"({prepared['total_pages']} pages, {prepared['total_chunks']} chunks, {elapsed:.1f}s)"
        )
        return prepared["documents"]

    except Exception as e:
        print(f"[{index}/{total}] [ERR] Error: {pdf_path.name} - {e}")
        logger.error(f"Error processing {pdf_path.name}: {e}")
        return []


def main():
    """Main optimized indexing function."""
    parser = argparse.ArgumentParser(description='Optimized PDF reindexing')
    parser.add_argument('--workers', '-w', type=int, default=None)
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Elasticsearch bulk chunk size (default: 500)')
    parser.add_argument('--text-chunk-size', type=int, default=10000,
                       help='Text chunk size for large documents (default: 10000)')
    parser.add_argument('--no-chunks', action='store_true',
                       help='Disable text chunking')
    parser.add_argument('--pdf-dir', default='Source_files',
                       help='Directory containing PDF files (default: Source_files)')
    parser.add_argument('--yes', '-y', action='store_true')
    args = parser.parse_args()

    print("=" * 80)
    print("  OPTIMIZED PDF REINDEXING")
    print("  Elasticsearch Performance Tuning Enabled")
    print("=" * 80)
    print()

    # System info
    cpu_count = multiprocessing.cpu_count()
    workers = args.workers or max(2, cpu_count - 1)
    print(f"CPUs: {cpu_count} | Workers: {workers}")
    print(f"Bulk chunk size: {args.chunk_size} | Text chunk size: {args.text_chunk_size}")
    print()

    # Find PDFs
    source_dir = Path(args.pdf_dir)
    pdf_files = sorted(source_dir.glob("*.pdf"))

    if not pdf_files:
        print("ERROR: No PDF files found")
        return 1

    total_size = sum(f.stat().st_size for f in pdf_files)
    print(f"Found {len(pdf_files)} PDFs ({total_size / (1024**3):.2f} GB)")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf.name} ({pdf.stat().st_size / (1024*1024):.1f} MB)")
    print()

    # Estimate time
    est_time = (total_size / (1024 * 1024) * 2) / workers / 60
    print(f"Estimated time: {est_time:.1f} minutes")
    print()

    if not args.yes:
        response = input("Continue? (Y/N): ").strip().lower()
        if response not in ['yes', 'y']:
            return 0
    print()

    start_time = datetime.now()

    # STAGE 1: Process PDFs in parallel
    print("-" * 80)
    print("STAGE 1: Parallel PDF Processing")
    print("-" * 80)
    print()

    documents = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_pdf_parallel,
                pdf,
                i,
                len(pdf_files),
                args.text_chunk_size,
                args.no_chunks,
            ): pdf
            for i, pdf in enumerate(pdf_files, 1)
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                documents.extend(result)

    print()
    print(f"[OK] Processed {len(documents)}/{len(pdf_files)} files")
    print()

    if not documents:
        print("ERROR: No documents processed successfully")
        return 1

    # STAGE 2: Optimized Elasticsearch indexing
    print("-" * 80)
    print("STAGE 2: Elasticsearch Indexing (Optimized)")
    print("-" * 80)
    print()

    try:
        indexer = OptimizedElasticsearchIndexer(
            settings.elasticsearch_url,
            settings.elasticsearch_index
        )

        # Clear existing index
        try:
            indexer.es.indices.delete(index=indexer.index_name, ignore=[404])
            print("[OK] Cleared old index")
        except Exception as e:
            logger.warning(f"Could not clear index: {e}")

        # Recreate index with optimal settings
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",  # Disable during bulk indexing
                "index.translog.durability": "async",  # Faster writes
                "index.translog.sync_interval": "30s"
            },
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "filename": {"type": "text"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "excerpt": {"type": "text"},
                    "summary": {"type": "text"},
                    "category": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "source_filename": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "jurisdiction": {"type": "keyword"},
                    "tax_year": {"type": "integer"},
                    "client_name": {"type": "keyword"},
                    "entity_type": {"type": "keyword"},
                    "source_name": {"type": "keyword"},
                    "section_reference": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "file_size": {"type": "long"},
                    "page_number": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "upload_date": {"type": "date"},
                    "metadata": {"type": "object", "enabled": False},
                    "indexed_at": {"type": "date"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        indexer.es.indices.create(index=indexer.index_name, body=index_settings)
        print("[OK] Created optimized index")
        print()

        # Apply additional optimizations
        print("Applying Elasticsearch optimizations...")
        indexer.optimize_for_bulk_indexing()
        print()

        # Bulk index documents
        print(f"Indexing {len(documents)} documents...")
        
        # Generate embeddings in batches
        print("Generating embeddings (this may take a while)...")
        batch_size = 32
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_docs = documents[i:batch_end]
            
            # Extract texts for embedding (use content if short, else excerpt + title)
            batch_texts = []
            for doc in batch_docs:
                text = doc.get("content", "")
                if len(text) > 1000:
                    text = doc.get("title", "") + ": " + text[:1000]
                batch_texts.append(text)
                
            try:
                embeddings = indexer.embedding_service.get_embeddings_batch(batch_texts)
                
                # Assign to documents
                for j, doc in enumerate(batch_docs):
                    doc["embedding"] = embeddings[j]
                    
                print(f"  Embeddings: {batch_end}/{total_docs}", end="\r")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                
        print()
        
        result = indexer.bulk_index_optimized(documents, chunk_size=args.chunk_size)

        success_count = result["success_count"]
        error_count = result["error_count"]

        print()
        print(f"[OK] Indexed: {success_count} documents")
        if error_count > 0:
            print(f"[ERR] Errors: {error_count}")
        print()

        # Restore normal settings
        print("Restoring normal Elasticsearch settings...")
        indexer.restore_normal_settings()
        print("✓ Index ready for searching")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Indexing error: {e}")
        return 1

    # Final stats
    elapsed = datetime.now() - start_time
    print("=" * 80)
    print("  INDEXING COMPLETE")
    print("=" * 80)
    print()
    print(f"Total time: {elapsed.total_seconds():.1f}s ({elapsed.total_seconds() / 60:.2f} min)")
    print(f"Files: {len(pdf_files)} | Documents: {success_count}")
    print(f"Avg: {elapsed.total_seconds() / len(pdf_files):.1f}s per file")
    print()
    print("Next: Run RUN_FINANCIAL_ASSISTANT.bat to start searching!")
    print()

    # Save stats to log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "files_processed": len(pdf_files),
        "documents_indexed": success_count,
        "errors": error_count,
        "processing_time_seconds": elapsed.total_seconds(),
        "settings": {
            "workers": workers,
            "bulk_chunk_size": args.chunk_size,
            "text_chunk_size": args.text_chunk_size
        }
    }

    with open("reindex.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Reindex completed: {datetime.now()}\n")
        f.write(f"Files: {len(pdf_files)} | Documents: {success_count} | Time: {elapsed.total_seconds():.1f}s\n")
        f.write(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
