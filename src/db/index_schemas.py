"""
Elasticsearch index schemas for Document Search & Retrieval System.
"""

from typing import Dict, Any

# Documents index schema
DOCUMENTS_INDEX_SETTINGS = {
    "number_of_shards": 1,  # Single shard for MVP
    "number_of_replicas": 0,  # No replicas for development (1 for production)
    "analysis": {
        "analyzer": {
            "part_number_analyzer": {
                "type": "custom",
                "tokenizer": "keyword",
                "filter": ["lowercase"]
            }
        }
    }
}

DOCUMENTS_INDEX_MAPPINGS = {
    "properties": {
        # Document Identification
        "document_id": {
            "type": "keyword"  # Groups pages from same document
        },
        "filename": {
            "type": "keyword"  # Original filename
        },

        # Content
        "content": {
            "type": "text",
            "analyzer": "standard",
            "fields": {
                "keyword": {  # For exact matching
                    "type": "keyword",
                    "ignore_above": 256
                }
            }
        },
        "summary": {
            "type": "text",
            "analyzer": "standard"
        },

        # Pagination
        "page": {
            "type": "integer"
        },

        # Metadata for Filtering
        "category": {
            "type": "keyword"  # maintenance, operations, spare_parts
        },
        "machine_model": {
            "type": "keyword"  # For filtering by model
        },
        "part_numbers": {
            "type": "keyword",  # Array of part numbers
            "fields": {
                "analyzed": {
                    "type": "text",
                    "analyzer": "part_number_analyzer"
                }
            }
        },

        # Timestamps
        "upload_date": {
            "type": "date"
        },
        "indexed_at": {
            "type": "date"
        },

        # File Info
        "file_size": {
            "type": "long"  # Bytes
        },
        "file_path": {
            "type": "keyword"  # Path to original PDF
        },

        # Processing Status
        "processing_status": {
            "type": "keyword"  # uploaded, parsing, summarizing, indexing, ready, failed
        },
        "error_message": {
            "type": "text"
        }
    }
}


def get_documents_index_schema() -> Dict[str, Any]:
    """
    Get the complete documents index schema.

    Returns:
        dict: Index schema with settings and mappings
    """
    return {
        "settings": DOCUMENTS_INDEX_SETTINGS,
        "mappings": DOCUMENTS_INDEX_MAPPINGS
    }


def create_documents_index(es_client) -> bool:
    """
    Create the documents index with proper schema.

    Args:
        es_client: Elasticsearch client instance

    Returns:
        bool: True if index was created, False if it already exists
    """
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    index_name = "documents"

    try:
        created = es_client.create_index(
            index_name=index_name,
            mappings=DOCUMENTS_INDEX_MAPPINGS,
            settings=DOCUMENTS_INDEX_SETTINGS
        )

        if created:
            logger.info(f"Documents index '{index_name}' created successfully")
        else:
            logger.info(f"Documents index '{index_name}' already exists")

        return created

    except Exception as e:
        logger.error(f"Failed to create documents index: {e}")
        raise
