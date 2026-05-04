#!/usr/bin/env python3
"""
Load sample PDF documents for testing and demonstration.

This script uploads PDFs from the test-files/ directory to the Document Search system.
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8100")
API_KEY = os.getenv("API_KEY")
SAMPLE_FILES_DIR = project_root / "test-files"

# Sample document metadata
SAMPLE_DOCUMENTS = [
    {
        "filename": "agv-opwekken.pdf",
        "category": "other",
        "machine_model": "AGV-OPWEKKEN",
    },
    {
        "filename": "manual.pdf",
        "category": "other",
        "machine_model": "GENERIC",
    },
]


class SampleDataLoader:
    """Load sample PDF documents into the system."""

    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the sample data loader.

        Args:
            api_url: Base URL of the API
            api_key: API key for authentication
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def check_health(self) -> bool:
        """
        Check if the API is healthy.

        Returns:
            bool: True if API is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"❌ API health check failed: {e}")
            return False

    def upload_document(
        self, file_path: Path, category: str, machine_model: str
    ) -> Dict[str, Any]:
        """
        Upload a document to the API.

        Args:
            file_path: Path to PDF file
            category: Document category
            machine_model: Machine model identifier

        Returns:
            dict: Response from API with document_id and status
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"📤 Uploading {file_path.name}...")

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            data = {"category": category, "machine_model": machine_model}

            response = requests.post(
                f"{self.api_url}/api/v1/documents/upload",
                headers=self.headers,
                files=files,
                data=data,
            )

        if response.status_code in (200, 201):
            result = response.json()
            print(f"✅ Uploaded: {result['document_id']}")
            return result
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            raise Exception(f"Upload failed: {response.text}")

    def wait_for_processing(self, document_id: str, timeout: int = 300) -> bool:
        """
        Wait for document to finish processing.

        Args:
            document_id: Document ID to check
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if processing succeeded, False if failed or timed out
        """
        print(f"⏳ Waiting for document {document_id} to process...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.api_url}/api/v1/documents/{document_id}",
                    headers=self.headers,
                )

                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data["status"]

                    if status == "ready":
                        total_pages = status_data.get("total_pages", "unknown")
                        print(f"✅ Processing complete ({total_pages} pages)")
                        if status_data.get("error_message"):
                            print(f"ℹ️  Note: {status_data['error_message']}")
                        return True
                    elif status == "failed":
                        error_msg = status_data.get("error_message", "Unknown error")
                        print(f"❌ Processing failed: {error_msg}")
                        return False
                    else:
                        # Still processing
                        print(f"   Status: {status}...", end="\r")
                        time.sleep(5)
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    return False

            except requests.RequestException as e:
                print(f"❌ Error checking status: {e}")
                return False

        print(f"⏰ Timeout waiting for processing")
        return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the system.

        Returns:
            list: List of document metadata
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/documents?page=1&page_size=100",
                headers=self.headers,
            )

            if response.status_code == 200:
                return response.json()["documents"]
            else:
                print(f"❌ Failed to list documents: {response.status_code}")
                return []

        except requests.RequestException as e:
            print(f"❌ Error listing documents: {e}")
            return []


def main():
    """Main function to load sample data."""
    print("📦 Document Search & Retrieval System - Sample Data Loader")
    print("=" * 60)
    print()

    # Check configuration
    if not API_KEY:
        print("❌ Error: API_KEY not found in .env file")
        print("Please set API_KEY in .env and try again")
        sys.exit(1)

    # Check if sample files directory exists
    if not SAMPLE_FILES_DIR.exists():
        print(f"❌ Error: Sample files directory not found: {SAMPLE_FILES_DIR}")
        print("Please create test-files/ directory and add sample PDFs")
        sys.exit(1)

    # Initialize loader
    loader = SampleDataLoader(API_BASE_URL, API_KEY)

    # Check API health
    print("1️⃣  Checking API health...")
    if not loader.check_health():
        print("❌ API is not healthy. Please start the application:")
        print("   ./scripts/run_app.sh")
        sys.exit(1)
    print("✅ API is healthy")
    print()

    # Get existing documents
    print("2️⃣  Checking existing documents...")
    existing_docs = loader.list_documents()
    existing_filenames = {doc["filename"] for doc in existing_docs}
    print(f"ℹ️  Found {len(existing_docs)} existing document(s)")
    print()

    # Upload sample documents
    print("3️⃣  Uploading sample documents...")
    print()

    uploaded_count = 0
    skipped_count = 0
    failed_count = 0

    for doc_info in SAMPLE_DOCUMENTS:
        filename = doc_info["filename"]
        file_path = SAMPLE_FILES_DIR / filename

        # Skip if already exists
        if filename in existing_filenames:
            print(f"⏭️  Skipping {filename} (already exists)")
            skipped_count += 1
            continue

        # Check if file exists
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path.name}")
            continue

        try:
            # Upload document
            result = loader.upload_document(
                file_path=file_path,
                category=doc_info["category"],
                machine_model=doc_info["machine_model"],
            )

            # Wait for processing
            success = loader.wait_for_processing(result["document_id"], timeout=300)

            if success:
                uploaded_count += 1
            else:
                failed_count += 1

            print()

        except Exception as e:
            print(f"❌ Error uploading {filename}: {e}")
            failed_count += 1
            print()

    # Summary
    print("=" * 60)
    print("📊 Summary:")
    print(f"   ✅ Successfully uploaded: {uploaded_count}")
    print(f"   ⏭️  Skipped (existing):   {skipped_count}")
    print(f"   ❌ Failed:               {failed_count}")
    print()

    # List all documents
    print("4️⃣  Final document list:")
    all_docs = loader.list_documents()
    if all_docs:
        print()
        for doc in all_docs:
            status_icon = "✅" if doc["processing_status"] == "ready" else "⏳"
            print(
                f"{status_icon} {doc['filename']:<30} | "
                f"{doc['category']:<15} | "
                f"Status: {doc['processing_status']}"
            )
    print()

    print("✨ Sample data loading complete!")
    print()
    print("🔍 Test search at: http://localhost:8100/")
    print("📚 API docs at:    http://localhost:8100/docs")
    print()


if __name__ == "__main__":
    main()
