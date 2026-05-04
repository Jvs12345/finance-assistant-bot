from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.main import app
from src.config import settings


def test_documents_upload_smoke(monkeypatch, tmp_path):
    class FakePG:
        def __init__(self):
            self.docs = {}

        def create_document(self, **kwargs):
            doc = SimpleNamespace(
                id=kwargs["document_id"],
                filename=kwargs["filename"],
                original_filename=kwargs["original_filename"],
                file_path=kwargs["file_path"],
                file_size=kwargs["file_size"],
                category=kwargs.get("category", "other"),
                machine_model=kwargs.get("machine_model"),
                jurisdiction=kwargs.get("jurisdiction"),
                tax_year=kwargs.get("tax_year"),
                client_name=kwargs.get("client_name"),
                entity_type=kwargs.get("entity_type"),
                source_name=kwargs.get("source_name"),
                section_reference=kwargs.get("section_reference"),
                processing_status="uploaded",
                upload_date=None,
                indexed_at=None,
                total_pages=None,
                error_message=None,
            )
            self.docs[doc.id] = doc
            return doc

        def update_document_status(self, document_id, status, error_message=None, total_pages=None, indexed_at=None):
            doc = self.docs[document_id]
            doc.processing_status = status
            doc.error_message = error_message
            doc.total_pages = total_pages
            doc.indexed_at = indexed_at
            return True

    class FakeIndexingService:
        def prepare_pdf_document(self, file_path, document_id, source_filename, category, metadata):
            return {
                "document_id": document_id,
                "source_filename": source_filename,
                "total_pages": 1,
                "total_chunks": 1,
                "documents": [
                    {
                        "id": f"{document_id}-p1-c1",
                        "chunk_id": f"{document_id}-p1-c1",
                        "document_id": document_id,
                        "filename": source_filename,
                        "source_filename": source_filename,
                        "title": f"{source_filename} - Page 1",
                        "content": "sample content",
                        "excerpt": "sample content",
                        "category": category,
                        "file_type": "pdf",
                        "page_number": 1,
                        "chunk_index": 1,
                        "total_chunks": 1,
                        "metadata": metadata,
                    }
                ],
                "metadata": metadata,
            }

        def add_embeddings(self, documents):
            return None

        def index_documents(self, es_client, documents):
            return {"success_count": len(documents), "error_count": 0, "errors": []}

    class FakeESClient:
        index_name = "documents"
        es = SimpleNamespace(indices=SimpleNamespace(refresh=lambda **kwargs: None))

    fake_pg = FakePG()

    monkeypatch.setattr("src.api.documents.get_postgres_client", lambda: fake_pg)
    monkeypatch.setattr("src.api.documents.DocumentIndexingService", lambda: FakeIndexingService())
    monkeypatch.setattr("src.api.documents.get_elasticsearch_client", lambda: FakeESClient())
    monkeypatch.setattr(settings, "pdf_storage_path", str(tmp_path))
    monkeypatch.setattr(settings, "api_key", "test-key")

    client = TestClient(app)
    response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": "Bearer test-key"},
        files={"file": ("sample.pdf", b"%PDF-1.4\n%test\n", "application/pdf")},
        data={"category": "tax_law", "jurisdiction": "Netherlands", "tax_year": "2025"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["number_of_pages"] == 1
    assert payload["number_of_chunks"] == 1
