# Financial Document Assistant

## Short description
This is a local financial document assistant. It answers questions about uploaded tax, accounting, and finance documents. It searches the documents first and answers with source references.

## Why I built it
I first built a Be Informed documentation chatbot. I reused the same local RAG setup to test whether it also works for accounting and tax documents. I wanted to keep everything local and make answers traceable to source text.

## What it can do
- Index PDFs from `Source_files/` in batch mode.
- Upload PDFs through the API and index them in the same Elasticsearch index.
- Ask questions through the web UI and API.
- Filter retrieval by document type, jurisdiction, and tax year.
- Show source snippets and open source PDFs from the answer.
- Run local calculations when the answer includes a calculation payload.

## How it works
1. PDF text is extracted and split into chunks.
2. Chunks are embedded with a local embedding model.
3. Chunks are stored in Elasticsearch with metadata.
4. A question triggers retrieval first.
5. Retrieved chunks are passed to the local LLM (Ollama).
6. The answer is returned with cited source documents.

## Tech stack
- Python + FastAPI
- Elasticsearch (vector + keyword retrieval)
- PostgreSQL (document metadata)
- Ollama (local LLM)
- Sentence Transformers (local embeddings)
- Simple static HTML/CSS/JS frontend
- Docker Compose

## How to run it locally
1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy env template:

```powershell
copy .env.example .env
```

3. Start Docker services:

```powershell
docker compose -p financial-bot up -d
```

4. Start the app:

```powershell
RUN_FINANCIAL_ASSISTANT.bat
```

5. Open UI:

- `http://localhost:8100/`

## How to add documents
### Option A: Batch indexing from `Source_files/`
- Put PDF files in `Source_files/`
- Run:

```powershell
INDEX_PDFS.bat
```

### Option B: API upload

```powershell
curl -X POST "http://localhost:8100/api/v1/documents/upload" `
  -H "Authorization: Bearer your_api_key" `
  -F "file=@Source_files\sample.pdf" `
  -F "category=tax_law" `
  -F "jurisdiction=Netherlands" `
  -F "tax_year=2025"
```

Then ask questions at:
- `POST /api/v1/llama/ask`

## Example questions
- Which expenses in these files look deductible?
- What tax obligations are mentioned for this client?
- Where do these documents mention VAT rules?
- Compare this income statement with the tax requirements in the uploaded docs.
- What information is still missing before drafting a tax position?
- Do these files contain inconsistent numbers?
- Explain this tax rule in simple language and quote the source.

## Limitations
- This is a proof of concept.
- It does not replace an accountant, tax advisor, or lawyer.
- It cannot guarantee correct tax advice.
- Output quality depends on document quality and OCR.
- Scanned PDFs can fail if text extraction is poor.
- RAG reduces hallucination risk but does not remove it completely.
- Human review is still required for important decisions.

## Disclaimer
This tool helps with document analysis. Check important tax and accounting decisions with a qualified professional.

## Project structure
```text
src/
  api/            FastAPI routes
  db/             Elasticsearch and PostgreSQL clients
  models/         Pydantic models
  services/       Retrieval, indexing, LLM, calculation sandbox
  utils/          Auth and logging helpers
scripts/          Local indexing and utility scripts
static/           UI and PDF highlight viewer
Source_files/     Local PDF input folder (ignored in git)
uploads/          Temporary upload folder (ignored in git)
```

## Manual test checklist
1. Start services and open `http://localhost:8100/`.
2. Put at least one PDF in `Source_files/` and run `INDEX_PDFS.bat`.
3. Ask a question in the UI and verify source cards appear.
4. Upload a PDF through `/api/v1/documents/upload` and query it.
5. Delete an uploaded document with `DELETE /api/v1/documents/{document_id}` and re-query.
