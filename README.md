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
- Run formula-driven calculations from natural language using the local formula registry.

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

## Latency tuning (optional)
You can tune response speed in `.env`:

```env
RETRIEVAL_TOP_K=10
FINAL_CONTEXT_CHUNKS=5
MAX_CONTEXT_CHARS=7500
MAX_CHARS_PER_CHUNK=1500
ENABLE_LATENCY_LOGS=false
OLLAMA_MODEL=llama3.2
OLLAMA_NUM_PREDICT=700
DEMO_MODE=false
DEMO_OLLAMA_MODEL=phi3
```

Notes:
- Lower `FINAL_CONTEXT_CHUNKS` or `MAX_CONTEXT_CHARS` to reduce prompt size.
- Keep `RETRIEVAL_TOP_K` high enough for source quality.
- For demo speed, set `DEMO_MODE=true` and choose a faster local model in `DEMO_OLLAMA_MODEL`.
- Calculation answers can bypass LLM generation when all values are source-validated, so they are usually faster than open-ended questions.

## How to add documents
### Option A: Batch indexing
- Put PDFs in `Source_files/`
- Run:

```powershell
INDEX_SOURCE_FILES.bat
```

Optional legacy scripts:
- `INDEX_BOTH.bat` indexes `Existing_files/` and `Source_files/`
- `INDEX_EXISTING_FILES.bat` indexes only `Existing_files/`

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

## Document folders
`Source_files/`
- Main local folder for PDFs
- Examples: annual reports, tax documents, invoices, client files

`Existing_files/` (optional)
- Kept for compatibility with earlier indexing scripts
- Not required for normal Q&A

## Example questions
- Which expenses in these files look deductible?
- What tax obligations are mentioned for this client?
- Where do these documents mention VAT rules?
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
Existing_files/   Baseline/reference PDFs (ignored in git)
Source_files/     Local PDF input folder (ignored in git)
uploads/          Temporary upload folder (ignored in git)
```

## Manual test checklist
1. Start services and open `http://localhost:8100/`.
2. Put one PDF in `Source_files/`, then run `INDEX_SOURCE_FILES.bat`.
3. Ask a normal question in the UI and verify source cards appear.
4. Upload a PDF through `/api/v1/documents/upload` and query it.
5. Delete an uploaded document with `DELETE /api/v1/documents/{document_id}` and re-query.
6. Run calculation checks:
   - `Calculate Palantir's revenue growth.`
   - `Calculate Palantir's gross margin.`
   - `Calculate Palantir's COGS.`
   - `Calculate Alphabet's revenue growth.`
   - `Compare Palantir and Alphabet revenue growth.`
   - `Calculate current ratio.`
