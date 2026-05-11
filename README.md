# Financial Document Assistant

## Short description
This project is a local document assistant for financial and technical files. It searches indexed documents first, then answers with source references so output can be checked quickly.

## Why this project exists
The project started as a local documentation assistant and was extended to accounting, tax, and engineering documents. The main goals are:
- keep document processing local
- keep answers traceable to source text
- support practical preparation work (not final sign-off)

## Why we switched to Gemma (Ollama)
Default model choice is now `gemma3:4b`.

Reason for the switch:
- better instruction-following for fixed answer structures used in this app
- more consistent Dutch output in mixed Dutch/English document sets
- stable local performance for Q&A, summary, checklist, and pilot-style prompts
- fully local runtime through Ollama (no cloud dependency)

This does not mean Gemma is always best in every setup. It is currently the best fit for this project’s local workflow and demo goals.

## What it can do
- Index PDFs from `Source_files/` in batch mode.
- Index reference PDFs from `Existing_files/`.
- Upload PDFs through the API and index them in the same Elasticsearch index.
- Ask questions through the web UI and API.
- Filter retrieval by document type, jurisdiction, and tax year.
- Show source snippets and open source PDFs from the answer.
- Run local calculations when the answer includes a calculation payload.
- Run formula-driven calculations from natural language using the local formula registry.
- Handle advisor workflows (missing info checks, inconsistency checks, advisory points, insurance risk scan, client dossier summary).
- Handle document-summary and engineering demo workflows (technical requirements, CE gaps, quotation prep, use-case prioritization, pilot framing).

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
OLLAMA_MODEL=gemma3:4b
OLLAMA_NUM_PREDICT=700
DEMO_MODE=false
DEMO_OLLAMA_MODEL=gemma3:4b
```

Notes:
- Lower `FINAL_CONTEXT_CHUNKS` or `MAX_CONTEXT_CHARS` to reduce prompt size.
- Keep `RETRIEVAL_TOP_K` high enough for source quality.
- For demo speed, set `DEMO_MODE=true` and choose a faster local model in `DEMO_OLLAMA_MODEL` if needed.
- Calculation answers can bypass LLM generation when all values are source-validated.

## How to add documents
### Option A: Batch indexing
- Put PDFs in `Source_files/`
- Run:

```powershell
INDEX_BOTH.bat
```

`INDEX_BOTH.bat` indexes `Existing_files/` and `Source_files/` into the local Elasticsearch index.

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
- Examples: winst-en-verliesrekening, balans, btw-overzicht, klantnotities, contracten, polisoverzicht

`Existing_files/`
- Reference or baseline documents
- Examples: tax law, VAT guidance, accounting guidance, jaarrekening checklist, insurance risk checklist

## Example questions
- Welke informatie ontbreekt nog voordat de jaarrekening kan worden opgesteld?
- Controleer of er inconsistenties zijn tussen de winst-en-verliesrekening, btw-overzicht en klantnotities.
- Welke drie adviespunten kan ik met deze MKB-klant bespreken op basis van de documenten?
- Welke verzekeringsrisico’s zie je op basis van de bedrijfsactiviteiten, activa en contractinformatie?
- Vat de belangrijkste punten uit dit klantdossier samen.
- Bereken de omzetgroei als de benodigde cijfers beschikbaar zijn.

## Limitations
- This is a working prototype.
- It does not replace an accountant, tax advisor, lawyer, or compliance officer.
- Output quality depends on document quality and OCR quality.
- Scanned PDFs can fail if text extraction is poor.
- RAG reduces hallucination risk but does not remove it completely.
- Human review is still required for important decisions.

## Local-first note
- Default setup is local: Elasticsearch + Ollama run on your own machine.
- Client data stays in your local documents unless you explicitly change the setup.
- This project supports preparation and document review, not final certified conclusions.

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
2. Put one PDF in `Source_files/`, then run `INDEX_BOTH.bat`.
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
