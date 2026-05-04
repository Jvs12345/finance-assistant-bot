# Run This Project On Your Own Machine

This is a practical setup guide for running the Financial Document Assistant locally.

## 1. What you need
- Windows 10/11
- Python 3.10+
- Docker Desktop
- Git
- Ollama

## 2. Clone the project
```powershell
git clone https://github.com/Jvs12345/BeInformed_chatbot.git
cd BeInformed_chatbot
```

If you renamed the repository, use your new URL instead.

## 3. Create local environment file
```powershell
copy .env.example .env
```

Open `.env` and set at least:
- `API_KEY` to any local value (example: `demo-key`)
- keep `ELASTICSEARCH_URL=http://localhost:39200`

## 4. Create Python environment and install packages
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Start Docker services
```powershell
docker compose -p financial-bot up -d
```

This starts:
- app on `http://localhost:8100`
- Elasticsearch on `http://localhost:39200`
- PostgreSQL on `localhost:55432`

## 6. Start Ollama
In a separate terminal:
```powershell
ollama serve
```

Then pull a model (first time only):
```powershell
ollama pull llama3.2
```

## 7. Start the app (easy way)
Run:
```powershell
RUN_FINANCIAL_ASSISTANT.bat
```

This checks Ollama and opens the app in your browser.

## 8. Add PDFs and index them
1. Put your PDFs in `Source_files/`
2. Run:
```powershell
INDEX_PDFS.bat
```

After indexing, the documents are searchable in chat.

## 9. Open and test
- UI: `http://localhost:8100/index.html`
- API docs: `http://localhost:8100/docs`

Try a question like:
- `Which expenses in these files look deductible?`

## 10. Optional: upload via API instead of batch indexing
```powershell
curl -X POST "http://localhost:8100/api/v1/documents/upload" `
  -H "Authorization: Bearer demo-key" `
  -F "file=@Source_files\sample.pdf" `
  -F "category=tax_law" `
  -F "jurisdiction=Netherlands" `
  -F "tax_year=2025"
```

## 11. Common issues

### Ollama connection error
If you see connection errors:
- make sure `ollama serve` is running
- test in terminal:
```powershell
curl.exe http://localhost:11434/api/tags
```

### No results in chat
- make sure you indexed files with `INDEX_PDFS.bat`
- check that PDFs contain selectable text (OCR issues can fail extraction)

### Docker memory too high
Set WSL memory limit in `%USERPROFILE%\.wslconfig` and restart WSL.

## 12. Stop services
```powershell
docker compose -p financial-bot down
```

If needed, stop Ollama manually by closing its terminal or ending the process.

## Notes
- This is a local proof-of-concept project.
- It is for document analysis support, not final professional tax/legal advice.
