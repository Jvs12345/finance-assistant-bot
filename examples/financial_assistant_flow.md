# Financial Assistant Example Flow

## 1. Upload a financial/tax document

```powershell
curl -X POST "http://localhost:8100/api/v1/documents/upload" `
  -H "Authorization: Bearer your_api_key" `
  -F "file=@Source_files\client_annual_report_2025.pdf" `
  -F "category=annual_report" `
  -F "jurisdiction=Netherlands" `
  -F "tax_year=2025" `
  -F "client_name=Client A" `
  -F "entity_type=bv" `
  -F "source_name=Client A Annual Report 2025"
```

## 2. Ask a question

```powershell
curl -X POST "http://localhost:8100/api/v1/llama/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"What information is missing before preparing this tax position?\",\"jurisdiction\":\"Netherlands\",\"tax_year\":2025,\"entity_type\":\"bv\",\"client_name\":\"Client A\",\"model\":\"llama3.2\"}"
```

## 3. Expected response

- Structured answer in five sections:
  - Direct answer
  - Evidence from documents
  - Important assumptions or missing information
  - Suggested next step
  - Disclaimer when relevant
- Source list includes document name, page, snippet, and metadata.
- If unsupported, response includes:
  - `I could not find this in the provided documents.`
