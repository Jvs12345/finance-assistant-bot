# Gemini/Copilot Integration

This note explains how to swap the current Ollama step for Gemini or Copilot Studio.

Important: this is only about the answer generation step. Keep your current indexing and retrieval flow as-is.

## What stays the same
- PDF upload
- chunking + embeddings
- Elasticsearch index
- metadata filters
- source snippets in the final answer

## What changes
Right now the answer call goes through:
- `src/services/llama_service.py`
- `src/services/ollama_client.py`

You replace only that part.

## Clean way to do it
Add a small provider layer so you can switch models with an env var.

Suggested files:
- `src/services/llm_provider.py`
- `src/services/providers/ollama_provider.py`
- `src/services/providers/gemini_provider.py`
- `src/services/providers/copilot_provider.py`

Then `llama_service.py` calls `llm_provider.py` instead of calling Ollama directly.

## Env vars to add
Put placeholders in `.env.example`:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2

# Gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-1.5-pro

# Copilot Studio
COPILOT_ENDPOINT=
COPILOT_TENANT_ID=
COPILOT_CLIENT_ID=
COPILOT_CLIENT_SECRET=
```

Keep `ollama` as default.

## Gemini path (quickest)
1. Add Gemini SDK dependency.
2. Build `gemini_provider.py`.
3. Map your existing message format to Gemini request format.
4. Return plain text answer.
5. Keep existing error style (auth, timeout, invalid model, quota).

## Copilot Studio path
1. Confirm which endpoint you can call from your backend.
2. Set up service auth (client credentials).
3. Build `copilot_provider.py` to get token + call endpoint.
4. Parse answer text and pass it back in your normal response shape.

Copilot setup depends on your tenant, so expect more setup time than Gemini.

## Provider switch logic
In `llm_provider.py`:
- read `LLM_PROVIDER`
- route to `ollama`, `gemini`, or `copilot`
- fail early with a clear message if config is missing

## Minimal step-by-step plan
1. Move current Ollama code into an `ollama_provider.py` file.
2. Add `llm_provider.py` router.
3. Update `llama_service.py` to call the router.
4. Add Gemini provider and test one question.
5. Add Copilot provider if needed.

## Quick test checklist
- App still starts with `LLM_PROVIDER=ollama`.
- App starts with `LLM_PROVIDER=gemini` when key is set.
- Answers still show sources.
- Missing-context behavior still works.
- Invalid key gives a clear backend error.
- No keys exposed in frontend.

## Security
- Do not commit real keys.
- Keep `.env` ignored.
- Use placeholders in `.env.example`.
- Cloud providers mean retrieved text leaves your local machine.

## Recommendation
If you want the fastest swap for a demo, do Gemini first.
Keep Copilot as a second phase.
