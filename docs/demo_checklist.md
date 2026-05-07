# Demo checklist

## Goal
Show that a local AI agent can support accounting/advisory prep work without sending client data to external LLM APIs by default.

## Setup
1. Put reference documents in `Existing_files/`.
2. Put client documents in `Source_files/`.
3. Start services:
   - `docker compose -p financial-bot up -d`
   - `RUN_FINANCIAL_ASSISTANT.bat`
4. Index documents:
   - `INDEX_EXISTING_FILES.bat`
   - `INDEX_SOURCE_FILES.bat`

## Demo questions
1. Welke informatie ontbreekt nog voordat de jaarrekening kan worden opgesteld?
2. Controleer of er inconsistenties zijn tussen de winst-en-verliesrekening, btw-overzicht en klantnotities.
3. Welke drie adviespunten kan ik met deze MKB-klant bespreken op basis van de documenten?
4. Welke verzekeringsrisico’s zie je op basis van de bedrijfsactiviteiten, activa en contractinformatie?
5. Vat de belangrijkste punten uit dit klantdossier samen.
6. Bereken de omzetgroei als de benodigde cijfers beschikbaar zijn.

## What to verify during the demo
1. Answers show source references.
2. Missing information is explicitly listed when evidence is incomplete.
3. Inconsistencies are presented as manual review points, not final audit conclusions.
4. Calculation answers include formula, values, calculation steps, and sources.
5. The app stays local by default.

## Notes
- This is a proof of concept.
- Use outputs as preparation support.
- Important accounting, tax, and insurance conclusions should be verified by a qualified professional.
