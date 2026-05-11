# Demo Checklist

## Doel
Laat zien dat een lokale documentassistent voorbereiding kan ondersteunen zonder bedrijfsdocumenten standaard naar externe LLM-API’s te sturen.

## Setup
1. Zet referentiedocumenten in `Existing_files/`.
2. Zet project-/klantdocumenten in `Source_files/`.
3. Start services:
   - `docker compose -p financial-bot up -d`
   - `RUN_FINANCIAL_ASSISTANT.bat`
4. Indexeer documenten:
   - `INDEX_BOTH.bat`

## Demo vragen (accounting/advisor)
1. Welke informatie ontbreekt nog voordat de jaarrekening kan worden opgesteld?
2. Controleer of er inconsistenties zijn tussen de winst-en-verliesrekening, btw-overzicht en klantnotities.
3. Welke drie adviespunten kan ik met deze MKB-klant bespreken op basis van de documenten?
4. Welke verzekeringsrisico’s zie je op basis van de bedrijfsactiviteiten, activa en contractinformatie?
5. Vat de belangrijkste punten uit dit klantdossier samen.
6. Bereken de omzetgroei als de benodigde cijfers beschikbaar zijn.

## Wat je tijdens de demo controleert
1. Antwoorden bevatten bronverwijzingen.
2. Ontbrekende informatie wordt expliciet benoemd.
3. Inconsistenties worden als controlepunten gepresenteerd, niet als definitieve auditconclusie.
4. Berekeningen tonen formule, waarden, uitkomst en bronkoppeling.
5. De setup blijft lokaal, tenzij je die bewust wijzigt.

## Notities
- Gebruik output als voorbereiding en versneller, niet als eindbeslissing.
- Belangrijke conclusies blijven onder menselijke verantwoordelijkheid.
