from src.services.llama_service import LlamaService


def test_repairs_common_mojibake_in_dutch_answer_templates():
    service = LlamaService.__new__(LlamaService)

    text = "financiÃ«le risicoâ€™s/risicoâs en geÃ¯ndexeerde bedragen â‚¬100; geïndexeerd: ÃƒÂ¯"

    assert service._repair_text_artifacts(text) == "financiële risico's/risico's en geïndexeerde bedragen €100; geïndexeerd: ï"
