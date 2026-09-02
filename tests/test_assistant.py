from __future__ import annotations

import json
from io import BytesIO

import pytest

import assistant as assistant_mod
from assistant import AssistantError, answer, extract_actions, reset_ollama_probe, run_assistant, summarize


SAMPLE = """
[10:00:01] A reunião começa em dois minutos. Precisamos fechar o prazo da entrega na sexta.
[10:00:08] Vamos agendar uma call amanhã com o cliente da Magazine.
[10:00:15] Combinado eu envio o relatório e você testa o PDV depois do almoço.
[10:00:22] Se o áudio parar a janela continua aberta sem fechar o aplicativo.
[10:00:30] O transcritor captura o que o Windows está reproduzindo sem usar o microfone.
"""


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    reset_ollama_probe()
    monkeypatch.setattr(assistant_mod, "_has_ollama", lambda: False)
    yield
    reset_ollama_probe()


class TestAssistantGuards:
    def test_too_short(self):
        with pytest.raises(AssistantError, match="algumas frases"):
            run_assistant("resumo", "oi")

    def test_question_without_text(self):
        with pytest.raises(AssistantError, match="Escreva uma pergunta"):
            run_assistant("pergunta", SAMPLE, "")

    def test_unknown_task(self):
        with pytest.raises(AssistantError, match="Tarefa desconhecida"):
            run_assistant("traduzir", SAMPLE)


class TestSummarizeAndActions:
    def test_resumo_has_header_and_bullets(self):
        text = run_assistant("resumo", SAMPLE)
        assert text.startswith("Resumo — assistente local")
        assert "• " in text
        assert "[" not in text  # timestamps stripped

    def test_tarefas_finds_hints(self):
        text = run_assistant("tarefas", SAMPLE)
        assert "Possíveis tarefas" in text
        assert "prazo" in text.lower() or "agendar" in text.lower()

    def test_tarefas_without_hints_falls_back(self):
        blob = (
            "O céu está azul hoje de manhã na cidade inteira. "
            "Os pássaros cantam perto da janela aberta agora. "
            "Ninguém mencionou reunião cliente relatório ou entrega fiscal."
        )
        text = extract_actions(blob)
        assert "Nenhum compromisso explícito" in text

    def test_answer_uses_overlap(self):
        text = run_assistant("pergunta", SAMPLE, "qual o prazo da entrega?")
        assert "Com base na transcrição" in text
        assert "sexta" in text.lower()

    def test_answer_no_overlap(self):
        text = answer(
            "O transcritor captura o playback do Windows sem microfone ligado agora.",
            "qual o preço do bitcoin",
        )
        assert "Não achei um trecho" in text

    def test_accents_survive(self):
        text = summarize(
            "A ação emergencial começa agora mesmo neste exato instante. "
            "Reunião às quinze horas com o time de parametrização fiscal."
        )
        assert "ação" in text.lower() or "reunião" in text.lower()


class TestOllamaOptional:
    def test_uses_ollama_when_available(self, monkeypatch):
        reset_ollama_probe()
        assistant_mod._ollama_ready = True
        assistant_mod._ollama_model = "llama3.2"
        monkeypatch.setattr(assistant_mod, "_has_ollama", lambda: True)

        class FakeResp(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        payload = json.dumps({"message": {"content": "  resumo ollama  "}}).encode()

        def fake_urlopen(req, timeout=20):
            assert "/api/chat" in getattr(req, "full_url", str(req))
            return FakeResp(payload)

        monkeypatch.setattr(assistant_mod.urllib.request, "urlopen", fake_urlopen)
        out = run_assistant("resumo", SAMPLE)
        assert out == "resumo ollama"

    def test_probe_failure_is_cached(self, monkeypatch):
        reset_ollama_probe()

        def boom(*a, **k):
            raise OSError("down")

        monkeypatch.setattr(assistant_mod.urllib.request, "urlopen", boom)
        assert assistant_mod._has_ollama() is False
        # segunda vez não chama de novo
        monkeypatch.setattr(
            assistant_mod.urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("não deveria reprobar")),
        )
        assert assistant_mod._has_ollama() is False
        reset_ollama_probe()
