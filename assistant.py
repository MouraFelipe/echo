"""Assistente local e gratuito sobre a transcrição.

Usa Ollama se estiver rodando neste PC (LLM de verdade, 100% local).
Caso contrário, um resumidor extrativo — numpy + heurística, sem nuvem.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import Counter

STOP = {
    "a", "o", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "pra", "com", "sem", "que", "se",
    "é", "são", "foi", "ser", "ter", "tem", "há", "ao", "à", "e", "ou", "mas", "como",
    "mais", "menos", "muito", "já", "não", "sim", "eu", "você", "ele", "ela", "nós",
    "eles", "isso", "este", "esta", "esse", "essa", "isto", "aqui", "ali", "the",
    "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "be", "this",
    "that", "it", "we", "you", "they", "a", "an",
}

ACTION_HINTS = (
    "vamos", "preciso", "precisa", "combinado", "combinamos", "prazo", "entregar",
    "ação", "tarefa", "fazer", "agendar", "marcar", "decidir", "decisão", "próximo",
    "amanhã", "segunda", "terça", "quarta", "quinta", "sexta", "enviar", "ligar",
    "need to", "let's", "action", "deadline", "follow up", "next step", "todo",
)

_ollama_ready: bool | None = None
_ollama_model: str | None = None


class AssistantError(RuntimeError):
    """Falha recuperável — a UI mostra, o app não fecha."""


def reset_ollama_probe() -> None:
    """Zera o cache do probe — só para testes e se o Ollama subir depois do 1º clique."""
    global _ollama_ready, _ollama_model
    _ollama_ready = None
    _ollama_model = None


def run_assistant(task: str, transcript: str, question: str = "") -> str:
    body = _plain(transcript)
    if len(body) < 40:
        raise AssistantError("Transcreva pelo menos algumas frases antes de usar o assistente.")
    if task == "pergunta" and not question.strip():
        raise AssistantError("Escreva uma pergunta sobre a transcrição.")

    if _has_ollama():
        generated = _ollama_chat(task, body, question)
        if generated:
            return generated.strip()

    if task == "resumo":
        return summarize(body)
    if task == "tarefas":
        return extract_actions(body)
    if task == "pergunta":
        return answer(body, question)
    raise AssistantError(f"Tarefa desconhecida: {task}")


def summarize(text: str, n: int = 5) -> str:
    sentences = _sentences(text)
    if not sentences:
        return "Não encontrei frases para resumir."
    ranked = _rank(sentences)
    picked = sorted(ranked[: min(n, len(ranked))], key=lambda item: item[0])
    bullets = "\n".join(f"• {item[1]}" for item in picked)
    return f"Resumo — assistente local (offline)\n\n{bullets}"


def extract_actions(text: str) -> str:
    sentences = _sentences(text)
    hits = [s for s in sentences if any(h in s.lower() for h in ACTION_HINTS)]
    if not hits:
        ranked = _rank(sentences)
        hits = [item[1] for item in ranked[:3]]
        note = "Nenhum compromisso explícito. Frases mais relevantes:"
    else:
        note = "Possíveis tarefas / combinados:"
    bullets = "\n".join(f"• {s}" for s in hits[:8])
    return f"{note}\n\n{bullets}"


def answer(text: str, question: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return "A transcrição está vazia."
    q_tokens = _tokens(question)
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        overlap = len(q_tokens & _tokens(sentence))
        scored.append((overlap, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return "Não achei um trecho que responda isso. Reformule com palavras da transcrição."
    bits = [s for score, s in scored[:3] if score > 0]
    return "Com base na transcrição:\n\n" + "\n".join(f"• {s}" for s in bits)


def _plain(transcript: str) -> str:
    lines = []
    for raw in transcript.splitlines():
        line = re.sub(r"^\s*\[[^\]]+\]\s*", "", raw).strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…;:])\s+|\n+", text)
    out = []
    for part in parts:
        clean = re.sub(r"\s+", " ", part).strip(" -•\t")
        if len(clean) >= 24:
            out.append(clean)
    return out or ([text.strip()] if text.strip() else [])


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOP}


def _rank(sentences: list[str]) -> list[tuple[int, str]]:
    freq: Counter[str] = Counter()
    tokenized = []
    for sentence in sentences:
        tokens = _tokens(sentence)
        tokenized.append(tokens)
        freq.update(tokens)
    scored = []
    total = len(sentences)
    for index, (sentence, tokens) in enumerate(zip(sentences, tokenized)):
        tf = sum(freq[t] for t in tokens)
        position = 1.15 if index < 2 or index >= total - 1 else 1.0
        length_pen = 1.0 if 8 <= len(tokens) <= 28 else 0.75
        scored.append((tf * position * length_pen, index, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(index, sentence) for _score, index, sentence in scored]


def _has_ollama() -> bool:
    global _ollama_ready, _ollama_model
    if _ollama_ready is not None:
        return _ollama_ready
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [item.get("name") for item in payload.get("models") or [] if item.get("name")]
        _ollama_model = models[0] if models else None
        _ollama_ready = bool(_ollama_model)
    except Exception:
        _ollama_ready = False
        _ollama_model = None
    return _ollama_ready


def _ollama_chat(task: str, transcript: str, question: str) -> str | None:
    if not _ollama_model:
        return None
    clipped = transcript[:6000]
    if task == "resumo":
        user = (
            "Resuma em português, em 4 a 6 bullets, só com o que está no texto:\n\n"
            f"{clipped}"
        )
    elif task == "tarefas":
        user = (
            "Liste tarefas, prazos e combinados em bullets. Se não houver, diga isso. Texto:\n\n"
            f"{clipped}"
        )
    else:
        user = f"Pergunta: {question.strip()}\n\nResponda só com base neste texto:\n\n{clipped}"

    body = json.dumps(
        {
            "model": _ollama_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "Você é o assistente do Voxa. Use apenas a transcrição. Não invente.",
                },
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return (payload.get("message") or {}).get("content") or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
