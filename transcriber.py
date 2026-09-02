"""faster-whisper local + deduplicação por timestamp de palavra."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from utils import OVERLAP_SECONDS, WHISPER_SR

SUPPORTED_LANGUAGES: dict[str, str] = {
    "pt": "Português",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "auto": "Detectar automaticamente",
}

SUPPORTED_MODELS: tuple[str, ...] = ("tiny", "base", "small")


class TranscribeError(RuntimeError):
    """Falha recuperável de transcrição — a UI mostra, o app não fecha."""


@dataclass(frozen=True, slots=True)
class WordStamp:
    word: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    text: str
    words: list[WordStamp]
    from_cache: bool


class Transcriber:
    """WhisperModel int8. Primeira execução baixa o modelo; as seguintes são locais."""

    def __init__(self, model_size: str = "base", language: str = "pt") -> None:
        self.model_size = _validate_model(model_size)
        self.language = _validate_language(language)
        self._model = None
        self._lock = threading.Lock()
        self.device = "cpu"
        self.compute_type = "int8"
        self.from_local_cache = False

    def set_language(self, language: str) -> None:
        self.language = _validate_language(language)

    def set_model_size(self, model_size: str) -> None:
        size = _validate_model(model_size)
        if size == self.model_size:
            return
        with self._lock:
            self.model_size = size
            self._model = None
            self.from_local_cache = False

    def ensure_loaded(self, on_status: Callable[[str], None] | None = None) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise TranscribeError(
                    "faster-whisper não está instalado. Rode: pip install -r requirements.txt"
                ) from exc

            device, compute_type = _pick_runtime()
            self.device = device
            self.compute_type = compute_type

            if on_status:
                on_status(
                    f"Carregando modelo {self.model_size} (cache local, se já baixado)…"
                )
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True,
                )
                self.from_local_cache = True
                return
            except Exception:
                if on_status:
                    on_status(
                        f"Primeira execução: baixando o modelo {self.model_size} "
                        "(depois fica 100% offline)…"
                    )
                try:
                    self._model = WhisperModel(
                        self.model_size,
                        device=device,
                        compute_type=compute_type,
                        local_files_only=False,
                    )
                    self.from_local_cache = False
                except Exception as exc:
                    if device != "cpu":
                        try:
                            self.device = "cpu"
                            self._model = WhisperModel(
                                self.model_size,
                                device="cpu",
                                compute_type="int8",
                                local_files_only=False,
                            )
                            self.from_local_cache = False
                            return
                        except Exception as fallback_exc:
                            raise TranscribeError(_friendly_model_error(fallback_exc)) from fallback_exc
                    raise TranscribeError(_friendly_model_error(exc)) from exc

    def transcribe(self, audio_16k_mono: np.ndarray) -> TranscriptChunk:
        self.ensure_loaded()
        samples = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        if samples.size < int(WHISPER_SR * 0.20):
            return TranscriptChunk("", [], self.from_local_cache)

        peak = float(np.max(np.abs(samples)))
        if peak > 1e-6:
            samples = (samples * (0.95 / peak)).astype(np.float32)

        language = None if self.language == "auto" else self.language
        with self._lock:
            model = self._model
            if model is None:
                raise TranscribeError("Modelo ainda não foi carregado.")
            try:
                segments, _info = model.transcribe(
                    samples,
                    language=language,
                    task="transcribe",
                    word_timestamps=True,
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                    condition_on_previous_text=False,
                )
                words: list[WordStamp] = []
                for segment in segments:
                    for item in segment.words or []:
                        token = (item.word or "").strip()
                        if not token:
                            continue
                        words.append(
                            WordStamp(
                                word=item.word or "",
                                start=float(item.start),
                                end=float(item.end),
                            )
                        )
            except Exception as exc:
                raise TranscribeError(f"Falha ao transcrever o bloco: {exc}") from exc
        return TranscriptChunk(join_words(words), words, self.from_local_cache)


def dedupe_by_time(
    words: list[WordStamp],
    *,
    is_first_chunk: bool,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> list[WordStamp]:
    """Descarta palavras cujo início cai na janela já coberta pelo chunk anterior."""
    if is_first_chunk or overlap_seconds <= 0:
        return list(words)
    cutoff = overlap_seconds - 0.02
    return [word for word in words if word.start >= cutoff]


def join_words(words: list[WordStamp]) -> str:
    if not words:
        return ""
    parts: list[str] = []
    for word in words:
        token = word.word
        if not token:
            continue
        if token.startswith((" ", "\n")) or not parts:
            parts.append(token)
        elif token[0] in ".,;:!?…)]}":
            parts.append(token)
        else:
            parts.append(" " + token)
    return "".join(parts).strip()


def _pick_runtime() -> tuple[str, str]:
    return "auto", "int8"


def _validate_language(language: str) -> str:
    key = (language or "pt").strip().lower()
    if key not in SUPPORTED_LANGUAGES:
        raise TranscribeError(f"Idioma não suportado: {language}")
    return key


def _validate_model(model_size: str) -> str:
    size = (model_size or "base").strip().lower()
    if size not in SUPPORTED_MODELS:
        raise TranscribeError(f"Modelo não suportado: {model_size}")
    return size


def _friendly_model_error(exc: BaseException) -> str:
    text = str(exc)
    lowered = text.lower()
    if "huggingface" in lowered or "connection" in lowered or "timed out" in lowered:
        return (
            "Não foi possível baixar o modelo (só a primeira execução precisa de internet). "
            "Depois o app usa local_files_only=True e funciona offline."
        )
    return f"Não foi possível carregar o modelo: {text}"
