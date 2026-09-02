"""Transcrição local com faster-whisper (CPU int8, GPU se existir)."""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from utils import WHISPER_SR, peak_normalize

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
    """Falha recuperável de transcrição — a UI deve mostrar, não fechar."""


class LocalTranscriber:
    """Carrega o modelo uma vez e transcreve blocos numpy em uma lock."""

    def __init__(self, model_size: str = "base", language: str = "pt") -> None:
        self.model_size = _validate_model(model_size)
        self.language = _validate_language(language)
        self._model = None
        self._lock = threading.Lock()
        self.device = "cpu"
        self.compute_type = "int8"
        self.loaded = False

    def set_language(self, language: str) -> None:
        self.language = _validate_language(language)

    def set_model_size(self, model_size: str) -> None:
        size = _validate_model(model_size)
        if size != self.model_size:
            with self._lock:
                self.model_size = size
                self._model = None
                self.loaded = False

    def ensure_loaded(self, on_status: Callable[[str], None] | None = None) -> None:
        with self._lock:
            if self._model is not None:
                self.loaded = True
                return
            if on_status:
                on_status(
                    f"Carregando modelo {self.model_size}… "
                    "na primeira vez o arquivo é baixado (~75–500 MB) e depois fica local."
                )
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise TranscribeError(
                    "faster-whisper não está instalado. "
                    "Rode: pip install -r requirements.txt"
                ) from exc

            device, compute_type = _pick_runtime()
            self.device = device
            self.compute_type = compute_type
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as exc:
                if device != "cpu":
                    try:
                        self.device = "cpu"
                        self.compute_type = "int8"
                        self._model = WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                        )
                    except Exception as fallback_exc:
                        raise TranscribeError(_friendly_model_error(fallback_exc)) from fallback_exc
                else:
                    raise TranscribeError(_friendly_model_error(exc)) from exc
            self.loaded = True

    def transcribe(self, audio_16k_mono: np.ndarray) -> str:
        self.ensure_loaded()
        samples = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        if samples.size < int(WHISPER_SR * 0.20):
            return ""
        samples = peak_normalize(samples)
        language = None if self.language == "auto" else self.language

        with self._lock:
            model = self._model
            if model is None:
                raise TranscribeError("Modelo de transcrição ainda não foi carregado.")
            try:
                segments, _info = model.transcribe(
                    samples,
                    language=language,
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                parts = [seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()]
            except Exception as exc:
                raise TranscribeError(f"Falha ao transcrever o bloco: {exc}") from exc
        return " ".join(parts).strip()


def _pick_runtime() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


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
            "Não foi possível baixar o modelo (primeira execução precisa de internet). "
            "Depois do download o app funciona 100% offline."
        )
    return f"Não foi possível carregar o modelo: {text}"
