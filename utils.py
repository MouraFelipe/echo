"""Funções auxiliares: tempo, áudio e persistência da transcrição."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

WHISPER_SR = 16_000
SPEECH_RMS_THRESHOLD = 0.008


def now_clock() -> str:
    """Hora local no formato HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def format_elapsed(seconds: float) -> str:
    """Formata segundos decorridos como HH:MM:SS."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_line(clock: str, text: str) -> str:
    return f"[{clock}] {text.strip()}"


def audio_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def to_mono(frames: np.ndarray) -> np.ndarray:
    """Converte bloco (n,) ou (n, ch) para float32 mono."""
    data = np.asarray(frames, dtype=np.float32)
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1, dtype=np.float32)


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int = WHISPER_SR) -> np.ndarray:
    """Reamostra com interpolação linear (sem dependências extras)."""
    mono = to_mono(samples)
    if orig_sr <= 0 or target_sr <= 0 or mono.size == 0 or orig_sr == target_sr:
        return mono.astype(np.float32, copy=False)
    target_len = max(1, int(round(mono.size * (target_sr / orig_sr))))
    src_x = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, mono).astype(np.float32)


def peak_normalize(samples: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-6:
        return samples
    return (samples * (ceiling / peak)).astype(np.float32)


def is_speech(samples: np.ndarray, threshold: float = SPEECH_RMS_THRESHOLD) -> bool:
    return audio_rms(samples) >= threshold


def default_transcript_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path.cwd() / "transcripts"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"transcricao_{stamp}.txt"


def save_transcript(text: str, path: Path | None = None) -> Path:
    """Grava o texto com BOM UTF-8 para abrir bem no Bloco de Notas."""
    target = path or default_transcript_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    body = text.rstrip() + "\n"
    target.write_text(body, encoding="utf-8-sig")
    return target
