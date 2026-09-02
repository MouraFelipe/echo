from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from utils import WHISPER_SR, LoopbackDevice


def speakers(**overrides) -> LoopbackDevice:
    data = dict(index=7, name="Speakers (Realtek) [Loopback]", sample_rate=48_000, channels=2)
    data.update(overrides)
    return LoopbackDevice(**data)


def sine(seconds: float, sr: int = WHISPER_SR, freq: float = 440.0, amp: float = 0.2) -> np.ndarray:
    n = max(1, int(round(seconds * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def int16_stereo_bytes(frames: int, amp: int = 4000) -> bytes:
    left = np.linspace(-amp, amp, frames, dtype=np.int16)
    right = np.linspace(amp, -amp, frames, dtype=np.int16)
    interleaved = np.empty(frames * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    return interleaved.tobytes()


class FakeWord:
    def __init__(self, word: str, start: float, end: float) -> None:
        self.word = word
        self.start = start
        self.end = end


class FakeSegment:
    def __init__(self, words: list[FakeWord]) -> None:
        self.words = words


class FakeWhisper:
    def __init__(self, words: list[FakeWord] | None = None) -> None:
        self.words = words or []
        self.heard: list[np.ndarray] = []
        self.kwargs: list[dict] = []

    def transcribe(self, samples, **kwargs):
        self.heard.append(np.asarray(samples, dtype=np.float32).copy())
        self.kwargs.append(kwargs)
        return iter([FakeSegment(self.words)]), SimpleNamespace()


class FakeCapture:
    """Substitui AudioCapture: devolve N blocos 16 kHz e depois para."""

    def __init__(self, device, stop_event=None, pieces=None, error=None) -> None:
        self.device = device
        self._stop = stop_event or threading.Event()
        self._pieces = list(pieces or [])
        self._error = error
        self.started = False
        self.closed = False
        self.record_calls: list[float] = []

    def start(self) -> None:
        self.started = True
        if self._error == "start":
            from audio_capture import CaptureError

            raise CaptureError("Não foi possível abrir o loopback «fake».")

    def record(self, duration: float) -> np.ndarray:
        self.record_calls.append(duration)
        if self._stop.is_set():
            return np.zeros(0, dtype=np.float32)
        if self._error == "record":
            from audio_capture import CaptureError

            raise CaptureError(
                "O áudio do sistema parou. A captura foi pausada. Clique em Iniciar para retomar."
            )
        if not self._pieces:
            self._stop.set()
            return np.zeros(0, dtype=np.float32)
        return self._pieces.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def device() -> LoopbackDevice:
    return speakers()
