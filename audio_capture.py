"""Captura loopback WASAPI via PyAudioWPatch + downmix + resample 16 kHz."""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from utils import (
    CHUNK_SECONDS,
    HOP_SECONDS,
    OVERLAP_SECONDS,
    WHISPER_SR,
    DeviceError,
    LoopbackDevice,
    audio_rms,
    int16_to_float32,
    resample_to_whisper,
)

FRAMES_PER_BUFFER = 1024


class CaptureError(RuntimeError):
    """Falha recuperável de captura — a UI mostra, o app não fecha."""


class AudioCapture:
    """Abre o loopback na taxa nativa e devolve blocos já em 16 kHz mono float32."""

    def __init__(self, device: LoopbackDevice, stop_event: threading.Event | None = None) -> None:
        self.device = device
        self._stop = stop_event or threading.Event()
        self._pa = None
        self._stream = None
        self._pyaudio = None

    def start(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise CaptureError(
                "PyAudioWPatch não está instalado. Rode: pip install -r requirements.txt"
            ) from exc

        self._pyaudio = pyaudio
        self._pa = pyaudio.PyAudio()
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.device.channels,
                rate=self.device.sample_rate,
                frames_per_buffer=FRAMES_PER_BUFFER,
                input=True,
                input_device_index=self.device.index,
            )
        except Exception as exc:
            self.close()
            raise CaptureError(_friendly_open_error(exc, self.device)) from exc

    def record(self, duration: float) -> np.ndarray:
        """Captura `duration` segundos na taxa nativa, downmix mono, resample 16 kHz."""
        if self._stream is None or self._pyaudio is None:
            raise CaptureError("A captura não foi iniciada.")

        native_sr = self.device.sample_rate
        needed = max(1, int(round(duration * native_sr)))
        chunks: list[np.ndarray] = []
        collected = 0
        empty_reads = 0

        while collected < needed and not self._stop.is_set():
            frames = min(FRAMES_PER_BUFFER, needed - collected)
            try:
                raw = self._stream.read(frames, exception_on_overflow=False)
            except Exception as exc:
                raise CaptureError(
                    "O áudio do sistema parou. A captura foi pausada. "
                    "Clique em Iniciar para retomar."
                ) from exc
            block = int16_to_float32(raw, self.device.channels)
            if block.size == 0:
                empty_reads += 1
                if empty_reads >= 8:
                    break
                continue
            empty_reads = 0
            chunks.append(block)
            collected += block.size

        if not chunks:
            if self._stop.is_set():
                return np.zeros(0, dtype=np.float32)
            raise CaptureError(
                "O áudio do sistema parou. A captura foi pausada. "
                "Clique em Iniciar para retomar."
            )

        native = np.concatenate(chunks)
        if native.size > needed:
            native = native[:needed]
        return resample_to_whisper(native, native_sr)

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        pa = self._pa
        self._pa = None
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def capture_loop(
    device: LoopbackDevice,
    stop_event: threading.Event,
    on_chunk: Callable[[np.ndarray, bool, float], None],
    on_error: Callable[[str], None],
    on_level: Callable[[float], None] | None = None,
    capture_cls: type[AudioCapture] = AudioCapture,
) -> None:
    """
    Primeiro chunk: 6 s. Seguintes: 4,5 s novos + cauda de 1,5 s (sobreposição).
    Cada array entregue já está em 16 kHz mono.
    """
    capture = capture_cls(device, stop_event=stop_event)
    try:
        capture.start()
        tail = np.zeros(0, dtype=np.float32)
        first = True
        while not stop_event.is_set():
            seconds = CHUNK_SECONDS if first else HOP_SECONDS
            piece = capture.record(seconds)
            if stop_event.is_set():
                break
            if piece.size == 0:
                continue
            chunk = piece if first else np.concatenate([tail, piece])
            overlap_samples = int(OVERLAP_SECONDS * WHISPER_SR)
            tail = chunk[-overlap_samples:] if chunk.size >= overlap_samples else chunk.copy()
            if on_level:
                on_level(audio_rms(chunk))
            on_chunk(chunk.astype(np.float32, copy=False), first, OVERLAP_SECONDS)
            first = False
    except (CaptureError, DeviceError) as exc:
        on_error(str(exc))
    except Exception as exc:
        on_error(f"A captura foi interrompida: {exc}")
    finally:
        capture.close()


def _friendly_open_error(exc: BaseException, device: LoopbackDevice) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()
    if "invalid" in lowered or "unavailable" in lowered:
        return (
            f"Não foi possível abrir o loopback «{device.name}». "
            "Escolha outro dispositivo na lista ou verifique o playback padrão."
        )
    return f"Falha ao abrir o loopback: {text}"
