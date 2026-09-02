"""Captura o áudio que o Windows está reproduzindo (WASAPI loopback)."""

from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd

from utils import WHISPER_SR, audio_rms, is_speech, resample_audio, to_mono

SPEECH_FLUSH_SILENCE_S = 0.80
MAX_SEGMENT_S = 5.0
COLLECT_TIMEOUT_S = 0.25


class CaptureError(RuntimeError):
    """Falha recuperável de captura — a UI deve mostrar, não fechar."""


class LoopbackCapture:
    """Stream WASAPI loopback + segmentação por silêncio, em threads próprias."""

    def __init__(
        self,
        on_segment: Callable[[np.ndarray], None],
        on_error: Callable[[str], None],
        on_level: Callable[[float], None] | None = None,
        on_device: Callable[[str], None] | None = None,
    ) -> None:
        self._on_segment = on_segment
        self._on_error = on_error
        self._on_level = on_level
        self._on_device = on_device
        self._blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stream: sd.InputStream | None = None
        self._collector: threading.Thread | None = None
        self._running = threading.Event()
        self._native_sr = 48_000
        self.device_label = "—"

    def start(self) -> None:
        if self._running.is_set():
            return
        if sys.platform != "win32":
            raise CaptureError(
                "WASAPI loopback só existe no Windows. "
                "Execute este app em um PC com Windows."
            )

        device_index, info = _resolve_wasapi_output()
        self._native_sr = int(info.get("default_samplerate") or 48_000)
        channels = 2 if int(info.get("max_output_channels") or 0) >= 2 else 1
        self.device_label = f"{info['name']}  ·  WASAPI loopback  ·  {self._native_sr} Hz"
        if self._on_device:
            self._on_device(self.device_label)

        extra = _wasapi_loopback_settings()
        self._running.set()
        self._drain_queue()

        try:
            self._stream = sd.InputStream(
                device=device_index,
                samplerate=self._native_sr,
                channels=channels,
                dtype="float32",
                extra_settings=extra,
                callback=self._callback,
                finished_callback=self._on_stream_finished,
                blocksize=0,
                latency="low",
            )
            self._stream.start()
        except Exception as exc:
            self._running.clear()
            raise CaptureError(_friendly_capture_error(exc)) from exc

        self._collector = threading.Thread(
            target=self._collect_loop,
            name="audio-segmenter",
            daemon=True,
        )
        self._collector.start()

    def stop(self) -> None:
        self._running.clear()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        worker = self._collector
        if worker is not None and worker.is_alive() and threading.current_thread() is not worker:
            worker.join(timeout=1.5)
        self._collector = None
        self._drain_queue()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if not self._running.is_set():
            return
        if status:
            message = str(status)
            if "abort" in message.lower() or "invalid" in message.lower():
                self._emit_error(f"O dispositivo de áudio falhou: {message}")
                return
        try:
            self._blocks.put_nowait(indata.copy())
        except queue.Full:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                pass
            try:
                self._blocks.put_nowait(indata.copy())
            except queue.Full:
                pass

    def _on_stream_finished(self) -> None:
        if self._running.is_set():
            self._emit_error(
                "O áudio do sistema parou. A captura foi pausada. "
                "Clique em Iniciar para retomar."
            )

    def _collect_loop(self) -> None:
        pending: list[np.ndarray] = []
        voiced = False
        silence_samples = 0
        silence_needed = int(self._native_sr * SPEECH_FLUSH_SILENCE_S)
        max_samples = int(self._native_sr * MAX_SEGMENT_S)

        def flush(force: bool = False) -> None:
            nonlocal pending, voiced, silence_samples
            if not pending:
                return
            audio = np.concatenate(pending)
            pending = []
            silence_samples = 0
            had_voice = voiced
            voiced = False
            if not had_voice and not force:
                return
            if audio.size < int(self._native_sr * 0.25):
                return
            if not is_speech(audio) and not force:
                return
            try:
                resampled = resample_audio(audio, self._native_sr, WHISPER_SR)
                self._on_segment(resampled)
            except Exception as exc:
                self._emit_error(f"Falha ao processar o bloco de áudio: {exc}")

        try:
            while self._running.is_set():
                try:
                    block = self._blocks.get(timeout=COLLECT_TIMEOUT_S)
                except queue.Empty:
                    continue
                mono = to_mono(block)
                pending.append(mono)
                rms = audio_rms(mono)
                if self._on_level:
                    self._on_level(rms)
                if is_speech(mono):
                    voiced = True
                    silence_samples = 0
                else:
                    silence_samples += mono.size

                total = sum(part.size for part in pending)
                if voiced and silence_samples >= silence_needed:
                    flush()
                elif total >= max_samples:
                    flush()
        except Exception as exc:
            self._emit_error(f"A captura foi interrompida: {exc}")
        finally:
            flush(force=True)

    def _emit_error(self, message: str) -> None:
        self._running.clear()
        try:
            self._on_error(message)
        except Exception:
            pass

    def _drain_queue(self) -> None:
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                break


def _wasapi_hostapi_index() -> int:
    for index, api in enumerate(sd.query_hostapis()):
        name = str(api.get("name") or "")
        if "WASAPI" in name.upper():
            return index
    raise CaptureError(
        "WASAPI não foi encontrado neste sistema. "
        "O transcritor precisa do Windows com drivers de áudio ativos."
    )


def _resolve_wasapi_output() -> tuple[int, dict[str, Any]]:
    hostapi = _wasapi_hostapi_index()
    api = sd.query_hostapis(hostapi)
    default_out = api.get("default_output_device", -1)
    devices = sd.query_devices()

    if isinstance(default_out, int) and default_out >= 0:
        info = dict(sd.query_devices(default_out))
        if int(info.get("max_output_channels") or 0) > 0:
            return default_out, info

    for index, device in enumerate(devices):
        if device.get("hostapi") == hostapi and int(device.get("max_output_channels") or 0) > 0:
            return index, dict(device)

    raise CaptureError(
        "Nenhum dispositivo de saída WASAPI foi encontrado. "
        "Verifique se há um playback padrão no Windows."
    )


def _wasapi_loopback_settings() -> Any:
    settings_cls = getattr(sd, "WasapiSettings", None)
    if settings_cls is None:
        raise CaptureError(
            "Esta instalação do sounddevice não expõe WasapiSettings. "
            "Reinstale sounddevice no Windows (pip install sounddevice==0.5.1)."
        )
    try:
        return settings_cls(loopback=True, exclusive=False)
    except TypeError:
        return settings_cls(loopback=True)


def _friendly_capture_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()
    if "invalid device" in lowered or "device unavailable" in lowered:
        return (
            "O dispositivo de áudio ficou indisponível. "
            "Confira o playback padrão do Windows e clique em Iniciar."
        )
    if "wasapi" in lowered or "host api" in lowered:
        return (
            "Falha ao abrir o loopback WASAPI. "
            "Feche apps que estejam em modo exclusivo (alguns jogos/DAWs)."
        )
    return f"Não foi possível iniciar a captura: {text}"
