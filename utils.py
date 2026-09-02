"""Diagnóstico de loopback, tempo, resample 16 kHz e persistência."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample

WHISPER_SR = 16_000
CHUNK_SECONDS = 6.0
OVERLAP_SECONDS = 1.5
HOP_SECONDS = CHUNK_SECONDS - OVERLAP_SECONDS


@dataclass(frozen=True, slots=True)
class LoopbackDevice:
    index: int
    name: str
    sample_rate: int
    channels: int

    @property
    def label(self) -> str:
        return f"{self.name}  ·  {self.sample_rate} Hz  ·  {self.channels} ch"


class DeviceError(RuntimeError):
    """Nenhum loopback utilizável — a UI mostra, o app não fecha."""


def now_clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_line(clock: str, text: str) -> str:
    return f"[{clock}] {text.strip()}"


def int16_to_float32(raw: bytes, channels: int) -> np.ndarray:
    samples = np.frombuffer(raw, dtype=np.int16)
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    audio = samples.astype(np.float32) / 32768.0
    if channels > 1:
        usable = audio.size - (audio.size % channels)
        audio = audio[:usable].reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32)


def resample_to_whisper(mono: np.ndarray, native_sr: int) -> np.ndarray:
    """Downmix já feito. Reamostra explicitamente para 16 kHz — nunca pule esta etapa."""
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return mono
    if native_sr == WHISPER_SR:
        return mono
    if native_sr <= 0:
        raise ValueError(f"Taxa nativa inválida: {native_sr}")
    target_len = max(1, int(round(mono.size * (WHISPER_SR / native_sr))))
    return resample(mono, target_len).astype(np.float32)


def audio_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def list_loopback_devices() -> list[LoopbackDevice]:
    """Lista só dispositivos WASAPI loopback — nunca o microfone."""
    if sys.platform != "win32":
        raise DeviceError(
            "WASAPI loopback só existe no Windows. "
            "Execute este app em um PC com Windows 10/11."
        )
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise DeviceError(
            "PyAudioWPatch não está instalado. Rode: pip install -r requirements.txt"
        ) from exc

    pa = pyaudio.PyAudio()
    try:
        try:
            pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as exc:
            raise DeviceError(
                "WASAPI não está disponível. Verifique os drivers de áudio do Windows."
            ) from exc

        found: list[LoopbackDevice] = []
        for info in pa.get_loopback_device_info_generator():
            channels = int(info.get("maxInputChannels") or 0)
            if channels <= 0:
                continue
            found.append(
                LoopbackDevice(
                    index=int(info["index"]),
                    name=str(info.get("name") or f"Loopback {info['index']}"),
                    sample_rate=int(info.get("defaultSampleRate") or 48_000),
                    channels=channels,
                )
            )
        return found
    finally:
        pa.terminate()


def default_loopback_device(devices: list[LoopbackDevice] | None = None) -> LoopbackDevice:
    """Prefere o loopback do playback padrão (speakers), não o primeiro da lista."""
    devices = list(devices) if devices is not None else list_loopback_devices()
    if not devices:
        raise DeviceError(
            "Nenhum dispositivo loopback foi encontrado. "
            "Confira se há um playback padrão (fones/caixas) e rode "
            "`python -m pyaudiowpatch` para o diagnóstico completo."
        )

    try:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = pa.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
            default_name = str(default_out.get("name") or "")
        finally:
            pa.terminate()
        if default_name:
            for device in devices:
                if default_name in device.name:
                    return device
    except Exception:
        pass
    return devices[0]


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def save_transcript(text: str, path: Path | None = None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = app_dir() / "transcripts" / f"transcricao_{stamp}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8-sig")
    return path
