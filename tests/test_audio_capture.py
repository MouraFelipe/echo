from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from audio_capture import AudioCapture, CaptureError, capture_loop, FRAMES_PER_BUFFER
from tests.conftest import FakeCapture, int16_stereo_bytes, sine, speakers
from utils import CHUNK_SECONDS, HOP_SECONDS, OVERLAP_SECONDS, WHISPER_SR


class TestAudioCaptureRecord:
    def test_record_without_start(self, device):
        cap = AudioCapture(device)
        with pytest.raises(CaptureError, match="não foi iniciada"):
            cap.record(0.1)

    def test_missing_pyaudiowpatch(self, device, monkeypatch):
        import builtins

        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "pyaudiowpatch":
                raise ImportError("nope")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        cap = AudioCapture(device)
        with pytest.raises(CaptureError, match="PyAudioWPatch não está instalado"):
            cap.start()

    def test_open_failure_is_friendly(self, device, monkeypatch):
        class BoomPa:
            def open(self, **kwargs):
                raise OSError("Invalid device")

            def terminate(self):
                self.terminated = True

        fake = SimpleNamespace(PyAudio=lambda: BoomPa(), paInt16=8)
        import sys
        from types import ModuleType

        mod = ModuleType("pyaudiowpatch")
        mod.PyAudio = fake.PyAudio
        mod.paInt16 = 8
        monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)
        cap = AudioCapture(device)
        with pytest.raises(CaptureError, match="Não foi possível abrir o loopback"):
            cap.start()

    def test_record_downmix_and_resample(self, device, monkeypatch):
        frames_needed = int(0.05 * device.sample_rate)  # 2400 stereo frames at 48k

        class Stream:
            def __init__(self):
                self.reads = 0

            def read(self, frames, exception_on_overflow=False):
                self.reads += 1
                return int16_stereo_bytes(frames)

            def stop_stream(self):
                pass

            def close(self):
                pass

        class Pa:
            def open(self, **kwargs):
                assert kwargs["input"] is True
                assert kwargs["input_device_index"] == device.index
                assert kwargs["rate"] == 48_000
                assert kwargs["channels"] == 2
                assert kwargs["frames_per_buffer"] == FRAMES_PER_BUFFER
                return Stream()

            def terminate(self):
                pass

        import sys
        from types import ModuleType

        mod = ModuleType("pyaudiowpatch")
        mod.PyAudio = Pa
        mod.paInt16 = 8
        monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)

        cap = AudioCapture(device)
        cap.start()
        out = cap.record(0.05)
        cap.close()
        assert out.dtype == np.float32
        # 0.05s @ 48k → 16k
        assert out.size == pytest.approx(int(round(0.05 * WHISPER_SR)), abs=2)
        assert float(np.max(np.abs(out))) > 0

    def test_read_error_pauses_capture(self, device):
        class Stream:
            def read(self, frames, exception_on_overflow=False):
                raise OSError("overflow")

        cap = AudioCapture(device)
        cap._stream = Stream()
        cap._pyaudio = object()
        with pytest.raises(CaptureError, match="O áudio do sistema parou"):
            cap.record(0.2)

    def test_empty_chunks_without_stop_raises(self, device):
        class Stream:
            def read(self, frames, exception_on_overflow=False):
                return b""

        cap = AudioCapture(device)
        cap._stream = Stream()
        cap._pyaudio = object()
        with pytest.raises(CaptureError, match="O áudio do sistema parou"):
            cap.record(0.05)

    def test_stop_during_record_returns_empty(self, device):
        stop = threading.Event()
        stop.set()

        class Stream:
            def read(self, frames, exception_on_overflow=False):
                return int16_stereo_bytes(frames)

        cap = AudioCapture(device, stop_event=stop)
        cap._stream = Stream()
        cap._pyaudio = object()
        out = cap.record(1.0)
        assert out.size == 0


class TestCaptureLoop:
    def test_first_chunk_is_6s_then_4_5s(self, device):
        pieces = [sine(6.0), sine(4.5), sine(4.5)]
        fake = FakeCapture(device, pieces=pieces)
        chunks = []
        flags = []
        levels = []
        errors = []
        stop = threading.Event()

        def on_chunk(arr, first, overlap):
            chunks.append(arr.copy())
            flags.append(first)
            if len(chunks) >= 3:
                stop.set()

        capture_loop(
            device,
            stop,
            on_chunk,
            errors.append,
            on_level=levels.append,
            capture_cls=lambda d, stop_event=None: fake,
        )
        assert errors == []
        assert fake.started and fake.closed
        assert fake.record_calls[0] == CHUNK_SECONDS
        assert fake.record_calls[1] == HOP_SECONDS
        assert flags == [True, False, False]
        assert chunks[0].size == sine(6.0).size
        assert chunks[1].size == pytest.approx(int(6.0 * WHISPER_SR), abs=2)
        assert levels and all(lv >= 0 for lv in levels)

    def test_open_error_goes_to_callback_and_closes(self, device):
        fake = FakeCapture(device, error="start")
        errors = []
        capture_loop(
            device,
            threading.Event(),
            lambda *a: None,
            errors.append,
            capture_cls=lambda d, stop_event=None: fake,
        )
        assert errors and "loopback" in errors[0]
        assert fake.closed

    def test_record_drop_pauses_without_raising(self, device):
        fake = FakeCapture(device, pieces=[sine(6.0)], error="record")
        # force record error on first call
        fake._pieces = []
        fake._error = "record"
        errors = []
        capture_loop(
            device,
            threading.Event(),
            lambda *a: None,
            errors.append,
            capture_cls=lambda d, stop_event=None: fake,
        )
        assert any("áudio do sistema parou" in e for e in errors)
        assert fake.closed
