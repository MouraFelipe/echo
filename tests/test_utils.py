from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils import (
    WHISPER_SR,
    app_dir,
    audio_rms,
    default_loopback_device,
    format_elapsed,
    format_line,
    int16_to_float32,
    list_loopback_devices,
    resample_to_whisper,
    save_transcript,
    DeviceError,
)
from tests.conftest import speakers


class TestFormatElapsed:
    def test_zero(self):
        assert format_elapsed(0) == "00:00:00"

    def test_negative_clamped(self):
        assert format_elapsed(-12.9) == "00:00:00"

    def test_seconds_truncated(self):
        assert format_elapsed(59.9) == "00:00:59"

    def test_one_minute(self):
        assert format_elapsed(60) == "00:01:00"

    def test_one_hour(self):
        assert format_elapsed(3600) == "01:00:00"

    def test_mixed(self):
        assert format_elapsed(3661) == "01:01:01"


class TestFormatLine:
    def test_strips_and_wraps(self):
        assert format_line("12:00:01", "  olá mundo  ") == "[12:00:01] olá mundo"

    def test_keeps_accents(self):
        line = format_line("08:09:10", "Ação: reunião às 15h")
        assert "Ação" in line and line.startswith("[08:09:10]")


class TestInt16ToFloat32:
    def test_empty(self):
        out = int16_to_float32(b"", 2)
        assert out.dtype == np.float32
        assert out.size == 0

    def test_mono_full_scale(self):
        raw = np.array([32767, -32768], dtype=np.int16).tobytes()
        out = int16_to_float32(raw, 1)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(32767 / 32768.0)
        assert out[1] == pytest.approx(-1.0)

    def test_stereo_downmix_mean(self):
        # L=1000, R=-1000 → mono 0
        stereo = np.array([1000, -1000, 2000, -2000], dtype=np.int16).tobytes()
        out = int16_to_float32(stereo, 2)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.0)

    def test_drops_incomplete_frame(self):
        # 3 int16 with 2 channels → 1 leftover sample discarded
        raw = np.array([1, 2, 3], dtype=np.int16).tobytes()
        out = int16_to_float32(raw, 2)
        assert out.shape == (1,)


class TestResample:
    def test_empty_passthrough(self):
        out = resample_to_whisper(np.zeros(0, dtype=np.float32), 48_000)
        assert out.size == 0

    def test_already_16k_same_object_values(self):
        src = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)
        out = resample_to_whisper(src, WHISPER_SR)
        np.testing.assert_array_equal(out, src)

    def test_48k_to_16k_length(self):
        src = np.ones(48000, dtype=np.float32)
        out = resample_to_whisper(src, 48_000)
        assert out.shape == (16000,)
        assert out.dtype == np.float32

    def test_invalid_rate(self):
        with pytest.raises(ValueError, match="Taxa nativa inválida"):
            resample_to_whisper(np.ones(10, dtype=np.float32), 0)


class TestRms:
    def test_empty_is_zero(self):
        assert audio_rms(np.zeros(0, dtype=np.float32)) == 0.0

    def test_silence(self):
        assert audio_rms(np.zeros(1024, dtype=np.float32)) == 0.0

    def test_known_square(self):
        samples = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
        assert audio_rms(samples) == pytest.approx(0.5)


class TestDevicesLinux:
    def test_list_raises_outside_windows(self):
        with pytest.raises(DeviceError, match="Windows"):
            list_loopback_devices()

    def test_default_empty_list(self):
        with pytest.raises(DeviceError, match="Nenhum dispositivo loopback"):
            default_loopback_device([])

    def test_default_prefers_name_match(self, monkeypatch):
        devices = [
            speakers(index=1, name="HDMI [Loopback]"),
            speakers(index=2, name="Speakers (Realtek) [Loopback]"),
        ]

        class FakePa:
            def get_host_api_info_by_type(self, _t):
                return {"defaultOutputDevice": 9}

            def get_device_info_by_index(self, idx):
                assert idx == 9
                return {"name": "Speakers (Realtek)"}

            def terminate(self):
                pass

        fake_mod = type("m", (), {"PyAudio": lambda self=None: FakePa(), "paWASAPI": 13})
        monkeypatch.setattr("utils.pyaudio", fake_mod, raising=False)
        # default_loopback_device imports pyaudiowpatch inside the try
        import sys
        from types import ModuleType

        mod = ModuleType("pyaudiowpatch")
        mod.PyAudio = lambda: FakePa()
        mod.paWASAPI = 13
        monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)
        chosen = default_loopback_device(devices)
        assert chosen.index == 2

    def test_default_falls_back_to_first(self):
        devices = [speakers(index=3, name="A"), speakers(index=4, name="B")]
        assert default_loopback_device(devices).index == 3

    def test_label(self):
        assert speakers().label == "Speakers (Realtek) [Loopback]  ·  48000 Hz  ·  2 ch"


class TestSaveTranscript:
    def test_writes_utf8_sig_and_newline(self, tmp_path: Path):
        path = tmp_path / "out.txt"
        saved = save_transcript("ação ção\nlinha 2", path)
        raw = saved.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        text = saved.read_text(encoding="utf-8-sig")
        assert text.endswith("\n")
        assert "ação" in text

    def test_default_creates_transcripts_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("utils.app_dir", lambda: tmp_path)
        saved = save_transcript("hello")
        assert saved.parent.name == "transcripts"
        assert saved.name.startswith("transcricao_")
        assert saved.read_text(encoding="utf-8-sig").strip() == "hello"

    def test_app_dir_not_frozen(self):
        assert app_dir().is_dir()
