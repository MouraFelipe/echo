from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from utils import DeviceError, default_loopback_device, list_loopback_devices
from tests.conftest import speakers


def install_pyaudio(monkeypatch, *, loopbacks, wasapi=True, default_out="Speakers (Realtek)"):
    class FakePa:
        def get_host_api_info_by_type(self, _t):
            if not wasapi:
                raise OSError("no wasapi")
            return {"defaultOutputDevice": 1}

        def get_loopback_device_info_generator(self):
            yield from loopbacks

        def get_device_info_by_index(self, _i):
            return {"name": default_out}

        def terminate(self):
            self.dead = True

    mod = ModuleType("pyaudiowpatch")
    mod.PyAudio = FakePa
    mod.paWASAPI = 13
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)
    monkeypatch.setattr(sys, "platform", "win32")
    return FakePa


class TestListLoopback:
    def test_filters_zero_channels_and_maps_fields(self, monkeypatch):
        install_pyaudio(
            monkeypatch,
            loopbacks=[
                {"index": 1, "name": "Mic [Loopback]", "maxInputChannels": 0, "defaultSampleRate": 44100},
                {
                    "index": 4,
                    "name": "Speakers (Realtek) [Loopback]",
                    "maxInputChannels": 2,
                    "defaultSampleRate": 48000,
                },
            ],
        )
        found = list_loopback_devices()
        assert len(found) == 1
        assert found[0].index == 4
        assert found[0].channels == 2
        assert found[0].sample_rate == 48000

    def test_missing_name_gets_fallback(self, monkeypatch):
        install_pyaudio(
            monkeypatch,
            loopbacks=[{"index": 9, "maxInputChannels": 1, "defaultSampleRate": 16000}],
        )
        found = list_loopback_devices()
        assert found[0].name == "Loopback 9"

    def test_wasapi_missing(self, monkeypatch):
        install_pyaudio(monkeypatch, loopbacks=[], wasapi=False)
        with pytest.raises(DeviceError, match="WASAPI não está disponível"):
            list_loopback_devices()

    def test_import_error(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "pyaudiowpatch", None)
        # force import failure
        import builtins

        real = builtins.__import__

        def blocked(name, *a, **k):
            if name == "pyaudiowpatch":
                raise ImportError("x")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(DeviceError, match="PyAudioWPatch não está instalado"):
            list_loopback_devices()

    def test_always_terminates_pa(self, monkeypatch):
        dead = {"n": 0}

        class FakePa:
            def get_host_api_info_by_type(self, _t):
                raise RuntimeError("boom")

            def terminate(self):
                dead["n"] += 1

        mod = ModuleType("pyaudiowpatch")
        mod.PyAudio = FakePa
        mod.paWASAPI = 13
        monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(RuntimeError):
            list_loopback_devices()
        assert dead["n"] == 1
