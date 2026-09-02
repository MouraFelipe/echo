from __future__ import annotations

from tests import tk_stub  # noqa: F401

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from main import TranscriberApp
from transcriber import TranscribeError, Transcriber
from tests.conftest import FakeWhisper, FakeWord, sine


def fake_app() -> SimpleNamespace:
    app = SimpleNamespace()
    app._running = True
    app._busy_model = False
    app._audio_q = queue.Queue(maxsize=3)
    app._events = queue.Queue()
    app.transcriber = Transcriber()
    return app


class TestQueueDrop:
    def test_third_plus_one_drops_oldest(self):
        app = fake_app()
        for i in range(3):
            TranscriberApp._on_chunk(app, np.array([i], dtype=np.float32), True, 1.5)
        TranscriberApp._on_chunk(app, np.array([99], dtype=np.float32), False, 1.5)
        items = []
        while not app._audio_q.empty():
            items.append(int(app._audio_q.get_nowait()[0][0]))
        assert 0 not in items
        assert 99 in items
        assert len(items) == 3

    def test_ignores_chunks_when_idle(self):
        app = fake_app()
        app._running = False
        app._busy_model = False
        TranscriberApp._on_chunk(app, sine(0.1), True, 1.5)
        assert app._audio_q.empty()

    def test_accepts_during_boot_busy_flag(self):
        app = fake_app()
        app._running = False
        app._busy_model = True
        TranscriberApp._on_chunk(app, sine(0.1), True, 1.5)
        assert app._audio_q.qsize() == 1


class TestTranscribeLoop:
    def test_emits_line_on_text(self):
        app = fake_app()
        fake = FakeWhisper([FakeWord("Olá", 0, 0.2), FakeWord(" mundo", 0.3, 0.5)])
        app.transcriber._model = fake
        t = threading.Thread(target=TranscriberApp._transcribe_loop, args=(app,), daemon=True)
        t.start()
        app._audio_q.put((sine(0.5), True, 1.5))
        deadline = time.time() + 2
        kinds = []
        while time.time() < deadline:
            try:
                kinds.append(app._events.get(timeout=0.05))
            except queue.Empty:
                if kinds:
                    break
        assert any(k[0] == "line" and "Olá mundo" in k[1] for k in kinds)

    def test_empty_transcript_keeps_capturing_status(self):
        app = fake_app()
        app.transcriber._model = FakeWhisper([])
        t = threading.Thread(target=TranscriberApp._transcribe_loop, args=(app,), daemon=True)
        t.start()
        app._audio_q.put((sine(0.5), True, 1.5))
        kind, payload = app._events.get(timeout=2)
        assert kind == "status"
        assert "Capturando" in payload

    def test_transcribe_error_becomes_error_event(self):
        """Possível defeito: um chunk ruim publica 'error' e a UI PARA a sessão inteira."""
        app = fake_app()

        class Boom(Transcriber):
            def transcribe(self, audio):
                raise TranscribeError("Falha ao transcrever o bloco: boom")

        app.transcriber = Boom()
        t = threading.Thread(target=TranscriberApp._transcribe_loop, args=(app,), daemon=True)
        t.start()
        app._audio_q.put((sine(0.5), True, 1.5))
        kind, payload = app._events.get(timeout=2)
        assert kind == "error"
        assert "Falha ao transcrever" in payload


class TestStartGuard:
    def test_start_is_noop_when_running(self):
        app = fake_app()
        app._running = True
        app._busy_model = False
        called = []
        app._selected_device = lambda: called.append("device")
        TranscriberApp._on_start(app)
        assert called == []

    def test_start_is_noop_when_busy(self):
        app = fake_app()
        app._running = False
        app._busy_model = True
        app._selected_device = lambda: (_ for _ in ()).throw(AssertionError("não"))
        TranscriberApp._on_start(app)
