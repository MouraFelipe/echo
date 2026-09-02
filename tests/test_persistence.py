from __future__ import annotations

from tests import tk_stub  # noqa: F401

import threading
from pathlib import Path
from types import SimpleNamespace

from main import TranscriberApp


def harness(tmp_path: Path | None = None) -> SimpleNamespace:
    app = SimpleNamespace()
    app._running = True
    app._stop_event = threading.Event()
    app._lines = ["[10:00:01] Olá", "[10:00:08] Ação"]
    app.status = ""
    app.level = None
    app._set_buttons = lambda **k: None
    app._set_status = lambda msg, error=False: setattr(app, "status", msg)
    app._draw_level = lambda v: setattr(app, "level", v)
    return app


class TestStop:
    def test_stop_sets_event_and_saves(self, tmp_path, monkeypatch):
        app = harness()
        saved = tmp_path / "x.txt"
        monkeypatch.setattr("main.save_transcript", lambda text: saved.write_text(text) or saved)
        TranscriberApp._on_stop(app)
        assert app._running is False
        assert app._stop_event.is_set()
        assert app.level == 0.0
        assert "salvo em" in app.status
        assert "Olá" in saved.read_text()

    def test_stop_without_lines_does_not_save(self, monkeypatch):
        app = harness()
        app._lines = []
        monkeypatch.setattr(
            "main.save_transcript",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("não deveria salvar")),
        )
        TranscriberApp._on_stop(app)
        assert app.status == "Parado"

    def test_stop_swallows_save_oserror(self, monkeypatch):
        """Possível defeito: falha ao salvar no Parar é silenciada."""
        app = harness()
        monkeypatch.setattr("main.save_transcript", lambda text: (_ for _ in ()).throw(OSError("disk")))
        TranscriberApp._on_stop(app)
        assert app._running is False
        assert app.status == "Parado"


class TestCopyClear:
    def test_copy_empty(self):
        app = harness()
        app._lines = []
        TranscriberApp._on_copy(app)
        assert app.status == "Nada para copiar"

    def test_copy_joins_lines(self):
        app = harness()
        copied = {}
        app.clipboard_clear = lambda: None
        app.clipboard_append = lambda body: copied.setdefault("b", body)
        TranscriberApp._on_copy(app)
        assert copied["b"] == "[10:00:01] Olá\n[10:00:08] Ação"
        assert app.status == "Transcrição copiada"

    def test_save_empty(self):
        app = harness()
        app._lines = []
        TranscriberApp._on_save(app)
        assert app.status == "Nada para salvar"
