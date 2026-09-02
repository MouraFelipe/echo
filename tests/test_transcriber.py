from __future__ import annotations

import numpy as np
import pytest

from transcriber import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODELS,
    TranscribeError,
    Transcriber,
    WordStamp,
    dedupe_by_time,
    join_words,
)
from tests.conftest import FakeWhisper, FakeWord, sine


def words(*pairs: tuple[str, float, float]) -> list[WordStamp]:
    return [WordStamp(word=w, start=s, end=e) for w, s, e in pairs]


class TestJoinWords:
    def test_empty(self):
        assert join_words([]) == ""

    def test_space_prefixed_tokens_from_whisper(self):
        text = join_words(words(("Hello", 0, 0.2), (" world", 0.2, 0.4), ("!", 0.4, 0.5)))
        assert text == "Hello world!"

    def test_punctuation_without_space(self):
        text = join_words(words(("ok", 0, 0.1), (".", 0.1, 0.2), (" Sim", 0.2, 0.4)))
        assert text == "ok. Sim"

    def test_accents(self):
        assert "Ação" in join_words(words(("Ação", 0, 0.3)))


class TestDedupeByTime:
    def test_first_chunk_keeps_all(self):
        src = words(("a", 0.1, 0.2), ("b", 0.4, 0.5), ("c", 1.6, 1.7))
        assert dedupe_by_time(src, is_first_chunk=True, overlap_seconds=1.5) == src

    def test_drops_overlap_window(self):
        src = words(("old", 0.4, 0.6), ("edge", 1.47, 1.6), ("new", 1.55, 1.8))
        kept = dedupe_by_time(src, is_first_chunk=False, overlap_seconds=1.5)
        assert [w.word for w in kept] == ["new"]

    def test_zero_overlap_keeps_all(self):
        src = words(("a", 0.1, 0.2))
        assert dedupe_by_time(src, is_first_chunk=False, overlap_seconds=0) == src


class TestValidate:
    def test_language_default_and_supported(self):
        t = Transcriber()
        assert t.language == "pt"
        t.set_language("EN")
        assert t.language == "en"
        t.set_language("auto")
        assert t.language == "auto"

    def test_language_rejects_unknown(self):
        t = Transcriber()
        with pytest.raises(TranscribeError, match="Idioma não suportado"):
            t.set_language("zh")

    def test_model_rejects_medium(self):
        t = Transcriber()
        with pytest.raises(TranscribeError, match="Modelo não suportado"):
            t.set_model_size("medium")

    def test_model_change_unloads(self):
        t = Transcriber(model_size="tiny")
        t._model = object()
        t.from_local_cache = True
        t.set_model_size("base")
        assert t._model is None
        assert t.from_local_cache is False
        assert t.model_size == "base"

    def test_same_model_is_noop(self):
        t = Transcriber(model_size="base")
        sentinel = object()
        t._model = sentinel
        t.set_model_size("base")
        assert t._model is sentinel

    def test_supported_tables(self):
        assert "pt" in SUPPORTED_LANGUAGES
        assert SUPPORTED_MODELS == ("tiny", "base", "small")


class TestTranscribeBehavior:
    def test_short_audio_skips_model(self):
        t = Transcriber()
        fake = FakeWhisper(words=[FakeWord("oi", 0, 0.2)])
        t._model = fake
        out = t.transcribe(np.zeros(int(0.1 * 16000), dtype=np.float32))
        assert out.text == ""
        assert out.words == []
        assert fake.heard == []

    def test_peak_normalizes_to_0_95(self):
        t = Transcriber()
        fake = FakeWhisper()
        t._model = fake
        audio = np.array([0.1, -0.2, 0.05], dtype=np.float32)
        audio = np.pad(audio, (0, 4000))
        t.transcribe(audio)
        heard = fake.heard[0]
        assert float(np.max(np.abs(heard))) == pytest.approx(0.95, rel=1e-5)

    def test_near_silence_is_amplified_current_behavior(self):
        """Possível defeito: pico 1e-5 ainda passa do corte 1e-6 e vira quase 0.95."""
        t = Transcriber()
        fake = FakeWhisper()
        t._model = fake
        audio = np.full(4000, 1e-5, dtype=np.float32)
        t.transcribe(audio)
        assert float(np.max(np.abs(fake.heard[0]))) == pytest.approx(0.95, rel=1e-4)

    def test_auto_language_sends_none(self):
        t = Transcriber(language="auto")
        fake = FakeWhisper()
        t._model = fake
        t.transcribe(sine(0.5))
        assert fake.kwargs[0]["language"] is None
        assert fake.kwargs[0]["word_timestamps"] is True
        assert fake.kwargs[0]["vad_filter"] is True
        assert fake.kwargs[0]["condition_on_previous_text"] is False

    def test_pt_language_forwarded(self):
        t = Transcriber(language="pt")
        fake = FakeWhisper()
        t._model = fake
        t.transcribe(sine(0.5))
        assert fake.kwargs[0]["language"] == "pt"

    def test_words_joined(self):
        t = Transcriber()
        fake = FakeWhisper([FakeWord("Olá", 0, 0.2), FakeWord(" mundo", 0.2, 0.4)])
        t._model = fake
        out = t.transcribe(sine(0.5))
        assert out.text == "Olá mundo"
        assert len(out.words) == 2

    def test_model_exception_becomes_transcribe_error(self):
        t = Transcriber()

        class Boom:
            def transcribe(self, *a, **k):
                raise RuntimeError("cuda died")

        t._model = Boom()
        with pytest.raises(TranscribeError, match="Falha ao transcrever"):
            t.transcribe(sine(0.5))

    def test_missing_faster_whisper(self, monkeypatch):
        t = Transcriber()
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(TranscribeError, match="faster-whisper não está instalado"):
            t.ensure_loaded()
