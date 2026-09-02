"""Voxa — transcritor de áudio de sistema (PyAudioWPatch loopback + faster-whisper)."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

from audio_capture import CaptureError, capture_loop
from assistant import AssistantError, run_assistant
from transcriber import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_MODELS,
    Transcriber,
    TranscribeError,
    dedupe_by_time,
    join_words,
)
from utils import (
    CHUNK_SECONDS,
    OVERLAP_SECONDS,
    DeviceError,
    LoopbackDevice,
    default_loopback_device,
    format_elapsed,
    format_line,
    list_loopback_devices,
    now_clock,
    save_transcript,
)

APP_TITLE = "Voxa  ·  Transcritor de áudio de sistema"
POLL_MS = 80

BG = "#0b0c0b"
SURFACE = "#141512"
SURFACE_2 = "#1c1d19"
FG = "#ecece4"
MUTED = "#9a9a90"
LINE = "#2a2b27"
ACCENT = "#d8d4c8"
LIVE = "#8a9a8c"
DANGER = "#c45c4a"


class TranscriberApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("920x760")
        self.minsize(760, 620)
        self.configure(bg=BG)

        self._events: queue.Queue[tuple] = queue.Queue()
        self._audio_q: queue.Queue[tuple] = queue.Queue(maxsize=3)
        self._running = False
        self._started_at = 0.0
        self._lines: list[str] = []
        self._busy_model = False
        self._stop_event = threading.Event()
        self._devices: list[LoopbackDevice] = []

        self.transcriber = Transcriber(model_size="base", language="pt")

        self._build_style()
        self._build_ui()
        self._refresh_devices()
        threading.Thread(target=self._transcribe_loop, name="whisper-worker", daemon=True).start()
        self.after(POLL_MS, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick_elapsed()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground=BG,
            font=("Segoe UI Semibold", 10),
            padding=(16, 8),
            borderwidth=0,
        )
        style.map("Accent.TButton", background=[("disabled", "#6f6d66"), ("active", "#f0ece0")])
        style.configure(
            "Ghost.TButton",
            background=SURFACE_2,
            foreground=FG,
            padding=(14, 8),
            borderwidth=0,
        )
        style.map("Ghost.TButton", background=[("disabled", SURFACE), ("active", "#262722")])
        style.configure(
            "TCombobox",
            fieldbackground=SURFACE_2,
            background=SURFACE_2,
            foreground=FG,
            arrowcolor=FG,
            padding=4,
        )
        style.map("TCombobox", fieldbackground=[("readonly", SURFACE_2)], foreground=[("readonly", FG)])
        style.configure("TScrollbar", background=SURFACE, troughcolor=BG, bordercolor=BG, arrowcolor=MUTED)
        self.option_add("*TCombobox*Listbox.background", SURFACE_2)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", LIVE)
        self.option_add("*TCombobox*Listbox.selectForeground", BG)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=18)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Voxa", font=("Segoe UI Semibold", 22)).pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="  loopback WASAPI · 16 kHz · offline depois do 1º download",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, pady=(8, 0))

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(16, 8))

        self.btn_start = ttk.Button(controls, text="Iniciar", style="Accent.TButton", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(
            controls, text="Parar", style="Ghost.TButton", command=self._on_stop, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 12))
        ttk.Button(controls, text="Diagnosticar", style="Ghost.TButton", command=self._refresh_devices).pack(
            side=tk.LEFT
        )

        row2 = ttk.Frame(outer)
        row2.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(row2, text="Loopback", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.cmb_device = ttk.Combobox(row2, state="readonly", width=42)
        self.cmb_device.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(row2, text="Idioma", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 6))
        lang_values = [f"{code}  —  {label}" for code, label in SUPPORTED_LANGUAGES.items()]
        self.cmb_lang = ttk.Combobox(row2, state="readonly", width=22, values=lang_values)
        self.cmb_lang.set("pt  —  Português")
        self.cmb_lang.pack(side=tk.LEFT)
        self.cmb_lang.bind("<<ComboboxSelected>>", self._on_language_change)

        ttk.Label(row2, text="Modelo", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 6))
        self.cmb_model = ttk.Combobox(row2, state="readonly", width=8, values=list(SUPPORTED_MODELS))
        self.cmb_model.set("base")
        self.cmb_model.pack(side=tk.LEFT)
        self.cmb_model.bind("<<ComboboxSelected>>", self._on_model_change)

        meter_row = ttk.Frame(outer)
        meter_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(meter_row, text="Nível", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self.level = tk.Canvas(meter_row, width=220, height=10, bg=SURFACE_2, highlightthickness=0)
        self.level.pack(side=tk.LEFT)
        self._level_bar = self.level.create_rectangle(0, 0, 0, 10, fill=LIVE, width=0)
        ttk.Label(
            meter_row,
            text=f"chunk {CHUNK_SECONDS:.0f}s  ·  overlap {OVERLAP_SECONDS}s  ·  hop {CHUNK_SECONDS - OVERLAP_SECONDS}s",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(14, 0))

        card = tk.Frame(outer, bg=SURFACE, highlightbackground=LINE, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(
            card,
            wrap=tk.WORD,
            bg=SURFACE,
            fg=FG,
            insertbackground=FG,
            relief=tk.FLAT,
            padx=16,
            pady=14,
            font=("Consolas", 11),
            state=tk.DISABLED,
            highlightthickness=0,
            spacing3=8,
        )
        scroll = ttk.Scrollbar(card, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.tag_configure("ts", foreground=LIVE)
        self.text.tag_configure("placeholder", foreground=MUTED)

        self._set_placeholder(
            "1. Clique em Diagnosticar e escolha o loopback (não o microfone).\n"
            "2. Iniciar. Reproduza um vídeo — o app captura o que o Windows está tocando.\n"
            "3. Depois use Resumir / Tarefas / Perguntar — assistente local, sem nuvem."
        )

        assist = ttk.Frame(outer)
        assist.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(assist, text="Assistente (local / Ollama se existir)", style="Muted.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Button(assist, text="Resumir", style="Ghost.TButton", command=lambda: self._ask_ai("resumo")).pack(
            side=tk.RIGHT
        )
        ttk.Button(assist, text="Tarefas", style="Ghost.TButton", command=lambda: self._ask_ai("tarefas")).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        ask_row = ttk.Frame(outer)
        ask_row.pack(fill=tk.X, pady=(8, 0))
        self.ask_var = tk.StringVar()
        self.ask_entry = ttk.Entry(ask_row, textvariable=self.ask_var)
        self.ask_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ask_entry.bind("<Return>", lambda _e: self._ask_ai("pergunta"))
        ttk.Button(ask_row, text="Perguntar", style="Ghost.TButton", command=lambda: self._ask_ai("pergunta")).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        assist_card = tk.Frame(outer, bg=SURFACE, highlightbackground=LINE, highlightthickness=1)
        assist_card.pack(fill=tk.X, pady=(8, 0))
        self.assist_text = tk.Text(
            assist_card,
            wrap=tk.WORD,
            bg=SURFACE,
            fg=FG,
            height=7,
            relief=tk.FLAT,
            padx=14,
            pady=10,
            font=("Segoe UI", 10),
            state=tk.DISABLED,
            highlightthickness=0,
        )
        self.assist_text.pack(fill=tk.X)
        self._set_assist("O assistente é gratuito e local. Se o Ollama estiver aberto neste PC, ele entra no lugar do resumidor.")

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(12, 0))
        self.status_var = tk.StringVar(value="Pronto  ·  offline após o 1º download do modelo")
        self.elapsed_var = tk.StringVar(value="00:00:00")
        self.status_label = ttk.Label(footer, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        ttk.Label(footer, textvariable=self.elapsed_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(14, 0))

        ttk.Button(footer, text="Limpar", style="Ghost.TButton", command=self._on_clear).pack(side=tk.RIGHT)
        ttk.Button(footer, text="Salvar", style="Ghost.TButton", command=self._on_save).pack(
            side=tk.RIGHT, padx=(0, 8)
        )
        ttk.Button(footer, text="Copiar", style="Ghost.TButton", command=self._on_copy).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

    def _refresh_devices(self) -> None:
        try:
            self._devices = list_loopback_devices()
            default = default_loopback_device(self._devices)
        except DeviceError as exc:
            self._devices = []
            self.cmb_device.set("")
            self.cmb_device.configure(values=[])
            self._set_status(str(exc), error=True)
            return

        labels = [device.label for device in self._devices]
        self.cmb_device.configure(values=labels)
        self.cmb_device.set(default.label)
        self._set_status(f"{len(self._devices)} loopback(s) · padrão: {default.name}")

    def _selected_device(self) -> LoopbackDevice | None:
        label = self.cmb_device.get()
        for device in self._devices:
            if device.label == label:
                return device
        return self._devices[0] if self._devices else None

    def _on_start(self) -> None:
        if self._running or self._busy_model:
            return
        device = self._selected_device()
        if device is None:
            self._refresh_devices()
            device = self._selected_device()
        if device is None:
            self._set_status("Nenhum loopback disponível. Clique em Diagnosticar.", error=True)
            return

        model = self.cmb_model.get().strip() or "base"
        language = self.cmb_lang.get().split("—")[0].strip() or "pt"
        self.transcriber.set_model_size(model)
        self.transcriber.set_language(language)
        self._busy_model = True
        self._stop_event = threading.Event()
        self._set_buttons(active=False, starting=True)
        self._set_status("Carregando modelo…")

        def boot() -> None:
            try:
                self.transcriber.ensure_loaded(on_status=lambda msg: self._events.put(("status", msg)))
                self._running = True
                threading.Thread(
                    target=capture_loop,
                    kwargs={
                        "device": device,
                        "stop_event": self._stop_event,
                        "on_chunk": self._on_chunk,
                        "on_error": lambda msg: self._events.put(("error", msg)),
                        "on_level": lambda rms: self._events.put(("level", rms)),
                    },
                    name="loopback-capture",
                    daemon=True,
                ).start()
                self._events.put(("started", device.label))
            except (CaptureError, DeviceError, TranscribeError) as exc:
                self._running = False
                self._events.put(("error", str(exc)))
            except Exception as exc:
                self._running = False
                self._events.put(("error", f"Não foi possível iniciar: {exc}"))
            finally:
                self._events.put(("boot_done", None))

        threading.Thread(target=boot, name="voxa-boot", daemon=True).start()

    def _on_stop(self) -> None:
        self._running = False
        self._stop_event.set()
        self._set_buttons(active=False)
        self._set_status("Parado")
        self._draw_level(0.0)
        if self._lines:
            try:
                saved = save_transcript("\n".join(self._lines))
                self._set_status(f"Parado  ·  salvo em {saved}")
            except Exception:
                pass

    def _on_language_change(self, _event: object | None = None) -> None:
        code = self.cmb_lang.get().split("—")[0].strip()
        try:
            self.transcriber.set_language(code)
            self._set_status(f"Idioma: {SUPPORTED_LANGUAGES.get(code, code)}")
        except TranscribeError as exc:
            self._set_status(str(exc), error=True)

    def _on_model_change(self, _event: object | None = None) -> None:
        if self._running:
            self.cmb_model.set(self.transcriber.model_size)
            self._set_status("Pare a captura para trocar o modelo.", error=True)
            return
        try:
            self.transcriber.set_model_size(self.cmb_model.get())
            self._set_status(f"Modelo {self.transcriber.model_size} será carregado no próximo Iniciar")
        except TranscribeError as exc:
            self._set_status(str(exc), error=True)

    def _on_clear(self) -> None:
        self._lines.clear()
        self._set_placeholder("Transcrição limpa. Clique em Iniciar para continuar.")

    def _on_copy(self) -> None:
        body = "\n".join(self._lines).strip()
        if not body:
            self._set_status("Nada para copiar")
            return
        self.clipboard_clear()
        self.clipboard_append(body)
        self._set_status("Transcrição copiada")

    def _on_save(self) -> None:
        body = "\n".join(self._lines).strip()
        if not body:
            self._set_status("Nada para salvar")
            return
        suggested = f"transcricao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            title="Salvar transcrição",
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt"), ("Todos", "*.*")],
            initialfile=suggested,
        )
        if not path:
            return
        try:
            saved = save_transcript(body, Path(path))
        except Exception as exc:
            self._set_status(f"Falha ao salvar: {exc}", error=True)
            return
        self._set_status(f"Salvo em {saved}")

    def _ask_ai(self, task: str) -> None:
        body = "\n".join(self._lines).strip()
        question = self.ask_var.get().strip()
        self._set_assist("Pensando…")
        self._set_status("Assistente trabalhando…")

        def worker() -> None:
            try:
                text = run_assistant(task, body, question)
                self._events.put(("assist", text))
            except AssistantError as exc:
                self._events.put(("assist_error", str(exc)))
            except Exception as exc:
                self._events.put(("assist_error", f"Falha no assistente: {exc}"))

        threading.Thread(target=worker, name="voxa-assistant", daemon=True).start()

    def _set_assist(self, text: str, error: bool = False) -> None:
        self.assist_text.configure(state=tk.NORMAL)
        self.assist_text.delete("1.0", tk.END)
        self.assist_text.insert("1.0", text)
        self.assist_text.configure(fg=DANGER if error else FG, state=tk.DISABLED)

    def _on_close(self) -> None:
        self._running = False
        self._stop_event.set()
        self.destroy()

    def _on_chunk(self, audio, is_first: bool, overlap: float) -> None:
        if not self._running and not self._busy_model:
            return
        item = (audio, is_first, overlap)
        try:
            self._audio_q.put_nowait(item)
        except queue.Full:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_q.put_nowait(item)
            except queue.Full:
                pass
        self._events.put(("status", "Transcrevendo…"))

    def _transcribe_loop(self) -> None:
        while True:
            audio, is_first, overlap = self._audio_q.get()
            try:
                result = self.transcriber.transcribe(audio)
                words = dedupe_by_time(result.words, is_first_chunk=is_first, overlap_seconds=overlap)
                text = join_words(words)
                if text:
                    self._events.put(("line", format_line(now_clock(), text)))
                elif self._running:
                    self._events.put(("status", "Capturando áudio do sistema"))
            except TranscribeError as exc:
                self._events.put(("error", str(exc)))
            except Exception as exc:
                self._events.put(("error", f"Erro na transcrição: {exc}"))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "line":
                    self._append_line(str(payload))
                    if self._running:
                        self._set_status("Capturando áudio do sistema  ·  latência ≈ 8–14 s")
                elif kind == "status":
                    self._set_status(str(payload))
                elif kind == "level":
                    self._draw_level(float(payload))
                elif kind == "started":
                    self._running = True
                    self._started_at = time.monotonic()
                    self._set_buttons(active=True)
                    cache = "cache local" if self.transcriber.from_local_cache else "1º download"
                    self._set_status(
                        f"Capturando {payload}  ·  {self.transcriber.device}/{self.transcriber.compute_type}  ·  {cache}"
                    )
                elif kind == "boot_done":
                    self._busy_model = False
                    if not self._running:
                        self._set_buttons(active=False)
                elif kind == "assist":
                    self._set_assist(str(payload))
                    self._set_status("Assistente pronto")
                elif kind == "assist_error":
                    self._set_assist(str(payload), error=True)
                    self._set_status(str(payload), error=True)
                elif kind == "error":
                    self._running = False
                    self._busy_model = False
                    self._stop_event.set()
                    self._set_buttons(active=False)
                    self._draw_level(0.0)
                    self._set_status(str(payload), error=True)
        except queue.Empty:
            pass
        self.after(POLL_MS, self._drain_events)

    def _tick_elapsed(self) -> None:
        if self._running and self._started_at:
            self.elapsed_var.set(format_elapsed(time.monotonic() - self._started_at))
        self.after(250, self._tick_elapsed)

    def _set_buttons(self, active: bool, starting: bool = False) -> None:
        if starting:
            self.btn_start.configure(state=tk.DISABLED)
            self.btn_stop.configure(state=tk.DISABLED)
            self.cmb_model.configure(state="disabled")
            self.cmb_device.configure(state="disabled")
            return
        self.btn_start.configure(state=tk.DISABLED if active else tk.NORMAL)
        self.btn_stop.configure(state=tk.NORMAL if active else tk.DISABLED)
        self.cmb_model.configure(state="disabled" if active else "readonly")
        self.cmb_device.configure(state="disabled" if active else "readonly")

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_var.set(message)
        self.status_label.configure(foreground=DANGER if error else FG)

    def _set_placeholder(self, text: str) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text, ("placeholder",))
        self.text.configure(state=tk.DISABLED)

    def _append_line(self, line: str) -> None:
        if not self._lines:
            self.text.configure(state=tk.NORMAL)
            self.text.delete("1.0", tk.END)
            self.text.configure(state=tk.DISABLED)
        self._lines.append(line)
        self.text.configure(state=tk.NORMAL)
        stamp, _, body = line.partition("] ")
        if line.startswith("[") and body:
            self.text.insert(tk.END, stamp + "] ", ("ts",))
            self.text.insert(tk.END, body + "\n")
        else:
            self.text.insert(tk.END, line + "\n")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def _draw_level(self, rms: float) -> None:
        width = max(0, min(220, int(rms * 1400)))
        self.level.coords(self._level_bar, 0, 0, width, 10)


def main() -> None:
    import multiprocessing

    multiprocessing.freeze_support()

    def _hook(args: threading.ExceptHookArgs) -> None:
        err = sys.stderr
        if err is None:
            return
        err.write(f"thread {args.thread}: {args.exc_type.__name__}: {args.exc_value}\n")

    threading.excepthook = _hook
    app = TranscriberApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
