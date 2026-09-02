"""Voxa — transcritor de áudio de sistema (WASAPI loopback + faster-whisper)."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

from audio_capture import CaptureError, LoopbackCapture
from transcriber import SUPPORTED_LANGUAGES, SUPPORTED_MODELS, LocalTranscriber, TranscribeError
from utils import format_elapsed, format_line, now_clock, save_transcript

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
        self.geometry("860x620")
        self.minsize(720, 520)
        self.configure(bg=BG)

        self._events: queue.Queue[tuple] = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue(maxsize=8)
        self._running = False
        self._started_at = 0.0
        self._lines: list[str] = []
        self._busy_model = False

        self.transcriber = LocalTranscriber(model_size="base", language="pt")
        self.capture = LoopbackCapture(
            on_segment=self._on_segment,
            on_error=self._on_capture_error,
            on_level=lambda rms: self._events.put(("level", rms)),
            on_device=lambda label: self._events.put(("device", label)),
        )

        self._build_style()
        self._build_ui()
        self._worker = threading.Thread(
            target=self._transcribe_loop,
            name="whisper-worker",
            daemon=True,
        )
        self._worker.start()
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
            text="  captura o que o Windows está tocando, transcreve localmente",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, pady=(8, 0))

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(16, 10))

        self.btn_start = ttk.Button(controls, text="Iniciar", style="Accent.TButton", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(
            controls, text="Parar", style="Ghost.TButton", command=self._on_stop, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 18))

        ttk.Label(controls, text="Idioma", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        lang_values = [f"{code}  —  {label}" for code, label in SUPPORTED_LANGUAGES.items()]
        self.cmb_lang = ttk.Combobox(controls, state="readonly", width=26, values=lang_values)
        self.cmb_lang.set("pt  —  Português")
        self.cmb_lang.pack(side=tk.LEFT)
        self.cmb_lang.bind("<<ComboboxSelected>>", self._on_language_change)

        ttk.Label(controls, text="Modelo", style="Muted.TLabel").pack(side=tk.LEFT, padx=(16, 6))
        self.cmb_model = ttk.Combobox(controls, state="readonly", width=10, values=list(SUPPORTED_MODELS))
        self.cmb_model.set("base")
        self.cmb_model.pack(side=tk.LEFT)
        self.cmb_model.bind("<<ComboboxSelected>>", self._on_model_change)

        self.device_var = tk.StringVar(value="Dispositivo: aguardando início")
        ttk.Label(outer, textvariable=self.device_var, style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 8))

        meter_row = ttk.Frame(outer)
        meter_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(meter_row, text="Nível", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self.level = tk.Canvas(meter_row, width=220, height=10, bg=SURFACE_2, highlightthickness=0)
        self.level.pack(side=tk.LEFT)
        self._level_bar = self.level.create_rectangle(0, 0, 0, 10, fill=LIVE, width=0)

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
            "Clique em Iniciar para capturar o áudio do sistema.\n"
            "Reproduza um vídeo, uma reunião ou qualquer som no Windows.\n"
            "A transcrição aparece aqui, com horário, sem sair do computador."
        )

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(12, 0))
        self.status_var = tk.StringVar(value="Pronto  ·  offline")
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

    def _on_start(self) -> None:
        if self._running or self._busy_model:
            return
        model = self.cmb_model.get().strip() or "base"
        language = self.cmb_lang.get().split("—")[0].strip() or "pt"
        self.transcriber.set_model_size(model)
        self.transcriber.set_language(language)
        self._busy_model = True
        self._set_buttons(active=False, starting=True)
        self._set_status("Carregando modelo…")

        def boot() -> None:
            try:
                self.transcriber.ensure_loaded(on_status=lambda msg: self._events.put(("status", msg)))
                self._running = True
                self.capture.start()
                self._events.put(("started", None))
            except (CaptureError, TranscribeError) as exc:
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
        try:
            self.capture.stop()
        except Exception:
            pass
        self._set_buttons(active=False)
        self._set_status("Parado")
        self._draw_level(0.0)

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

    def _on_close(self) -> None:
        self._running = False
        try:
            self.capture.stop()
        except Exception:
            pass
        self.destroy()

    def _on_segment(self, audio) -> None:
        if not self._running:
            return
        try:
            self._audio_q.put_nowait(audio)
            self._events.put(("status", "Transcrevendo…"))
        except queue.Full:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_q.put_nowait(audio)
            except queue.Full:
                pass

    def _on_capture_error(self, message: str) -> None:
        self._events.put(("error", message))

    def _transcribe_loop(self) -> None:
        while True:
            audio = self._audio_q.get()
            if audio is None:
                return
            try:
                text = self.transcriber.transcribe(audio)
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
                        self._set_status("Capturando áudio do sistema")
                elif kind == "status":
                    self._set_status(str(payload))
                elif kind == "device":
                    self.device_var.set(f"Dispositivo: {payload}")
                elif kind == "level":
                    self._draw_level(float(payload))
                elif kind == "started":
                    self._running = True
                    self._started_at = time.monotonic()
                    self._set_buttons(active=True)
                    runtime = f"{self.transcriber.device}/{self.transcriber.compute_type}"
                    self._set_status(f"Capturando áudio do sistema  ·  {runtime}")
                elif kind == "boot_done":
                    self._busy_model = False
                    if not self._running:
                        self._set_buttons(active=False)
                elif kind == "error":
                    self._running = False
                    self._busy_model = False
                    try:
                        self.capture.stop()
                    except Exception:
                        pass
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
            return
        self.btn_start.configure(state=tk.DISABLED if active else tk.NORMAL)
        self.btn_stop.configure(state=tk.NORMAL if active else tk.DISABLED)
        self.cmb_model.configure(state="disabled" if active else "readonly")

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
    def _hook(args: threading.ExceptHookArgs) -> None:
        sys.stderr.write(f"thread {args.thread}: {args.exc_type.__name__}: {args.exc_value}\n")

    threading.excepthook = _hook
    app = TranscriberApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
