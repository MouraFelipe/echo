"""Stub de tkinter para Linux CI — o produto real usa Tk nativo no Windows."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


class _W:
    END = "end"
    DISABLED = "disabled"
    NORMAL = "normal"
    BOTH = "both"
    X = "x"
    Y = "y"
    LEFT = "left"
    RIGHT = "right"
    FLAT = "flat"
    WORD = "word"
    TclError = RuntimeError

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


def install() -> None:
    if "tkinter" in sys.modules and getattr(sys.modules["tkinter"], "__file__", None):
        try:
            import tkinter as tk  # noqa: F401

            return
        except ImportError:
            pass

    tk = ModuleType("tkinter")
    for name, value in _W.__dict__.items():
        if name.startswith("_"):
            continue
        setattr(tk, name, value)
    tk.Tk = _W
    tk.Frame = _W
    tk.Text = _W
    tk.Canvas = _W
    tk.StringVar = lambda *a, **k: SimpleNamespace(get=lambda: "", set=lambda *_: None)
    ttk = ModuleType("tkinter.ttk")
    ttk.Style = _W
    ttk.Frame = _W
    ttk.Label = _W
    ttk.Button = _W
    ttk.Combobox = _W
    ttk.Entry = _W
    ttk.Scrollbar = _W
    filedialog = ModuleType("tkinter.filedialog")
    filedialog.asksaveasfilename = lambda **k: ""
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk
    sys.modules["tkinter.filedialog"] = filedialog


install()
