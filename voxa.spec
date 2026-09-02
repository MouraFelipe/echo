# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onefile — gera dist/Voxa.exe no Windows."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas: list = []
binaries: list = []
hiddenimports = [
    "audio_capture",
    "transcriber",
    "utils",
    "assistant",
    "pyaudiowpatch",
    "numpy",
    "scipy",
    "scipy.signal",
    "scipy.signal._signaltools",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "av",
    "onnxruntime",
    "huggingface_hub",
]

for pkg in (
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "av",
    "onnxruntime",
    "huggingface_hub",
    "pyaudiowpatch",
    "scipy",
):
    try:
        collected_datas, collected_binaries, collected_hidden = collect_all(pkg)
        datas += collected_datas
        binaries += collected_binaries
        hiddenimports += collected_hidden
    except Exception:
        pass

try:
    binaries += collect_dynamic_libs("ctranslate2")
except Exception:
    pass

runtime_hook = str(Path("packaging") / "runtime_hook.py")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[runtime_hook],
    excludes=["pytest", "IPython", "matplotlib", "tkinter.test"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Voxa",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
