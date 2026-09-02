"""Roda antes do app no .exe: stdout nulo (windowed) e cache persistente do modelo."""

from __future__ import annotations

import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

appdata = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
os.environ.setdefault("HF_HOME", os.path.join(appdata, "Voxa", "hf"))
