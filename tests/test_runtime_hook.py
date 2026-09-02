from __future__ import annotations

import os
from pathlib import Path


def test_runtime_hook_sets_hf_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    hook = Path(__file__).resolve().parents[1] / "packaging" / "runtime_hook.py"
    exec(compile(hook.read_text(encoding="utf-8"), str(hook), "exec"), {})
    assert os.environ["HF_HOME"] == str(tmp_path / "Voxa" / "hf")
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
