"""Checkpoint loading must not execute arbitrary pickle payloads.

``e57bd9c`` switched ``train.py`` to ``weights_only=True`` but left a second
call site at ``scripts/evaluate.py`` (and the LoRA/merge/chat/server paths).
These tests pin the rule for the whole repository.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.checkpoints import load_checkpoint  # noqa: E402


class _Evil:
    """A pickle payload that runs a shell command when unpickled."""

    def __reduce__(self):
        return (os.system, ("echo pwned > pwned.txt",))


def test_load_checkpoint_rejects_arbitrary_pickle(tmp_path):
    path = tmp_path / "malicious.pt"
    torch.save({"model": {}, "payload": _Evil()}, path)
    with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError, EOFError)):
        load_checkpoint(path, map_location="cpu")


def test_load_checkpoint_accepts_a_plain_tensor_checkpoint(tmp_path):
    path = tmp_path / "ok.pt"
    torch.save({"model": {"w": torch.zeros(2)}, "step": 3, "best_val": 1.5,
                "config": {"n_layer": 1}, "tokenizer_path": "t.json"}, path)
    ckpt = load_checkpoint(path, map_location="cpu")
    assert ckpt["step"] == 3
    assert torch.equal(ckpt["model"]["w"], torch.zeros(2))


def test_no_call_site_loads_checkpoints_without_weights_only():
    """No ``torch.load`` in the project may pass ``weights_only=False``."""
    offenders = []
    this_file = Path(__file__).resolve()
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".git", "out", ".venv", "build"} for part in path.parts):
            continue
        if path.resolve() == this_file:
            continue  # this scanner necessarily names the forbidden pattern
        text = path.read_text(encoding="utf-8", errors="replace")
        if "weights_only=False" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"unsafe checkpoint loads: {offenders}"
