"""Safe checkpoint loading.

Every checkpoint this project writes is a plain dict of tensors and basic
Python types (``model``, ``optimizer``, ``step``, ``best_val``, ``config``,
``tokenizer_path``).  Loading one therefore never needs ``pickle``'s ability to
execute arbitrary code - and that ability is exactly what an attacker-supplied
``.pt`` file abuses.  The evaluation scripts kept calling ``torch.load`` with
that safety flag disabled after ``e57bd9c`` hardened the training path; this
module is the single place where checkpoints are read, so the flag cannot be
forgotten again (``tests/test_safe_loading.py`` scans the whole repository for a
disabled flag).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


def load_checkpoint(
    path: str | Path,
    map_location="cpu",
    required: Sequence[str] = (),
) -> dict:
    """Load a project checkpoint with ``weights_only=True``.

    Raises the underlying ``pickle.UnpicklingError`` if the file contains
    anything beyond tensors / basic types, and ``ValueError`` if it is not a
    dict or misses a required key.
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected a dict checkpoint, got {type(ckpt).__name__}")
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise ValueError(f"{path}: checkpoint is missing keys {missing}")
    return ckpt
