"""Shared fixtures: a tiny tokenizer + tiny trained checkpoint for fast tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.tokenizer import BPETokenizer  # noqa: E402

TINY_TEXT = (
    "The death cap is deadly poisonous. The chanterelle is edible. "
    "Morels must be cooked. Never eat an unknown mushroom. "
    "Take a spore print to identify mushrooms. "
) * 40


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    t = BPETokenizer(vocab_size=400)
    t.train([TINY_TEXT])
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    t.save(path)
    return t, path


@pytest.fixture(scope="session")
def tiny_bin_data(tmp_path_factory, tiny_tokenizer):
    tok, tok_path = tiny_tokenizer
    d = tmp_path_factory.mktemp("data")
    ids = np.array(tok.encode(TINY_TEXT), dtype=np.uint16)
    ids[:-50].tofile(d / "train.bin")
    ids[-50:].tofile(d / "val.bin")
    return d, tok_path


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, tiny_bin_data):
    """Train a tiny GPT for ~40 steps on CPU; returns the checkpoint path."""
    from nanollm.train import main as train_main

    data_dir, tok_path = tiny_bin_data
    out = tmp_path_factory.mktemp("ckpt") / "out"
    args = [
        "--data-dir", str(data_dir),
        "--out-dir", str(out),
        "--tokenizer", str(tok_path),
        "--device", "cpu",
        "--dtype", "fp32",
        "--vocab-size", "400",
        "--block-size", "32",
        "--n-layer", "2",
        "--n-head", "2",
        "--n-embd", "32",
        "--batch-size", "4",
        "--max-steps", "40",
        "--eval-interval", "10",
        "--eval-iters", "2",
        "--log-interval", "5",
        "--sample-interval", "100000",
        "--lr", "3e-3",
        "--warmup-steps", "5",
        "--max-minutes", "30",
    ]
    train_main(args)
    return out / "latest.ckpt"


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
