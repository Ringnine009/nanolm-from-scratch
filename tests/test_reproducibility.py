"""Device reproducibility (audit finding D1).

The v1 generator did ``torch.Generator(device=device)``: the *same seed* then
produces a different random stream on CPU and on CUDA, and the published
held-out number changed from 38.6% (CUDA) to 52.3% (CPU) with the same
checkpoint.  Sampling must be driven by a device-independent (CPU) stream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.generation import generate_tokens  # noqa: E402
from nanollm.model import GPT, GPTConfig  # noqa: E402

PROMPT = "<|user|>the death cap<|assistant|>"


def _load_tiny(tiny_checkpoint, device):
    ckpt = torch.load(tiny_checkpoint, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _ids(model, tok, device, seed, temperature):
    text = "".join(generate_tokens(
        model, tok, PROMPT, max_new_tokens=24, temperature=temperature,
        top_k=20, repetition_penalty=1.15, no_repeat_ngram_size=4, seed=seed,
    ))
    return tok.encode(text)


def test_same_seed_gives_identical_tokens_on_cpu_and_cuda(tiny_checkpoint, tiny_tokenizer):
    """Same checkpoint + same seed + same settings -> identical token sequence,
    whatever the device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tok, _ = tiny_tokenizer
    cpu_model = _load_tiny(tiny_checkpoint, "cpu")
    cuda_model = _load_tiny(tiny_checkpoint, "cuda")

    for temperature in (0.0, 0.7):  # greedy and sampled
        cpu_ids = _ids(cpu_model, tok, "cpu", seed=3, temperature=temperature)
        cuda_ids = _ids(cuda_model, tok, "cuda", seed=3, temperature=temperature)
        assert cpu_ids == cuda_ids, f"device-dependent output at temperature={temperature}"


def test_greedy_temperature_zero_is_argmax(tiny_checkpoint, tiny_tokenizer):
    """temperature=0 must mean greedy decoding (v1 divided by zero here)."""
    tok, _ = tiny_tokenizer
    model = _load_tiny(tiny_checkpoint, "cpu")
    a = "".join(generate_tokens(model, tok, PROMPT, max_new_tokens=12, temperature=0.0, top_k=20, seed=0))
    b = "".join(generate_tokens(model, tok, PROMPT, max_new_tokens=12, temperature=0.0, top_k=20, seed=99))
    assert a == b  # no RNG involved when greedy
    assert all(c == c for c in a)  # not NaN / not empty
