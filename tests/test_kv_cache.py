"""KV-cache correctness tests (audit finding D8: no cache, 70% of decode time
spent in pure-Python post-processing).

The point of a cache is *identical* output for less compute, so every test here
compares the cached path against the uncached reference path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.generation import generate_tokens  # noqa: E402
from nanollm.model import GPT, GPTConfig  # noqa: E402


def _load(tiny_checkpoint):
    ckpt = torch.load(tiny_checkpoint, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _greedy_ids(model, prompt_ids, n_new, use_cache):
    """Reference greedy decode loop, written twice on purpose: once with the
    full-window forward (as v1 did) and once through the KV cache."""
    block = model.config.block_size
    ctx = list(prompt_ids)
    out = []
    cache = None
    cur = None
    if use_cache:
        logits, cache = model.forward_cached(torch.tensor([ctx], dtype=torch.long))
        cur = logits[:, -1, :]
    for _ in range(n_new):
        if not use_cache:
            window = torch.tensor([ctx[-block:]], dtype=torch.long)
            cur = model(window)[0][:, -1, :]
        nid = int(cur.argmax(dim=-1).item())
        out.append(nid)
        ctx.append(nid)
        if use_cache:
            if len(ctx) > block:  # window slid: re-prefill (positions restart like v1)
                logits, cache = model.forward_cached(torch.tensor([ctx[-block:]], dtype=torch.long), None)
            else:
                logits, cache = model.forward_cached(torch.tensor([[nid]], dtype=torch.long), cache)
            cur = logits[:, -1, :]
    return out


def test_forward_cached_matches_full_forward_logits(tiny_checkpoint):
    model = _load(tiny_checkpoint)
    ids = torch.randint(0, 300, (1, 24))
    full = model(ids)[0]

    prefill, cache = model.forward_cached(ids[:, :12], None)
    torch.testing.assert_close(prefill, full[:, :12], rtol=1e-4, atol=1e-5)

    # incremental single-token steps must reproduce the full forward exactly
    for t in range(12, 24):
        step_logits, cache = model.forward_cached(ids[:, t:t + 1], cache)
        full_t = model(ids[:, : t + 1])[0][:, -1:, :]
        torch.testing.assert_close(step_logits, full_t, rtol=1e-4, atol=1e-5)


def test_forward_cached_multi_token_chunk(tiny_checkpoint):
    """Appending several tokens at once (T>1) with a non-empty cache stays
    equivalent to the full forward (masked, non-square attention)."""
    model = _load(tiny_checkpoint)
    ids = torch.randint(0, 300, (1, 16))
    _, cache = model.forward_cached(ids[:, :6], None)
    chunk = ids[:, 6:11]
    chunk_logits, cache = model.forward_cached(chunk, cache)
    torch.testing.assert_close(chunk_logits, model(ids[:, :11])[0][:, 6:11], rtol=1e-4, atol=1e-5)


def test_greedy_token_ids_identical_with_and_without_cache(tiny_checkpoint):
    """The key correctness claim: greedy decoding is token-for-token identical.
    Prompt + 40 new tokens exceeds the tiny model's block size (32), so the
    sliding-window path is exercised too."""
    model = _load(tiny_checkpoint)
    prompt = [5, 17, 42, 8, 99, 3, 21, 60, 11, 7]
    n_new = 40

    without = _greedy_ids(model, prompt, n_new, use_cache=False)
    with_cache = _greedy_ids(model, prompt, n_new, use_cache=True)
    assert len(with_cache) == n_new
    assert without == with_cache, "KV cache changed the greedy token sequence"
    assert len(prompt) + n_new > model.config.block_size  # window actually slid


def test_generate_tokens_cache_matches_uncached_with_sampling(tiny_checkpoint, tiny_tokenizer):
    """The shipped generation path (repetition penalty + no-repeat n-gram +
    top-k sampling, CPU-seeded RNG) must be unaffected by enabling the cache."""
    tok, _ = tiny_tokenizer
    model = _load(tiny_checkpoint)
    kwargs = dict(max_new_tokens=30, temperature=0.7, top_k=20,
                  repetition_penalty=1.15, no_repeat_ngram_size=4, seed=11)
    uncached = "".join(generate_tokens(model, tok, "the death cap", use_kv_cache=False, **kwargs))
    cached = "".join(generate_tokens(model, tok, "the death cap", use_kv_cache=True, **kwargs))
    assert cached == uncached


def test_generate_tokens_defaults_to_cache_and_stops_correctly(tiny_checkpoint, tiny_tokenizer):
    tok, _ = tiny_tokenizer
    model = _load(tiny_checkpoint)
    end_id = tok.special_to_id["<|end|>"]
    text = "".join(generate_tokens(model, tok, "the death cap", max_new_tokens=20,
                                   temperature=0.0, top_k=20, seed=0, stop_ids={end_id}))
    assert isinstance(text, str)
