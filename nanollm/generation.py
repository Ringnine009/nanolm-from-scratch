"""Unified text generation with repetition control.

Single implementation of decoding used by the CLI chat, the sample tool, the
FastAPI server and the evaluation script.  Features:

- ``repetition_penalty``: penalises tokens that already appear in the context
  (divide positive logits, multiply negative ones, >1.0 = stronger penalty);
- ``no_repeat_ngram_size``: masks any next token that would complete an n-gram
  already seen in the context+generation (blocks loops by construction);
- temperature scaling + top-k filtering (``temperature <= 0`` = greedy argmax);
- **KV cache** (``use_kv_cache=True``): the prompt is prefilled once and every
  later step feeds a single token, so per-step attention work is O(T) instead
  of O(T^2).  The cached and uncached paths produce identical tokens (tested);
- **device-independent sampling**: random numbers come from a *CPU* generator,
  and the draw is turned into a token by inverse-CDF on the (device-side)
  probabilities.  v1 used ``torch.Generator(device=device)``, so one seed gave
  different text on CPU and on CUDA - the published held-out score moved from
  38.6% (CUDA) to 52.3% (CPU) with the same checkpoint;
- single-token decoding with ``<|end|>`` stopping and deterministic seeds;
- post-processing helpers: strip leading punctuation, cut at ``<|end|>``.

The logit post-processing is fully vectorised (v1 looped over the vocabulary in
Python inside ``_no_repeat_mask`` and over ``set(context)`` inside the penalty;
that Python work dominated decode time).
"""

from __future__ import annotations

import codecs
import re
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from nanollm.tokenizer import BPETokenizer

END_MARKER = "<|end|>"
LEADING_PUNCT_RE = re.compile(r"^[\s,.;:!?\-–—'\"“”‘’()\[\]<>/\\…]+")
JUNK_TOKEN_RE = re.compile(r"[\s,.;:!?\-–—'\"“”‘’()\[\]<>/\\…]+")


def drop_leading_junk(tokens: Iterator[str]) -> Iterator[str]:
    """Drop leading punctuation/whitespace-only tokens (e.g. a stray comma the
    model emits before starting the real answer).  Streaming-friendly."""
    started = False
    for tok in tokens:
        if not started:
            if JUNK_TOKEN_RE.fullmatch(tok):
                continue
            started = True
        yield tok


def _apply_repetition_penalty(logits: torch.Tensor, tokens: list[int], penalty: float) -> torch.Tensor:
    """Penalize logits of tokens already present in ``tokens`` (batch of 1).

    Vectorised: a single gather/scatter over the unique seen ids instead of the
    v1 Python loop over ``set(tokens)``."""
    if penalty <= 1.0 or not tokens:
        return logits
    out = logits.clone()
    seen = torch.tensor(sorted(set(tokens)), dtype=torch.long, device=out.device)
    values = out[0, seen]
    out[0, seen] = torch.where(values > 0, values / penalty, values * penalty)
    return out


def _no_repeat_mask(logits: torch.Tensor, context: list[int], ngram_size: int) -> torch.Tensor:
    """Mask any candidate that would complete an n-gram already seen in context.

    Vectorised with ``Tensor.unfold``: the banned ids are the last elements of
    the seen n-grams whose prefix equals the current tail (v1 scanned all
    12,000 vocabulary entries in Python for every generated token)."""
    if ngram_size <= 1 or len(context) < ngram_size:
        return logits
    ctx = torch.as_tensor(context, dtype=torch.long, device=logits.device)
    ngrams = ctx.unfold(0, ngram_size, 1)              # (L-n+1, n)
    tail = ctx[-(ngram_size - 1):]                     # (n-1,)
    matches = (ngrams[:, : ngram_size - 1] == tail).all(dim=1)
    banned = ngrams[matches][:, -1].unique()
    if banned.numel() == 0:
        return logits
    out = logits.clone()
    out[0, banned] = float("-inf")
    return out


def _sample_from_cdf(cdf: torch.Tensor, u: torch.Tensor) -> int:
    """Map a uniform draw in [0, 1) onto a token id through the cumulative
    distribution.

    The index is clamped to the last token: a cumulative sum computed in a
    low-precision dtype can end *below* 1.0 (bf16: ``cdf[-1] == 0.99609375``),
    and ``searchsorted`` would then return ``len(cdf)`` - one past the end of
    the vocabulary, which crashed bf16 decoding on CUDA
    ("vectorized gather kernel index out of bounds").
    """
    index = int(torch.searchsorted(cdf, u.to(cdf.device)).item())
    return min(index, cdf.numel() - 1)


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    rng: torch.Generator,
) -> int:
    """Pick one token id from a ``(1, vocab)`` logits row.

    Greedy when ``temperature <= 0`` (v1 divided by zero here).  Otherwise the
    uniform draw comes from the *CPU* generator - identical on every device -
    and is mapped through the cumulative distribution on the logits' device.
    """
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())

    logits = logits.clone()
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits.float(), dim=-1)
    if not torch.isfinite(probs).all():
        raise RuntimeError("non-finite probabilities: every token was masked (all -inf logits)")
    cdf = probs.cumsum(dim=-1)[0].contiguous()
    u = torch.rand(1, generator=rng)
    return _sample_from_cdf(cdf, u)


@torch.no_grad()
def generate_tokens(
    model,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    seed: Optional[int] = None,
    stop_ids: set[int] | None = None,
    use_kv_cache: bool = True,
) -> Iterator[str]:
    """Yield decoded token strings one at a time (true token-level streaming).

    Generation starts *after* the prompt (the prompt is context only, never
    echoed).  Stops early when a token id in ``stop_ids`` is produced or when
    ``max_new_tokens`` is reached.
    """
    device = next(model.parameters()).device
    stop_ids = stop_ids or set()
    block_size = model.config.block_size
    ids = tokenizer.encode(prompt)[-block_size:]
    if not ids:
        raise ValueError("the prompt encodes to zero tokens")
    context: list[int] = list(ids)

    # CPU generator: the random stream (and therefore the output) is the same
    # on CPU and CUDA for a given seed.
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    cache = None
    if use_kv_cache and hasattr(model, "forward_cached"):
        logits, cache = model.forward_cached(torch.tensor([ids], dtype=torch.long, device=device))
        next_logits = logits[:, -1, :]
    else:
        next_logits = _full_window_logits(model, context, block_size, device)

    # Multi-byte UTF-8 (e.g. Chinese) spans several byte-level BPE tokens; a
    # single token may be an *incomplete* character.  Decode incrementally and
    # only yield COMPLETE characters so token-by-token streaming never emits
    # U+FFFD replacement chars for valid text.
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stopped = False
    for _ in range(max_new_tokens):
        logits = _apply_repetition_penalty(next_logits, context, repetition_penalty)
        logits = _no_repeat_mask(logits, context, no_repeat_ngram_size)
        nid = _sample_token(logits, temperature, top_k, rng)
        context.append(nid)
        if nid in stop_ids:
            stopped = True
            break
        chunk = dec.decode(tokenizer.byte_decoder.get(nid, b"<unk>"))
        if chunk:
            yield chunk

        if cache is not None:
            if len(context) > block_size:
                # The window slid: re-prefill the retained window so positions
                # restart at 0, exactly like the uncached path.
                window = torch.tensor([context[-block_size:]], dtype=torch.long, device=device)
                logits, cache = model.forward_cached(window, None)
            else:
                logits, cache = model.forward_cached(
                    torch.tensor([[nid]], dtype=torch.long, device=device), cache
                )
            next_logits = logits[:, -1, :]
        else:
            next_logits = _full_window_logits(model, context, block_size, device)

    if not stopped:  # flush any trailing complete characters at max tokens
        tail = dec.decode(b"", final=True)
        if tail:
            yield tail


def _full_window_logits(model, context: list[int], block_size: int, device):
    """Uncached reference path: re-run the last ``block_size`` tokens."""
    window = torch.tensor([context[-block_size:]], dtype=torch.long, device=device)
    return model(window)[0][:, -1, :]


def decode_stream(tokenizer: BPETokenizer, ids: Iterator[int]) -> Iterator[str]:
    """Decode a stream of (possibly multi-byte) token ids into complete UTF-8
    characters.  A Chinese character spans 3 byte-level BPE tokens, so a single
    token is often an *incomplete* character: those bytes are buffered and only
    emitted once the full character has arrived, so valid text never yields
    U+FFFD replacement characters.  Trailing incomplete bytes at end-of-stream
    flush as replacement characters (they are genuinely invalid UTF-8)."""
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for i in ids:
        chunk = dec.decode(tokenizer.byte_decoder.get(i, b"<unk>"))
        if chunk:
            yield chunk
    tail = dec.decode(b"", final=True)
    if tail:
        yield tail


def strip_leading_punct(text: str) -> str:
    """Remove leading whitespace / punctuation junk (e.g. a stray comma or
    period that the model emits before starting the real answer)."""
    return LEADING_PUNCT_RE.sub("", text)


def cut_at_end(text: str, marker: str = END_MARKER) -> str:
    cut = text.find(marker)
    return text[:cut] if cut != -1 else text


def generate_text(
    model,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    seed: Optional[int] = None,
    stop_ids: set[int] | None = None,
    clean: bool = True,
    use_kv_cache: bool = True,
) -> str:
    """Generate the full text (no streaming), with optional cleanup."""
    stop_ids = stop_ids or set()
    text = "".join(generate_tokens(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        seed=seed, stop_ids=stop_ids, use_kv_cache=use_kv_cache,
    ))
    if clean:
        text = cut_at_end(text)
        text = strip_leading_punct(text)
    return text


def generate_answer(
    model,
    tokenizer: BPETokenizer,
    question: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
    seed: Optional[int] = None,
    use_kv_cache: bool = True,
) -> str:
    """Instruction-formatted answer generation for QA-style prompts."""
    prompt = f"<|user|>{question}<|assistant|>"
    stop_ids = {tokenizer.special_to_id.get(END_MARKER, -1)}
    stop_ids.discard(-1)
    return generate_text(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        seed=seed, stop_ids=stop_ids, clean=True, use_kv_cache=use_kv_cache,
    )
