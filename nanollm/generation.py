"""Unified text generation with repetition control.

Single implementation of decoding used by the CLI chat, the sample tool, the
FastAPI server and the evaluation script.  Features:

- ``repetition_penalty``: penalises tokens that already appear in the context
  (divide positive logits, multiply negative ones, >1.0 = stronger penalty);
- ``no_repeat_ngram_size``: masks any next token that would complete an n-gram
  already seen in the context+generation (blocks loops by construction);
- temperature scaling + top-k filtering;
- single-token decoding with ``<|end|>`` stopping and deterministic seeds;
- post-processing helpers: strip leading punctuation, cut at ``<|end|>``.
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
    """Penalize logits of tokens already present in ``tokens`` (batch of 1)."""
    if penalty <= 1.0 or not tokens:
        return logits
    out = logits.clone()
    for tid in set(tokens):
        row = out[0]
        if row[tid] > 0:
            row[tid] = row[tid] / penalty
        else:
            row[tid] = row[tid] * penalty
    return out


def _no_repeat_mask(logits: torch.Tensor, context: list[int], ngram_size: int) -> torch.Tensor:
    """Mask any candidate that would complete an n-gram already seen in context."""
    if ngram_size <= 1 or len(context) < ngram_size - 1:
        return logits
    seen: set[tuple[int, ...]] = set()
    for i in range(len(context) - ngram_size + 1):
        seen.add(tuple(context[i : i + ngram_size]))
    tail = tuple(context[-(ngram_size - 1):])
    out = logits.clone()
    n_vocab = out.size(-1)
    for v in range(n_vocab):
        if tuple(tail) + (v,) in seen:
            out[0, v] = float("-inf")
    return out


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
) -> Iterator[str]:
    """Yield decoded token strings one at a time (true token-level streaming).

    Generation starts *after* the prompt (the prompt is context only, never
    echoed).  Stops early when a token id in ``stop_ids`` is produced or when
    ``max_new_tokens`` is reached.
    """
    device = next(model.parameters()).device
    stop_ids = stop_ids or set()
    ids = tokenizer.encode(prompt)[-model.config.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    context: list[int] = list(ids)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    # Multi-byte UTF-8 (e.g. Chinese) spans several byte-level BPE tokens; a
    # single token may be an *incomplete* character.  Decode incrementally and
    # only yield COMPLETE characters so token-by-token streaming never emits
    # U+FFFD replacement chars for valid text.
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stopped = False
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits = model(idx_cond)[0][:, -1, :]
        logits = _apply_repetition_penalty(logits, context, repetition_penalty)
        logits = _no_repeat_mask(logits, context, no_repeat_ngram_size)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nid = torch.multinomial(probs, num_samples=1, generator=rng).item()
        context.append(nid)
        idx = torch.cat([idx, torch.tensor([[nid]], dtype=torch.long, device=device)], dim=1)
        if nid in stop_ids:
            stopped = True
            break
        chunk = dec.decode(tokenizer.byte_decoder.get(nid, b"<unk>"))
        if chunk:
            yield chunk
    if not stopped:  # flush any trailing complete characters at max tokens
        tail = dec.decode(b"", final=True)
        if tail:
            yield tail


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
) -> str:
    """Generate the full text (no streaming), with optional cleanup."""
    stop_ids = stop_ids or set()
    text = "".join(generate_tokens(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        seed=seed, stop_ids=stop_ids,
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
) -> str:
    """Instruction-formatted answer generation for QA-style prompts."""
    prompt = f"<|user|>{question}<|assistant|>"
    stop_ids = {tokenizer.special_to_id.get(END_MARKER, -1)}
    stop_ids.discard(-1)
    return generate_text(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        seed=seed, stop_ids=stop_ids, clean=True,
    )
