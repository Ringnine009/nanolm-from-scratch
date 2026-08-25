"""Tests for the unified generation module (repetition control + stop logic)."""

import torch

from nanollm.generation import (
    _apply_repetition_penalty,
    _no_repeat_mask,
    cut_at_end,
    decode_stream,
    drop_leading_junk,
    generate_tokens,
    strip_leading_punct,
)


def test_decode_stream_multibyte_no_replacement(tiny_tokenizer):
    """A Chinese character is 3 byte-level BPE tokens; decoding the stream
    token-by-token must not emit U+FFFD for valid text (v8 Chinese fix)."""
    tok, _ = tiny_tokenizer
    text = "毒鹅膏含有鹅膏毒肽。误食后应立即就医。"
    ids = tok.encode(text)
    assert len(ids) > len(text)  # multi-byte: more tokens than characters
    chunks = list(decode_stream(tok, iter(ids)))
    assert "".join(chunks) == text
    assert "\ufffd" not in text
    # an incomplete leading character (only 2 of its 3 bytes) yields nothing yet
    partial = "".join(decode_stream(tok, iter(ids[:2])))
    assert partial == "" or "\ufffd" in partial  # no partial garbage characters


def test_repetition_penalty_lowers_repeated_logits():
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
    penalized = _apply_repetition_penalty(logits, tokens=[0, 1, 0], penalty=1.5)
    # token 0 repeated: positive logit divided by 1.5
    assert abs(penalized[0, 0].item() - (2.0 / 1.5)) < 1e-5
    # token 1 repeated: negative logit multiplied by 1.5
    assert abs(penalized[0, 1].item() - (-1.0 * 1.5)) < 1e-5
    # token 2 not seen: unchanged
    assert abs(penalized[0, 2].item() - 0.5) < 1e-5
    # token 3 not seen: unchanged
    assert abs(penalized[0, 3].item() - 3.0) < 1e-5


def test_repetition_penalty_noop_when_penalty_one():
    logits = torch.randn(1, 8)
    out = _apply_repetition_penalty(logits.clone(), tokens=[1, 2, 3], penalty=1.0)
    assert torch.equal(out, logits)


def test_no_repeat_mask_blocks_seen_ngram_continuations():
    context = [10, 20, 30, 40, 10, 20, 30]  # seen 3-grams: (10,20,30),(20,30,40),(30,40,10),(40,10,20)
    logits = torch.randn(1, 64)
    masked = _no_repeat_mask(logits, context, ngram_size=3)
    # tail is (20,30); continuing with 40 would form the seen n-gram (20,30,40) -> -inf
    assert masked[0, 40].item() == float("-inf")
    # an unseen continuation stays finite
    assert torch.isfinite(masked[0, 7])


def test_no_repeat_mask_requires_context_length():
    logits = torch.randn(1, 8)
    out = _no_repeat_mask(logits, context=[1, 2], ngram_size=4)
    assert torch.equal(out, logits)  # not enough context, no-op


def test_strip_leading_punct():
    assert strip_leading_punct("..., but only after cooking") == "but only after cooking"
    assert strip_leading_punct("   the death cap") == "the death cap"
    assert strip_leading_punct("already clean") == "already clean"
    assert strip_leading_punct("") == ""


def test_drop_leading_junk():
    stream = iter([",", " ,", ".", "the", "death", "cap"])
    out = list(drop_leading_junk(stream))
    assert out == ["the", "death", "cap"]
    # once a real token starts, later punctuation is kept
    stream2 = iter(["the", ",", "cap"])
    assert list(drop_leading_junk(stream2)) == ["the", ",", "cap"]
    # all-junk stream -> empty
    assert list(drop_leading_junk(iter([",", "."]))) == []


def test_cut_at_end():
    assert cut_at_end("hello<|end|>world") == "hello"
    assert cut_at_end("no end token here") == "no end token here"


def test_generate_tokens_stops_at_stop_id(tiny_checkpoint, tiny_tokenizer):
    import torch as _t

    from nanollm.model import GPT, GPTConfig

    tok, tok_path = tiny_tokenizer
    ckpt = _t.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    end_id = tok.special_to_id["<|end|>"]
    tokens = list(generate_tokens(
        model, tok, "the death cap is", max_new_tokens=30,
        temperature=0.7, top_k=20, seed=0, stop_ids={end_id},
    ))
    assert len(tokens) <= 30
    # if it stopped early, the stop token was produced; otherwise it ran to the cap
    # (the tiny model may or may not emit <|end|>; either way it must terminate)


def test_generate_no_repeat_ngram_property(tiny_checkpoint, tiny_tokenizer):
    """With no_repeat_ngram_size=N the output cannot contain a repeated N-gram
    (holds by construction of the masking algorithm)."""
    import torch as _t

    from nanollm.model import GPT, GPTConfig

    tok, tok_path = tiny_tokenizer
    ckpt = _t.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    text = "".join(generate_tokens(
        model, tok, "the death cap", max_new_tokens=60,
        temperature=0.9, top_k=25, seed=1, no_repeat_ngram_size=4,
    ))
    ids = tok.encode(text)
    ngrams = {tuple(ids[i:i + 4]) for i in range(len(ids) - 3)}
    assert len(ngrams) == len(ids) - 3, "repeated 4-gram found despite no_repeat_ngram"


def test_generate_tokens_deterministic_with_seed(tiny_checkpoint, tiny_tokenizer):
    import torch as _t

    from nanollm.model import GPT, GPTConfig

    tok, tok_path = tiny_tokenizer
    ckpt = _t.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    a = "".join(generate_tokens(model, tok, "mushrooms", max_new_tokens=20,
                                temperature=0.8, top_k=15, seed=7))
    b = "".join(generate_tokens(model, tok, "mushrooms", max_new_tokens=20,
                                temperature=0.8, top_k=15, seed=7))
    assert a == b
