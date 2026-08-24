"""Tests for multi-turn chat history: prompt concatenation and truncation."""

import pytest

from nanollm.server.app import ChatRequest, build_chat_prompt

H = [
    {"role": "user", "content": "What does the death cap look like?"},
    {"role": "assistant", "content": "It has an olive-green cap and white gills."},
    {"role": "user", "content": "How is it different from an edible mushroom?"},
    {"role": "assistant", "content": "Edible mushrooms do not have a volva."},
]


def _req(prompt="And the destroying angel?", history=None, wrap=True):
    return ChatRequest(prompt=prompt, history=history or [], wrap_instructions=wrap)


def test_history_concatenates_in_order(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    req = _req(history=H)
    prompt = build_chat_prompt(req, tok, block_size=256)
    assert prompt == (
        "<|user|>What does the death cap look like?<|assistant|>"
        "It has an olive-green cap and white gills.<|user|>"
        "How is it different from an edible mushroom?<|assistant|>"
        "Edible mushrooms do not have a volva.<|user|>And the destroying angel?<|assistant|>"
    )
    # the whole prompt stays within the block budget
    assert len(tok.encode(prompt)) <= 256


def test_no_history_matches_current_behaviour(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    req = _req(history=[])
    assert build_chat_prompt(req, tok, block_size=256) == "<|user|>And the destroying angel?<|assistant|>"


def test_wrap_instructions_false_uses_plain_join(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    req = _req(history=H, wrap=False)
    prompt = build_chat_prompt(req, tok, block_size=256)
    assert "<|user|>" not in prompt and "<|assistant|>" not in prompt
    for h in H:
        assert h["content"] in prompt
    assert prompt.endswith("And the destroying angel?")


def test_truncation_drops_oldest_turns(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    # a tiny block budget forces the oldest turns to be dropped
    req = _req(history=H)
    prompt = build_chat_prompt(req, tok, block_size=40)
    # the current question must always survive
    assert "And the destroying angel?" in prompt
    # oldest content should be gone (does not fit the budget)
    assert "What does the death cap look like?" not in prompt
    assert len(tok.encode(prompt)) <= 40 + 16  # reserve is accounted for


def test_truncation_keeps_current_question_even_if_huge(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    req = _req(prompt="A" * 500, history=H)
    prompt = build_chat_prompt(req, tok, block_size=100)
    assert "A" * 500 in prompt  # current question is never dropped
    assert "What does the death cap look like?" not in prompt


def test_history_accepts_bad_role_rejected_by_pydantic():
    with pytest.raises(Exception):
        ChatRequest(prompt="q", history=[{"role": "system", "content": "x"}])
