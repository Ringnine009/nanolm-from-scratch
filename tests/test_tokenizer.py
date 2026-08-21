"""Tests for the self-implemented byte-level BPE tokenizer."""

import pytest

from nanollm.tokenizer import BPETokenizer, SPECIAL_TOKENS

SAMPLE_TEXTS = [
    "The death cap mushroom (Amanita phalloides) is deadly poisonous.\n",
    "Chanterelles smell like apricots and grow near oak trees.\n",
    "Never eat a mushroom unless you are 100% sure of its identity!\n",
    "The <|user|> tag is special, but this sentence is plain text.\n",
]


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer(vocab_size=512)  # 256 bytes + 512-256-3 = 253 merges
    t.train(SAMPLE_TEXTS)
    return t


def test_vocab_size_matches_construction(tok):
    assert tok.vocab_size == 512
    assert tok.num_special == len(SPECIAL_TOKENS)


def test_encode_decode_roundtrip(tok):
    for text in SAMPLE_TEXTS:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_special_tokens_have_reserved_ids(tok):
    for s in SPECIAL_TOKENS:
        assert tok.special_to_id[s] >= 256
        # a special token encodes to exactly one id
        assert tok.encode(s) == [tok.special_to_id[s]]
        assert tok.decode([tok.special_to_id[s]]) == s


def test_special_token_embedded_in_text(tok):
    text = "Question?<|user|>Answer.<|end|>"
    ids = tok.encode(text)
    assert tok.special_to_id["<|user|>"] in ids
    assert tok.special_to_id["<|end|>"] in ids
    assert tok.decode(ids) == text


def test_deterministic_encoding(tok):
    a = tok.encode(SAMPLE_TEXTS[0])
    b = tok.encode(SAMPLE_TEXTS[0])
    assert a == b


def test_any_bytes_encodable():
    # byte-level BPE must never produce unknown tokens
    t = BPETokenizer(vocab_size=300)
    t.train(["hello world", "mushroom hunter", "a b c"])
    weird = "caf\u00e9 \u4e2d\u6587 \U0001f344 \x00\xff binary"
    ids = t.encode(weird)
    assert t.decode(ids) == weird
    assert all(0 <= i < t.vocab_size for i in ids)


def test_save_load_roundtrip(tmp_path):
    t = BPETokenizer(vocab_size=512)
    t.train(SAMPLE_TEXTS)
    path = tmp_path / "tokenizer.json"
    t.save(path)
    t2 = BPETokenizer.load(path)
    for text in SAMPLE_TEXTS:
        assert t2.encode(text) == t.encode(text)
        assert t2.decode(t2.encode(text)) == text


def test_smaller_vocab_fewer_merges():
    t = BPETokenizer(vocab_size=300)
    t.train(["aaaa bbbb cccc " * 50])
    # training stops early when no pair occurs >= 2 times; it must never
    # exceed the budget of ids reserved for merges
    assert 0 < len(t.merges) <= 300 - 256 - len(SPECIAL_TOKENS)
    assert len(t) == 300


def test_merge_learning_actually_merges():
    # repeated "ab" should become a merged token
    t = BPETokenizer(vocab_size=280)
    t.train(["ab" * 200])
    merged = t.encode("ab" * 10)
    assert len(merged) < 20  # strict fewer tokens than raw bytes
