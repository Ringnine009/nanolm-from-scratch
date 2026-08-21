"""Self-implemented byte-level BPE tokenizer, trained on the corpus.

Design notes (see README for the full justification):
- Byte-level: the base vocabulary is the 256 raw bytes, so *any* UTF-8
  text can be encoded and there is never an out-of-vocabulary token.
- GPT-2-style pre-tokenization with the ``regex`` package: merges are only
  learned inside "word" chunks, which keeps tokens linguistically sane and
  keeps whitespace handling exact (round-trip safe).
- Merge learning uses the classic subword-nmt algorithm: repeatedly find the
  most frequent adjacent byte pair and replace it with a new token id.  We
  keep per-chunk pair indices so each round only rescans affected chunks
  (fast enough for a multi-MB corpus in pure Python).
- A few special tokens (e.g. ``<|user|>``) are reserved at the end of the
  vocabulary for the instruction fine-tuning format.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import regex

# GPT-2 style pre-tokenization pattern: contractions, letter runs (with an
# optional leading space), digit runs, single punctuation, and whitespace.
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Reserved instruction-format tokens, placed at the end of the vocabulary.
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]

_VERSION = 1


class BPETokenizer:
    """A trainable byte-level BPE tokenizer with optional special tokens."""

    def __init__(self, vocab_size: int = 12000, special_tokens: list[str] | None = None):
        if vocab_size < 256 + 2:
            raise ValueError("vocab_size must be at least 258 (256 bytes + specials)")
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens if special_tokens is not None else SPECIAL_TOKENS)
        self.num_special = len(self.special_tokens)
        self.merges: list[tuple[tuple[int, int], int]] = []  # ((a, b), new_id) in creation order
        self.ranks: dict[tuple[int, int], int] = {}          # pair -> merged id
        self.byte_decoder: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_to_id: dict[str, int] = {}
        for i, s in enumerate(self.special_tokens):
            sid = 256 + (self.vocab_size - 256 - self.num_special) + i
            self.special_to_id[s] = sid
            self.byte_decoder[sid] = s.encode("utf-8")  # specials decode to their literal text

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, texts: list[str], verbose: bool = False) -> "BPETokenizer":
        """Learn merges from ``texts`` (subword-nmt style, chunk-aware)."""
        num_merges = self.vocab_size - 256 - self.num_special
        # Pre-tokenize every text into byte chunks.
        chunks: list[list[int]] = []
        for text in texts:
            for m in regex.finditer(GPT2_SPLIT_PATTERN, text):
                chunk = list(m.group().encode("utf-8"))
                if chunk:
                    chunks.append(chunk)

        counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_chunks: dict[tuple[int, int], set[int]] = defaultdict(set)
        for ci, ch in enumerate(chunks):
            for i in range(len(ch) - 1):
                p = (ch[i], ch[i + 1])
                counts[p] += 1
                pair_chunks[p].add(ci)

        next_id = 256
        for step in range(num_merges):
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], p))  # deterministic tie-break
            if counts[best] < 2:
                break
            a, b = best
            new_id = next_id
            next_id += 1
            self.merges.append((best, new_id))
            self.ranks[best] = new_id
            self.byte_decoder[new_id] = self.byte_decoder[a] + self.byte_decoder[b]

            affected = sorted(pair_chunks.pop(best, set()))
            del counts[best]
            for ci in affected:
                ch = chunks[ci]
                # remove old pair contributions
                for i in range(len(ch) - 1):
                    p = (ch[i], ch[i + 1])
                    counts[p] -= 1
                    if counts[p] <= 0:
                        del counts[p]
                # build merged chunk
                new_ch: list[int] = []
                i = 0
                while i < len(ch):
                    if i + 1 < len(ch) and (ch[i], ch[i + 1]) == best:
                        new_ch.append(new_id)
                        i += 2
                    else:
                        new_ch.append(ch[i])
                        i += 1
                chunks[ci] = new_ch
                # add new pair contributions
                for i in range(len(new_ch) - 1):
                    p = (new_ch[i], new_ch[i + 1])
                    counts[p] += 1
                    pair_chunks[p].add(ci)
            if verbose and (step + 1) % 1000 == 0:
                print(f"[bpe] merge {step + 1}/{num_merges} (best pair {(a, b)})")
        return self

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into token ids (special tokens become single ids)."""
        ids: list[int] = []
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            for part in re.split(pattern, text):
                if part in self.special_to_id:
                    ids.append(self.special_to_id[part])
                elif part:
                    ids.extend(self._encode_plain(part))
        else:
            ids.extend(self._encode_plain(text))
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for m in regex.finditer(GPT2_SPLIT_PATTERN, text):
            ids.extend(self._encode_chunk(list(m.group().encode("utf-8"))))
        return ids

    def _encode_chunk(self, chunk: list[int]) -> list[int]:
        """Apply learned merges to one byte chunk (greedy lowest-rank merge)."""
        if len(chunk) < 2:
            return chunk
        while len(chunk) >= 2:
            ranks = [self.ranks.get((chunk[i], chunk[i + 1])) for i in range(len(chunk) - 1)]
            valid = [(r, i) for i, r in enumerate(ranks) if r is not None]
            if not valid:
                break
            _, idx = min(valid, key=lambda t: (t[0], t[1]))
            merged = self.ranks[(chunk[idx], chunk[idx + 1])]
            chunk = chunk[:idx] + [merged] + chunk[idx + 2:]
        return chunk

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def decode(self, ids: list[int]) -> str:
        pieces: list[bytes] = []
        for i in ids:
            # a trained model may sample ids in the vocab budget that the
            # tokenizer never produced (e.g. when BPE training stopped early);
            # decode them as a placeholder instead of crashing
            pieces.append(self.byte_decoder.get(i, b"<unk>"))
        return b"".join(pieces).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for (a, b), _ in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=payload["vocab_size"], special_tokens=payload["special_tokens"])
        for pair in payload["merges"]:
            a, b = pair
            new_id = 256 + len(tok.merges)
            tok.merges.append(((a, b), new_id))
            tok.ranks[(a, b)] = new_id
            tok.byte_decoder[new_id] = tok.byte_decoder[a] + tok.byte_decoder[b]
        for s in tok.special_tokens:
            tok.byte_decoder[tok.special_to_id[s]] = s.encode("utf-8")
        return tok

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def n_embd_vocab(self) -> int:
        return self.vocab_size
