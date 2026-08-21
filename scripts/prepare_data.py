"""Tokenize the corpus and train the BPE tokenizer.

Writes: data/processed/tokenizer.json, data/processed/train.bin, val.bin.

Run:  python scripts/prepare_data.py [--vocab-size 12000] [--corpus data/corpus/corpus.txt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.data import tokenize_corpus  # noqa: E402
from nanollm.tokenizer import BPETokenizer  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Train tokenizer and tokenize corpus")
    p.add_argument("--corpus", type=str, default="data/corpus/corpus.txt")
    p.add_argument("--out-dir", type=str, default="data/processed")
    p.add_argument("--vocab-size", type=int, default=12000)
    p.add_argument("--val-fraction", type=float, default=0.01)
    args = p.parse_args(argv)

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = corpus_path.read_text(encoding="utf-8")
    print(f"[corpus] {len(text)/1024:.0f} KB of text")

    # BPE merge training on the corpus (deterministic)
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train([text], verbose=True)
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    print(f"[bpe] {len(tokenizer.merges)} merges learned; tokenizer -> {tok_path}")

    stats = tokenize_corpus(corpus_path, tokenizer, out_dir, val_fraction=args.val_fraction)
    print(f"[data] train_tokens={stats['train_tokens']} val_tokens={stats['val_tokens']}")
    print(f"[data] train.bin/val.bin written to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
