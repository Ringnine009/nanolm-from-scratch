"""Best-effort: fetch mushroom-related Wikipedia articles via HuggingFace
datasets and save them as plain text for the corpus.

The HF endpoint is read from the ``HF_ENDPOINT`` environment variable and
defaults to https://hf-mirror.com (a China-friendly mirror).

Dependency note: this optional script needs the ``datasets`` package, which is
NOT part of the base requirements (``pip install datasets`` manually if you
want to run it).

Two candidate datasets (tried in order):
  - pszemraj/simple_wikipedia (Simple English Wikipedia, Apache-2.0, small)
  - wikimedia/wikipedia config 20231101.en (CC BY-SA 4.0, large)

Only rows whose title matches mushroom/fungus keywords are kept, capped at
``--max-articles`` / ``--max-chars``.

Run:  python scripts/fetch_wikipedia_hf.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# use the mirror only when the user has not set HF_ENDPOINT explicitly
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TITLE_KEYWORDS = (
    "mushroom", "fungus", "fungi", "fungal", "mycology", "mycolog",
    "amanita", "bolet", "chanterelle", "morel", "truffle", "puffball",
    "agaric", "mycet", "mycelium", "mycorrhiza", "polypore", "stinkhorn",
    "inkcap", "webcap", "russula", "lactarius", "cordyceps", "ergot",
    "gilled", "basidio", "asco", "spore", "toadstool",
)

DATASETS = [
    ("pszemraj/simple_wikipedia", "default", "train", "title", "text"),
    ("wikimedia/wikipedia", "20231101.en", "train", "title", "text"),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--max-articles", type=int, default=1500)
    p.add_argument("--max-chars", type=int, default=1_800_000)
    p.add_argument("--out", type=str, default="data/corpus/raw/wikipedia_hf.txt")
    p.add_argument("--match-text", action="store_true",
                   help="also keep rows whose TEXT (not only title) mentions mushrooms")
    args = p.parse_args(argv)

    from datasets import load_dataset

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    chars = 0
    seen = 0
    skipped = 0

    for ds_name, config, split, title_col, text_col in DATASETS:
        if chars >= args.max_chars:
            break
        try:
            print(f"[fetch] streaming {ds_name} ({config}) ...", flush=True)
            ds = load_dataset(ds_name, config, split=split, streaming=True)
            for row in ds:
                seen += 1
                title = (row.get(title_col) or "").strip()
                text = (row.get(text_col) or "").strip()
                if not title or not text:
                    skipped += 1
                    continue
                if not any(k in title.lower() for k in TITLE_KEYWORDS):
                    if not (args.match_text and ("mushroom" in text.lower() or "fungus" in text.lower())):
                        continue
                # skip list/disambiguation-ish pages and very short stubs
                if text.lower().startswith(("list of", "this is a list")):
                    continue
                if len(text) < 500:
                    continue
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(f"\n===== {title} =====\n{text}\n")
                written += 1
                chars += len(text)
                if written % 50 == 0:
                    print(f"[fetch] {written} articles, {chars/1024:.0f} KB, scanned {seen} rows", flush=True)
                if written >= args.max_articles or chars >= args.max_chars:
                    break
        except Exception as e:
            print(f"[fetch] {ds_name} failed: {type(e).__name__}: {e}")
            continue

    print(f"[fetch] done: {written} articles, {chars/1024:.0f} KB -> {out_path}")
    return written


if __name__ == "__main__":
    sys.exit(main())
