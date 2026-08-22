"""Evaluate the LoRA-merged model on the held-out eval set.

Scoring (as requested: keyword / exact-match based):
- ``hit_any`` : fraction of items where at least one expected keyword appears
  in the generated answer (case-insensitive substring);
- ``hit_all`` : fraction where ALL expected keywords appear (strict);
- per-category breakdown + up to ``--show-failures`` failure examples.

Run:  python scripts/evaluate.py [--ckpt checkpoints/merged.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from nanollm.generation import generate_answer
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer

CATEGORIES = [
    ("edibility", slice(0, 10)),
    ("identification", slice(10, 18)),
    ("habitat", slice(18, 26)),
    ("symptoms", slice(26, 34)),
    ("first-aid/general", slice(34, 44)),
]


def load_model(ckpt_path: str, tokenizer_path: str, device: str):
    tokenizer = BPETokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/merged.pt")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--eval-jsonl", type=str, default="data/qa/qa_eval.jsonl")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--show-failures", type=int, default=3)
    p.add_argument("--out", type=str, default="out/eval_results.json")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.ckpt, args.tokenizer, device)

    items = [json.loads(line) for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[eval] {len(items)} items | model={args.ckpt} | device={device}")

    rows = []
    for i, item in enumerate(items):
        ans = generate_answer(
            model, tokenizer, item["question"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=1.15, no_repeat_ngram_size=4,
            seed=i,
        )
        low = ans.lower()
        hits = [kw for kw in item["keywords"] if kw in low]
        rows.append({"question": item["question"], "answer": ans,
                     "keywords": item["keywords"], "hits": hits,
                     "hit_any": len(hits) > 0, "hit_all": len(hits) == len(item["keywords"])})

    n = len(rows)
    any_rate = sum(r["hit_any"] for r in rows) / n
    all_rate = sum(r["hit_all"] for r in rows) / n
    print(f"\n== overall ==")
    print(f"keyword hit (any of expected): {any_rate*100:.1f}%  ({sum(r['hit_any'] for r in rows)}/{n})")
    print(f"keyword hit (all expected):    {all_rate*100:.1f}%  ({sum(r['hit_all'] for r in rows)}/{n})")

    cat_stats = {}
    for name, sl in CATEGORIES:
        sub = rows[sl]
        if not sub:
            continue
        a = sum(r["hit_any"] for r in sub) / len(sub)
        b = sum(r["hit_all"] for r in sub) / len(sub)
        cat_stats[name] = {"n": len(sub), "hit_any": round(a, 3), "hit_all": round(b, 3)}
        print(f"  {name:<18} n={len(sub):>2}  hit_any={a*100:5.1f}%  hit_all={b*100:5.1f}%")

    if args.show_failures > 0:
        fails = [r for r in rows if not r["hit_any"]]
        print(f"\n== {len(fails)} complete failures (no expected keyword) ==")
        for r in fails[: args.show_failures]:
            print(f"  Q: {r['question']}")
            print(f"  A: {r['answer'][:140]}")
            print(f"  expected: {r['keywords']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": n, "hit_any": any_rate, "hit_all": all_rate,
        "categories": cat_stats,
        "args": vars(args),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    sys.exit(main())
