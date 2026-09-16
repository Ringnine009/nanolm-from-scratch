"""Reproduce the **v1** held-out evaluation (pre-upgrade protocol) for the
audit record.

It runs the *old* code, extracted verbatim from git (``nanollm/generation.py``
as of the v1 commit), on the same checkpoint, and scores with the old rule
(``keyword in answer.lower()``).  Purpose:

- confirm the published 38.6% (CUDA) number, and
- confirm finding D1: the *same* checkpoint and the same script score
  differently on CPU vs CUDA, because v1 seeded ``torch.Generator(device=...)``
  per device.

Run:
  python scripts/reproduce_v1_eval.py --device cuda   # published protocol
  python scripts/reproduce_v1_eval.py --device cpu    # same code, other device
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

V1_COMMIT = "0c4ed5d"  # last commit before the evaluation-protocol upgrade
V1_MODULE_PATH = "nanollm/generation.py"


def load_v1_generation(commit: str = V1_COMMIT):
    """Extract the v1 generation module from git history and import it."""
    cache = ROOT / "out" / "legacy_v1" / "generation_v1.py"
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = subprocess.run(
            ["git", "show", f"{commit}:{V1_MODULE_PATH}"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout.decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        raise SystemExit(
            f"could not extract {V1_MODULE_PATH} from commit {commit}: {exc}\n"
            "run this script from the repository (git required)"
        )
    cache.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("generation_v1", cache)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/merged.pt")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--eval-jsonl", type=str, default="data/qa/qa_eval.jsonl")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--commit", type=str, default=V1_COMMIT)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    from nanollm.checkpoints import load_checkpoint
    from nanollm.model import GPT, GPTConfig
    from nanollm.tokenizer import BPETokenizer

    v1 = load_v1_generation(args.commit)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BPETokenizer.load(args.tokenizer)
    ckpt = load_checkpoint(args.ckpt, map_location=device, required=("model", "config"))
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    items = [
        json.loads(line)
        for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"[v1] commit={args.commit} module={V1_MODULE_PATH} device={device} items={len(items)}")

    rows = []
    for i, item in enumerate(items):
        answer = v1.generate_answer(
            model, tokenizer, item["question"],
            max_new_tokens=100, temperature=0.7, top_k=40,
            repetition_penalty=1.15, no_repeat_ngram_size=4, seed=i,   # v1 seeding
        )
        low = answer.lower()
        hits = [kw for kw in item["keywords"] if kw in low]            # v1 scoring
        rows.append({"question": item["question"], "answer": answer,
                     "keywords": item["keywords"], "hits": hits,
                     "hit_any": len(hits) > 0,
                     "hit_all": len(hits) == len(item["keywords"])})

    n = len(rows)
    hit_any = sum(r["hit_any"] for r in rows) / n
    hit_all = sum(r["hit_all"] for r in rows) / n
    print(f"[v1] device={device} keyword hit_any={hit_any*100:.1f}% ({sum(r['hit_any'] for r in rows)}/{n})"
          f"  hit_all={hit_all*100:.1f}%")

    out = Path(args.out or f"out/eval_v1_{device}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": "v1 (substring scoring, device-seeded RNG)",
        "commit": args.commit, "device": device, "n": n,
        "hit_any": hit_any, "hit_all": hit_all, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v1] written {out}")


if __name__ == "__main__":
    sys.exit(main())
