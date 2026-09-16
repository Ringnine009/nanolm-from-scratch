"""Held-out evaluation under **protocol v2** (word-boundary keywords + polarity
+ baselines + multi-seed interval).

What changed vs the v1 evaluator (audit findings D1/D2/D3/D5):

- **D2** v1 scored ``keyword in answer.lower()``: ``cook`` was credited by
  *cooking*, ``no`` by *pois**no**us* / *North America*.  v2 matches on word
  boundaries (``nanollm.eval_protocol.keyword_hit``) and reports exactly which
  old "hits" were pure substrings.
- **D3** v1 never read the items' ``polarity`` field, so an answer that flipped
  the stance scored the same as a correct one.  v2 detects the answer's stance
  (with negation scope) and reports violations; ``correct`` = fact present AND
  not contradicted.
- **D1** v1 sampled with a device-seeded ``torch.Generator``, so the same
  checkpoint scored 38.6% on CUDA and 52.3% on CPU.  Sampling is now driven by
  a device-independent CPU stream; run ``--device cpu`` and ``--device cuda``
  and the answers match.
- **D5** every report carries the "empty answer" floor, a lexical
  keyword-lookup baseline, and Wilson 95% intervals; ``--seeds`` repeats the run
  and reports mean/spread.

Run:
  python scripts/evaluate.py --seeds 0,1,2                 # CUDA if available
  python scripts/evaluate.py --device cpu --seeds 0,1,2    # device check
  python scripts/evaluate.py --from-generations out/eval_generations.json   # re-score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from nanollm.eval_protocol import (
    DEFAULT_EVAL_JSONL,
    DEFAULT_TRAIN_JSONL,
    build_report,
    format_report,
    load_items,
    summarize_seeds,
)
from nanollm.checkpoints import load_checkpoint
from nanollm.generation import generate_answer
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer

GENERATION_DEFAULTS = dict(
    max_new_tokens=100, temperature=0.7, top_k=40,
    repetition_penalty=1.15, no_repeat_ngram_size=4,
)


def load_model(ckpt_path: str, tokenizer_path: str, device: str, dtype: str = "fp32"):
    tokenizer = BPETokenizer.load(tokenizer_path)
    ckpt = load_checkpoint(ckpt_path, map_location=device, required=("model", "config"))
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.eval()
    return model, tokenizer


def generate_all(model, tokenizer, items, seed_base: int, use_kv_cache: bool = True) -> list[str]:
    """One answer per item; item ``i`` uses ``seed_base + i`` so that run 0
    reproduces the v1 seeding convention (``seed=i``) exactly."""
    answers = []
    for i, item in enumerate(items):
        answers.append(generate_answer(
            model, tokenizer, item["question"],
            seed=seed_base + i, use_kv_cache=use_kv_cache, **GENERATION_DEFAULTS,
        ))
    return answers


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/merged.pt")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--eval-jsonl", type=str, default=str(DEFAULT_EVAL_JSONL))
    p.add_argument("--train-jsonl", type=str, default=str(DEFAULT_TRAIN_JSONL))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="comma-separated run indices; item seed = index*1000 + item")
    p.add_argument("--no-kv-cache", action="store_true", help="decode with the uncached reference path")
    p.add_argument("--show-failures", type=int, default=3)
    p.add_argument("--out", type=str, default="out/eval_results.json")
    p.add_argument("--generations-out", type=str, default=None,
                   help="cache the generated answers here so scoring can be redone without the GPU")
    p.add_argument("--from-generations", type=str, default=None,
                   help="score a previously cached generation file instead of generating")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    items = load_items(args.eval_jsonl)
    seed_indices = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.from_generations:
        cached = json.loads(Path(args.from_generations).read_text(encoding="utf-8"))
        device = cached.get("device", "n/a")
        generations = {int(k): v for k, v in cached["generations"].items()}
        seed_indices = sorted(generations)
        print(f"[eval] re-scoring {len(seed_indices)} cached generation run(s) "
              f"from {args.from_generations} (orig device={device})")
    else:
        device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_model(args.ckpt, args.tokenizer, device, args.dtype)
        print(f"[eval] {len(items)} items | model={args.ckpt} | device={device} | dtype={args.dtype} "
              f"| kv_cache={not args.no_kv_cache} | runs={seed_indices}")
        generations = {}
        for run in seed_indices:
            answers = generate_all(model, tokenizer, items, seed_base=run * 1000,
                                   use_kv_cache=not args.no_kv_cache)
            generations[run] = answers
            print(f"[eval] run {run}: generated {len(answers)} answers")

        if args.generations_out:
            path = Path(args.generations_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "device": device, "dtype": args.dtype, "ckpt": args.ckpt,
                "generation": GENERATION_DEFAULTS,
                "kv_cache": not args.no_kv_cache,
                "seeds": seed_indices, "seed_base_per_run": {str(r): r * 1000 for r in seed_indices},
                "generations": {str(r): generations[r] for r in seed_indices},
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[eval] generations cached -> {path}")

    reports = []
    for run in seed_indices:
        report = build_report(items, generations[run], train_jsonl=args.train_jsonl)
        report["run_index"] = run
        report["answers_seed_base"] = run * 1000
        reports.append(report)
        print(f"\n########## run {run} (item seeds {run * 1000}..{run * 1000 + len(items) - 1}) ##########")
        print(format_report(report, show_failures=args.show_failures))

    summary = summarize_seeds(reports)
    if len(reports) > 1:
        print("\n== multi-seed summary (v2) ==")
        print(f"  runs={summary['seeds']} per-run hit_any: "
              + ", ".join(f"{r*100:.1f}%" for r in summary["per_seed_hit_any"]))
        print(f"  mean={summary['mean_hit_any']*100:.1f}%  stdev={summary['stdev_hit_any']*100:.1f}pp  "
              f"range=[{summary['min_hit_any']*100:.1f}%, {summary['max_hit_any']*100:.1f}%]")
        print(f"  pooled {summary['pooled_hit_any']*100:.1f}% "
              f"({summary['seeds']}x{summary['n_per_seed']} answers) "
              f"95% CI [{summary['pooled_hit_any_ci95'][0]*100:.1f}, {summary['pooled_hit_any_ci95'][1]*100:.1f}]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "protocol": "v2",
        "ckpt": args.ckpt,
        "device": device,
        "dtype": args.dtype,
        "kv_cache": not args.no_kv_cache,
        "generation": GENERATION_DEFAULTS,
        "summary": summary,
        "baselines": reports[0].get("baselines", {}),
        "per_run": reports,
        "args": vars(args),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    sys.exit(main())
