"""Decode-throughput benchmark: v1 vs KV cache + vectorised post-processing.

Measures, at batch=1 on a fixed prompt with a fixed number of generated tokens:

1. ``v1 (no cache, python postproc)`` - the code published before the upgrade,
   extracted verbatim from git (``nanollm/generation.py`` at the v1 commit);
2. ``no cache + vectorised postproc`` - isolates the post-processing win;
3. ``KV cache + vectorised postproc`` - the shipped path (default);
4. ``KV cache, no postproc`` - forward-bound upper bound (sampling only);

plus a microbenchmark of the logit post-processing itself (repetition penalty +
no-repeat n-gram) on a realistic 12k-vocab row with a ~150-token context.

Run:  python scripts/benchmark_generation.py --devices cpu,cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from nanollm.checkpoints import load_checkpoint
from nanollm.generation import (
    _apply_repetition_penalty,
    _no_repeat_mask,
    generate_text,
)
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer

V1_COMMIT = "0c4ed5d"  # last commit before the KV-cache / protocol upgrade
PROMPT = "<|user|>Would it be safe to eat the death cap?<|assistant|>"
GEN = dict(temperature=0.7, top_k=40, repetition_penalty=1.15, no_repeat_ngram_size=4)


def load_v1_module(commit: str = V1_COMMIT):
    cache = ROOT / "out" / "legacy_v1" / "generation_v1.py"
    cache.parent.mkdir(parents=True, exist_ok=True)
    source = subprocess.run(
        ["git", "show", f"{commit}:nanollm/generation.py"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    cache.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("generation_v1", cache)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def load_model(ckpt_path, tokenizer_path, device, dtype):
    tokenizer = BPETokenizer.load(tokenizer_path)
    ckpt = load_checkpoint(ckpt_path, map_location=device, required=("model", "config"))
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.eval()
    return model, tokenizer


def timed(fn, device, runs: int = 3):
    """Best-of-``runs`` wall time (best-of is the least noisy summary for a
    laptop GPU that is also driving the display)."""
    best = float("inf")
    for _ in range(runs):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_decode(model, tokenizer, device, n_tokens, v1=None, use_cache=True, gen=None):
    gen = GEN if gen is None else gen
    if v1 is not None:
        fn = lambda: v1.generate_text(  # noqa: E731
            model, tokenizer, PROMPT, max_new_tokens=n_tokens, seed=0,
            stop_ids=set(), clean=False, **gen)
    else:
        fn = lambda: generate_text(  # noqa: E731
            model, tokenizer, PROMPT, max_new_tokens=n_tokens, seed=0,
            stop_ids=set(), clean=False, use_kv_cache=use_cache, **gen)
    elapsed = timed(fn, device)
    return {"seconds": elapsed, "tokens": n_tokens, "tok_per_s": n_tokens / elapsed}


def bench_postproc(device, vocab=12000, context_len=150, iters=20, v1=None):
    """Per-step cost of repetition penalty + no-repeat n-gram masking, timed as a
    batch of ``iters`` calls (a single-call min-of-N is dominated by queue
    effects; what matters is the Python/sync cost paid per decode step)."""
    logits = torch.randn(1, vocab, device=device)
    context = list(range(context_len))
    if v1 is not None:
        fn = lambda: v1._no_repeat_mask(  # noqa: E731
            v1._apply_repetition_penalty(logits, context, 1.15), context, 4)
    else:
        fn = lambda: _no_repeat_mask(  # noqa: E731
            _apply_repetition_penalty(logits, context, 1.15), context, 4)
    fn()  # warmup
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    elapsed = time.perf_counter() - t0
    return {"ms_per_step": elapsed * 1000 / iters, "iters": iters, "total_s": elapsed}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/merged.pt")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--devices", type=str, default="cpu,cuda")
    p.add_argument("--dtypes", type=str, default="fp32,bf16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--rounds", type=int, default=5,
                   help="round-robin rounds per configuration (best time kept)")
    p.add_argument("--out", type=str, default="out/bench/benchmark.json")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    v1 = load_v1_module()
    results = {"prompt": PROMPT, "max_new_tokens": args.max_new_tokens,
               "batch_size": 1, "runs": args.runs, "v1_commit": V1_COMMIT,
               "torch": torch.__version__, "configs": {}}

    for device in [d.strip() for d in args.devices.split(",") if d.strip()]:
        if device == "cuda" and not torch.cuda.is_available():
            print("[bench] CUDA not available, skipping")
            continue
        for dtype in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
            if dtype == "bf16" and device != "cuda":
                continue  # bf16 CPU decode is not a supported configuration here
            model, tokenizer = load_model(args.ckpt, args.tokenizer, device, dtype)
            key = f"{device}/{dtype}"
            print(f"\n=== {key} | {args.max_new_tokens} new tokens | batch=1 ===")
            entry = {"device": device, "dtype": dtype}

            gen_no_proc = dict(GEN, repetition_penalty=1.0, no_repeat_ngram_size=0)
            variants = {
                "v1_no_cache": lambda: bench_decode(model, tokenizer, device, args.max_new_tokens, v1=v1),
                "no_cache_vectorised": lambda: bench_decode(model, tokenizer, device, args.max_new_tokens, use_cache=False),
                "kv_cache": lambda: bench_decode(model, tokenizer, device, args.max_new_tokens, use_cache=True),
                "kv_cache_no_postproc": lambda: bench_decode(model, tokenizer, device, args.max_new_tokens,
                                                             use_cache=True, gen=gen_no_proc),
            }
            # Round-robin rounds, keep the best time per variant: a laptop GPU
            # shared with the display throttles over a long sequential run, which
            # would otherwise penalise whichever variant runs last.
            for _ in range(args.rounds):
                for name, fn in variants.items():
                    result = fn()
                    if name not in entry or result["tok_per_s"] > entry[name]["tok_per_s"]:
                        entry[name] = result
            entry["postproc_v1"] = bench_postproc(device, v1=v1)
            entry["postproc_vectorised"] = bench_postproc(device)

            base = entry["v1_no_cache"]["tok_per_s"]
            for name in ("v1_no_cache", "no_cache_vectorised", "kv_cache", "kv_cache_no_postproc"):
                r = entry[name]
                print(f"  {name:<22} {r['tok_per_s']:8.1f} tok/s   ({r['seconds']:.2f}s)   "
                      f"x{r['tok_per_s']/base:.2f} vs v1")
            print(f"  post-processing: v1 {entry['postproc_v1']['ms_per_step']:.2f} ms/step  ->  "
                  f"vectorised {entry['postproc_vectorised']['ms_per_step']:.2f} ms/step")
            results["configs"][key] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    sys.exit(main())
