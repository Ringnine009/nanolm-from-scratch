"""Command-line sampling from a trained checkpoint.

Uses the unified generation module (repetition penalty + no-repeat n-gram +
<|end|> stopping + leading-punctuation cleanup).  With --wrap-instructions the
prompt is wrapped in the <|user|>/<|assistant|> format (for LoRA-finetuned
models) and the output is cleaned into a plain answer.
"""

from __future__ import annotations

import argparse
import sys

import torch

from nanollm.generation import END_MARKER, generate_answer, generate_text
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample text from a NanoLM checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="path to .ckpt (or merged .pt)")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--prompt", type=str, default="Mushroom hunting is")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--wrap-instructions", action="store_true",
                   help="wrap the prompt in <|user|>…<|assistant|> and return a clean answer")
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    config = GPTConfig(**ckpt.get("config", {"vocab_size": tokenizer.vocab_size}))
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stop_ids = {tokenizer.special_to_id.get(END_MARKER, -1)}
    stop_ids.discard(-1)

    for i in range(args.count):
        seed = None if args.seed is None else args.seed + i
        if args.wrap_instructions:
            text = generate_answer(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                seed=seed,
            )
        else:
            text = generate_text(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                seed=seed, stop_ids=stop_ids, clean=True,
            )
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    sys.exit(main())
