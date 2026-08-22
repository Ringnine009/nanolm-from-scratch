"""Interactive command-line chat with a trained (optionally LoRA-merged) model.

Tokens are streamed to the terminal as they are generated.
"""

from __future__ import annotations

import argparse
import sys

import torch

from nanollm.generation import END_MARKER, generate_tokens
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Chat with NanoLM in the terminal")
    p.add_argument("--ckpt", type=str, required=True, help="merged .pt or .ckpt checkpoint")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[loaded] {args.ckpt} | device={device} | "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("Type your question about mushroom safety (Ctrl+C / 'quit' to exit).\n")
    stop_ids = {tokenizer.special_to_id.get(END_MARKER, -1)}
    stop_ids.discard(-1)

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        prompt = f"<|user|>{question}<|assistant|>"
        print("NanoLM> ", end="", flush=True)
        for tok in generate_tokens(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            stop_ids=stop_ids,
        ):
            print(tok, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    sys.exit(main())
