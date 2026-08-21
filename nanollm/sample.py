"""Command-line sampling from a trained checkpoint (temperature / top-k)."""

from __future__ import annotations

import argparse
import sys

import torch

from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample text from a NanoLM checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="path to .ckpt (or merged .pt)")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--prompt", type=str, default="Mushroom hunting is")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--wrap-instructions", action="store_true",
                   help="wrap the prompt in <|user|>…<|assistant|> (for LoRA-finetuned models)")
    p.add_argument("--stop-at-end", action="store_true",
                   help="stop generation at the <|end|> token and strip the echoed prompt")
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt.get("config", {"vocab_size": tokenizer.vocab_size}))
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = args.prompt
    if args.wrap_instructions and "<|user|>" not in prompt:
        prompt = f"<|user|>{prompt}<|assistant|>"

    ids = tokenizer.encode(prompt)[-config.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for i in range(args.count):
        gen = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            seed=None if args.seed is None else args.seed + i,
        )
        text = tokenizer.decode(gen[0].tolist())
        if args.wrap_instructions:
            text = text[len(prompt):]
        if args.stop_at_end:
            cut = text.find("<|end|>")
            if cut != -1:
                text = text[:cut]
        print(f"--- sample {i + 1} ---")
        print(text.strip())
        print()


if __name__ == "__main__":
    sys.exit(main())
