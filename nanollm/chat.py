"""Interactive command-line chat with a trained (optionally LoRA-merged) model."""

from __future__ import annotations

import argparse
import sys

import torch

from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Chat with NanoLM in the terminal")
    p.add_argument("--ckpt", type=str, required=True, help="merged .pt or .ckpt checkpoint")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
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
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[loaded] {args.ckpt} | device={device} | "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("Type your question about mushroom safety (Ctrl+C / 'quit' to exit).\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        prompt = f"<|user|>{question}<|assistant|>"
        ids = tokenizer.encode(prompt)[-config.block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        end_id = tokenizer.special_to_id.get("<|end|>")
        generated = []
        cur = idx
        for _ in range(args.max_new_tokens):
            next_id = model.generate(cur, max_new_tokens=1, temperature=args.temperature, top_k=args.top_k)
            nid = next_id[0, -1].item()
            generated.append(nid)
            cur = next_id
            if nid == end_id:
                break
        text = tokenizer.decode(generated)[len(prompt):]
        cut = text.find("<|end|>")
        if cut != -1:
            text = text[:cut]
        print(f"NanoLM> {text.strip()}\n")


if __name__ == "__main__":
    sys.exit(main())
