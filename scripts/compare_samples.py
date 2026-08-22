"""Compare sampling of the base model vs the LoRA-merged model on the same
instruction prompts (the 'before / after fine-tuning' demo).

Run:  python scripts/compare_samples.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from nanollm.generation import generate_answer
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer

PROMPTS = [
    "Is the death cap mushroom poisonous?",
    "Are morels safe to eat?",
    "How can I tell a chanterelle from a jack-o'-lantern mushroom?",
    "What should I do if someone eats a poisonous mushroom?",
]


def load_model(ckpt_path: str, tokenizer_path: str, device: str):
    tokenizer = BPETokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def answer(model, tokenizer, question: str, device: str, max_new_tokens: int = 110,
           temperature: float = 0.7, top_k: int = 40) -> str:
    return generate_answer(
        model, tokenizer, question,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        repetition_penalty=1.15, no_repeat_ngram_size=4,
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="out/pretrain/best.ckpt")
    p.add_argument("--merged", type=str, default="checkpoints/merged.pt")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--prompts", type=str, nargs="+", default=PROMPTS)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=110)
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tokenizer = load_model(args.base, args.tokenizer, device)
    merged_model, _ = load_model(args.merged, args.tokenizer, device)
    torch.manual_seed(args.seed)

    for q in args.prompts:
        print("=" * 78)
        print(f"Q: {q}")
        print("-" * 78)
        print(f"[before LoRA] {answer(base_model, tokenizer, q, device, args.max_new_tokens)}")
        print()
        print(f"[after  LoRA] {answer(merged_model, tokenizer, q, device, args.max_new_tokens)}")
        print()


if __name__ == "__main__":
    sys.exit(main())
