"""Merge a saved LoRA adapter into its base checkpoint.

Produces a single plain model checkpoint that can be served directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nanollm.lora import inject_lora, load_lora, merge_lora
from nanollm.model import GPT, GPTConfig


def main(argv=None):
    p = argparse.ArgumentParser(description="Merge LoRA adapters into the base checkpoint")
    p.add_argument("--base-ckpt", type=str, required=True)
    p.add_argument("--lora", type=str, required=True)
    p.add_argument("--out", type=str, default="checkpoints/merged.pt")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else args.device
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])

    payload = torch.load(args.lora, map_location="cpu", weights_only=False)
    inject_lora(model, r=payload["config"]["r"], alpha=payload["config"]["alpha"])
    result = load_lora(model, args.lora)
    assert not result["skipped"], f"adapters not found on model: {result['skipped']}"
    merge_lora(model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.__dict__, "merged_from": args.lora}, out)
    print(f"[done] merged model saved to {out} "
          f"(params={sum(p.numel() for p in model.parameters())/1e6:.2f}M)")


if __name__ == "__main__":
    sys.exit(main())
