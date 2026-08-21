"""LoRA instruction fine-tuning of a pretrained NanoLM checkpoint.

Loads the base model, injects LoRA adapters into the attention (and MLP)
projections, freezes the base weights, and trains only the adapters on the
mushroom-safety QA instruction set.  Only the assistant portion of each
example is supervised (targets before ``<|assistant|>`` are masked with -100).
Adapters are saved independently (``lora.pt``) so the base checkpoint stays
untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nanollm.data import InstructionDataset, qa_pad_collate
from nanollm.lora import adapter_summary, inject_lora, save_lora
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LoRA fine-tune NanoLM on instruction QA")
    p.add_argument("--base-ckpt", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--train-jsonl", type=str, default="data/qa/qa_train.jsonl")
    p.add_argument("--val-jsonl", type=str, default="data/qa/qa_val.jsonl")
    p.add_argument("--out-dir", type=str, default="out/lora")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--dropout", type=float, default=0.05)

    # training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--max-minutes", type=float, default=55.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--sample-prompts", type=str, nargs="+", default=[
        "Is the death cap mushroom poisonous?",
        "How can I tell a chanterelle from a jack-o'-lantern mushroom?",
        "What should I do if someone eats a poisonous mushroom?",
        "Are morels safe to eat?",
    ])
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device, dtype) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda" and dtype == torch.bfloat16)):
            _, loss = model(x, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def answer(model, tokenizer, question, device, max_new_tokens=80, temperature=0.7, top_k=40):
    model.eval()
    prompt = f"<|user|>{question}<|assistant|>"
    ids = tokenizer.encode(prompt)[-model.config.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = tokenizer.decode(gen[0].tolist())
    model.train()
    # strip the echoed prompt, cut at the end token if present
    answer_text = text[len(prompt):]
    cut = answer_text.find("<|end|>")
    if cut != -1:
        answer_text = answer_text[:cut]
    return answer_text.strip()


def main(argv=None):
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.dtype == "bf16" and device == "cuda") else torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])

    n_injected = len(inject_lora(model, r=args.r, alpha=args.alpha, dropout=args.dropout))
    stats = adapter_summary(model)
    print(f"[lora] injected {n_injected} adapters; trainable={stats['trainable']/1e6:.3f}M "
          f"({100*stats['trainable']/stats['total']:.2f}% of {stats['total']/1e6:.2f}M)")

    train_ds = InstructionDataset(args.train_jsonl, tokenizer, args.block_size)
    val_ds = InstructionDataset(args.val_jsonl, tokenizer, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: qa_pad_collate(b, args.block_size))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: qa_pad_collate(b, args.block_size))
    print(f"[data] train_pairs={len(train_ds)} val_pairs={len(val_ds)}")

    params = [p for p in model.parameters() if p.requires_grad]
    decay, no_decay = [], []
    for p in params:
        (decay if p.ndim >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    max_steps = args.epochs * len(train_loader)
    csv_path = out_dir / "lora_losses.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr", "wall_s"])

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))

    start = time.time()
    step = 0
    best_val = float("inf")
    for epoch in range(args.epochs):
        for x, y in train_loader:
            if time.time() - start > args.max_minutes * 60:
                print("[time] budget reached")
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda" and dtype == torch.bfloat16)):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                val_loss = evaluate(model, val_loader, device, dtype)
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                msg = (f"epoch={epoch} step={step}/{max_steps} loss={loss.item():.4f} "
                       f"val={val_loss:.4f} lr={lr:.2e} wall={time.time()-start:.0f}s" + (" *best*" if improved else ""))
                print(msg, flush=True)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{val_loss:.4f}", f"{lr:.6f}", f"{time.time()-start:.1f}"])
                save_lora(model, out_dir / "lora.pt", base_ckpt=args.base_ckpt, r=args.r, alpha=args.alpha)
                if improved:
                    save_lora(model, out_dir / "lora_best.pt", base_ckpt=args.base_ckpt, r=args.r, alpha=args.alpha)
            if step % 50 == 0 and step > 0:
                q = random.choice(args.sample_prompts)
                print(f"[sample] Q: {q}")
                print(f"[sample] A: {answer(model, tokenizer, q, device)}", flush=True)
            step += 1
        else:
            continue
        break

    save_lora(model, out_dir / "lora.pt", base_ckpt=args.base_ckpt, r=args.r, alpha=args.alpha)
    meta = {
        "base_ckpt": args.base_ckpt, "r": args.r, "alpha": args.alpha,
        "steps": step, "best_val": best_val, "train_pairs": len(train_ds), "val_pairs": len(val_ds),
    }
    (out_dir / "lora_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] adapters saved to {out_dir/'lora.pt'} (best_val={best_val:.4f})")


if __name__ == "__main__":
    sys.exit(main())
