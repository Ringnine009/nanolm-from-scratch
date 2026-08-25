"""Pretraining loop for the from-scratch GPT.

Features: AdamW (separate decay groups), warmup + cosine LR schedule, grad
clipping, bf16 autocast, train/val evaluation, checkpoint save/resume, a wall
clock time budget (``--max-minutes``), and sample text generation during
training so progress is visible in the log.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from nanollm.data import CorpusBatcher
from nanollm.model import GPT, GPTConfig, default_config
from nanollm.tokenizer import BPETokenizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pretrain NanoLM GPT")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--out-dir", type=str, default="out/pretrain")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--init-from", type=str, default=None, help="resume from a checkpoint path")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=1337)

    # model
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=7)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-sdpa", action="store_true", help="use the manual attention instead of SDPA")

    # optimization
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--max-minutes", type=float, default=120.0, help="hard wall-clock budget")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))

    # evaluation / logging
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--sample-interval", type=int, default=500)
    p.add_argument("--sample-prompts", type=str, nargs="+", default=[
        "Mushroom hunting is",
        "The death cap mushroom",
        "To identify a mushroom,",
        "If you suspect mushroom poisoning,",
    ])
    p.add_argument("--sample-tokens", type=int, default=64)
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--sample-top-k", type=int, default=40)
    return p.parse_args(argv)


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, batcher, eval_iters: int, block_size: int, dtype, device) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = batcher.get_batch(split, random.Random(0))
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda" and dtype == torch.bfloat16)):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_tokens, temperature, top_k, device) -> list[str]:
    model.eval()
    samples = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)[-model.config.block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        gen = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        text = tokenizer.decode(gen[0].tolist())
        samples.append(f"### {prompt!r}\n{text}")
    model.train()
    return samples


def main(argv=None):
    args = parse_args(argv)
    # Windows consoles default to GBK; sampled text may contain non-GBK chars
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (args.dtype == "bf16" and device == "cuda") else torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_csv = out_dir / "losses.csv"

    # tokenizer + vocab size
    tokenizer = BPETokenizer.load(args.tokenizer)
    vocab_size = args.vocab_size or tokenizer.vocab_size

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_sdpa=not args.no_sdpa,
    )

    model = GPT(config).to(device)
    optimizer = model.configure_optimizer(args.weight_decay, args.lr, tuple(args.betas))
    step, best_val = 0, float("inf")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        # a merged/plain checkpoint has no optimizer state -> start fresh
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"[resume] step={step} best_val={best_val:.4f} from {args.init_from}"
              + ("" if ckpt.get("optimizer") is not None else " (fresh optimizer)"))
    else:
        latest = out_dir / "latest.ckpt"
        if latest.exists():
            ckpt = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            step = ckpt.get("step", 0)
            best_val = ckpt.get("best_val", float("inf"))
            print(f"[resume] auto-resumed from {latest} step={step}")

    batcher = CorpusBatcher(args.data_dir, args.batch_size, args.block_size, device)
    n_params = model.num_parameters()
    print(f"[init] device={device} dtype={dtype} params={n_params/1e6:.2f}M "
          f"vocab={vocab_size} layers={args.n_layer} heads={args.n_head} embd={args.n_embd}")
    print(f"[init] data=train.bin+val.bin steps={args.max_steps} max_minutes={args.max_minutes} "
          f"lr={args.lr} bs={args.batch_size} bs_tokens={args.batch_size*args.block_size}")

    # write run metadata
    (out_dir / "config.json").write_text(json.dumps({
        "args": vars(args), "config": asdict(config), "device": device, "dtype": str(dtype),
    }, indent=2))

    csv_fields = ["step", "train_loss", "val_loss", "lr", "tokens_per_s", "wall_s"]
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(csv_fields)

    start_time = time.time()
    tokens_seen = 0
    train_loss, val_loss = float("nan"), float("nan")

    def log_line(msg):
        line = f"[step {step:>6}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    while step < args.max_steps:
        # wall-clock time budget
        if time.time() - start_time > args.max_minutes * 60:
            log_line("time budget reached; stopping")
            break

        # evaluation
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, batcher, args.eval_iters, args.block_size, dtype, device)
            train_loss, val_loss = losses["train"], losses["val"]
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            log_line(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} best_val={best_val:.4f}"
                     + (" (best)" if improved else ""))
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_val": best_val,
                "config": asdict(config),
                "tokenizer_path": args.tokenizer,
            }
            torch.save(ckpt, out_dir / "latest.ckpt")
            if improved:
                torch.save(ckpt, out_dir / "best.ckpt")
        if step % args.sample_interval == 0:
            for s in generate_samples(model, tokenizer, args.sample_prompts, args.sample_tokens,
                                      args.sample_temperature, args.sample_top_k, device):
                log_line(s.replace("\n", "\\n"))

        # one optimization step
        x, y = batcher.get_batch("train", random.Random(step))
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda" and dtype == torch.bfloat16)):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += x.numel()
        if step % args.log_interval == 0:
            dt = time.time() - start_time
            tps = tokens_seen / max(dt, 1e-6)
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{val_loss:.4f}", f"{lr:.6f}", f"{tps:.0f}", f"{dt:.1f}"])
            log_line(f"loss={loss.item():.4f} lr={lr:.2e} tok/s={tps:.0f} wall={dt/60:.1f}min")
        step += 1

    # final checkpoint
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": asdict(config),
        "tokenizer_path": args.tokenizer,
    }
    torch.save(ckpt, out_dir / "latest.ckpt")
    if best_val < ckpt["best_val"] or not (out_dir / "best.ckpt").exists():
        torch.save(ckpt, out_dir / "best.ckpt")
    log_line(f"done at step={step} best_val={best_val:.4f}")
    print(f"[done] checkpoints in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
