"""Quick GPU sanity + throughput benchmark before a long training run.

Run:  python scripts/gpu_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from nanollm.model import GPT, GPTConfig  # noqa: E402


def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("NO CUDA — CPU mode only")
        return
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("vram GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)

    torch.manual_seed(0)
    device = "cuda"
    config = GPTConfig(vocab_size=12000, block_size=256, n_layer=7, n_head=8, n_embd=512)
    model = GPT(config).to(device)
    opt = model.configure_optimizer(0.1, 6e-4, (0.9, 0.95))
    x = torch.randint(0, 12000, (32, 256), device=device)
    y = torch.randint(0, 12000, (32, 256), device=device)

    # warmup
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t0 = time.time()
    n_steps = 20
    for _ in range(n_steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_steps
    tok_per_s = 32 * 256 / dt
    print(f"step time: {dt*1000:.1f} ms | throughput: {tok_per_s:.0f} tokens/s")
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | vram used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
