"""Low-Rank Adaptation (LoRA) implemented from scratch.

We wrap selected ``nn.Linear`` layers (by default the attention projections
``c_attn``/``c_proj``, optionally the MLP ``fc``/``proj``) in a ``LoRALinear``
that keeps the frozen base weight and adds a low-rank update:

    y = x W^T + (alpha / r) * (x A^T) B^T

The base weights are frozen during fine-tuning; only the small A/B matrices
are trained.  Adapters are saved as a standalone file (a few hundred KB) that
can be loaded on top of the base checkpoint, or merged back into the base
weights to produce a single plain model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """An nn.Linear with a trainable low-rank adapter and frozen base weight."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.use_bias = base.bias is not None

        self.weight = base.weight  # shared tensor, frozen
        self.bias = base.bias
        self.lora_dropout = nn.Dropout(dropout)
        # A ~ (r, in) initialized with Kaiming-uniform, B ~ (out, r) zero-init.
        # Create on the base weight's device so injection works whether the
        # model was moved to GPU before or after injection.
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        h = self.lora_dropout(x)
        adapter = (h @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + adapter * self.scaling

    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight and return a plain Linear."""
        with torch.no_grad():
            delta = self.lora_B.data @ self.lora_A.data * self.scaling
            merged = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)
            merged.weight.data.copy_(self.weight.data + delta)
            if self.use_bias:
                merged.bias.data.copy_(self.bias.data)
        return merged


DEFAULT_TARGETS = ("c_attn", "c_proj", "fc", "proj")


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> list[tuple[str, LoRALinear]]:
    """Replace matching Linear modules in-place with LoRALinear wrappers.

    Returns a list of (name, module) of the injected adapters.
    """
    injected: list[tuple[str, LoRALinear]] = []
    for name, module in list(model.named_modules()):  # snapshot before mutating
        leaf_name = name.split(".")[-1]
        if leaf_name not in targets or not isinstance(module, nn.Linear):
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        lora = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, name.split(".")[-1], lora)
        injected.append((name, lora))
    # freeze everything that is not a LoRA parameter
    for n, p in model.named_parameters():
        p.requires_grad_(("lora_" in n))
    return injected


def collect_adapters(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    adapters = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapters[name] = {"A": module.lora_A.data.clone(), "B": module.lora_B.data.clone()}
    return adapters


def save_lora(model: nn.Module, path: str | Path, base_ckpt: str, r: int, alpha: float, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "nanollm-lora-v1",
        "base_ckpt": base_ckpt,
        "config": {"r": r, "alpha": alpha},
        "adapters": collect_adapters(model),
        **extra,
    }
    torch.save(payload, path)


def load_lora(model: nn.Module, path: str | Path, r: int | None = None, alpha: float | None = None) -> dict:
    """Load a saved adapter file onto ``model``.

    The model must already have LoRALinear wrappers injected with the same
    target names and rank; matching adapters are copied in, unknown ones
    reported.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    r = r or cfg["r"]
    alpha = alpha or cfg["alpha"]
    loaded, skipped = [], []
    for name, tensors in payload["adapters"].items():
        parts = name.split(".")
        target = model
        for part in parts:
            target = getattr(target, part)
        if not isinstance(target, LoRALinear):
            skipped.append(name)
            continue
        if target.r != r:
            raise ValueError(f"adapter rank mismatch: {name} has r={target.r}, saved r={r}")
        with torch.no_grad():
            target.lora_A.data.copy_(tensors["A"])
            target.lora_B.data.copy_(tensors["B"])
        loaded.append(name)
    return {"loaded": loaded, "skipped": skipped, "payload": payload}


def merge_lora(model: nn.Module) -> None:
    """Fold every injected adapter into its base weight (in place).

    Afterwards the model is a plain GPT with normal trainable weights and no
    LoRA modules, so it can be checkpointed like a regular pretrained model.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent = model
            for part in name.split(".")[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split(".")[-1], module.merge())
    for p in model.parameters():
        p.requires_grad_(True)


def adapter_summary(model: nn.Module) -> dict:
    """Return total / trainable parameter counts after injection."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
