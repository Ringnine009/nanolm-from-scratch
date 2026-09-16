"""A GPT-style decoder-only transformer built from scratch in PyTorch.

Structure follows the classic nanoGPT design (which itself mirrors GPT-2):
token + positional embeddings -> N transformer blocks (LayerNorm -> causal
multi-head self-attention -> MLP with GELU) -> final LayerNorm -> tied lm
head.  Everything here is hand-written; the only optional accelerator is
``torch.nn.functional.scaled_dot_product_attention`` (a fused kernel), which
we verify against the manual attention implementation in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 12000
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False          # no biases in Linear/LayerNorm (GPT-2 style)
    use_sdpa: bool = True       # fused flash/mem-efficient attention kernel
    tie_embeddings: bool = True # share input embedding with the lm head


def default_config() -> GPTConfig:
    """Default pretraining configuration (~28M parameters)."""
    return GPTConfig(vocab_size=12000, block_size=256, n_layer=7, n_head=8, n_embd=512)


def _cached_attn_mask(q_len: int, kv_len: int, device) -> torch.Tensor:
    """Boolean mask for a *non-square* cached attention call: query ``i`` holds
    absolute position ``kv_len - q_len + i`` and may attend to keys up to it."""
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    return k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_sdpa = config.use_sdpa

        # c_attn produces q, k, v stacked along the feature axis (GPT-2 style)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        """Causal self-attention with an optional incremental KV cache.

        ``past_kv`` is a ``(k, v)`` pair of the keys/values computed for earlier
        positions (shape ``(B, n_head, past_len, head_dim)``); ``use_cache``
        makes the layer return ``(y, (k, v))`` so the caller can keep it.  With
        ``past_kv=None`` and ``use_cache=False`` this is the plain v1 path.
        """
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        if self.use_sdpa:
            # q_len == 1 with a cache attends to every cached key; a fresh
            # (square) call is plain causal attention.
            mask = None if (past_kv is None or T == 1) else _cached_attn_mask(T, k.size(2), x.device)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=(past_kv is None),
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if past_kv is None:
                att = att.masked_fill(self.tril[:T, :T] == False, float("-inf"))  # noqa: E712
            else:
                keep = _cached_attn_mask(T, k.size(2), x.device)
                att = att.masked_fill(~keep, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_head, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return (y, new_kv) if use_cache else y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        if use_cache:
            attn_out, new_kv = self.attn(self.ln_1(x), past_kv, True)
        else:
            attn_out, new_kv = self.attn(self.ln_1(x), past_kv, False), None
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return (x, new_kv) if use_cache else x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        if config.tie_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections by 1/sqrt(2*n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits, loss, _ = self._forward_impl(idx, targets)
        return logits, loss

    def _forward_impl(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv=None,
        use_cache: bool = False,
    ):
        B, T = idx.shape
        past_len = 0 if past_kv is None else past_kv[0][0].size(2)
        assert past_len + T <= self.config.block_size, (
            f"sequence longer than block_size {self.config.block_size} "
            f"(cache {past_len} + new {T})"
        )
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        new_kv = [] if use_cache else None
        for i, block in enumerate(self.h):
            layer_past = None if past_kv is None else past_kv[i]
            if use_cache:
                x, kv = block(x, layer_past, True)
                new_kv.append(kv)
            else:
                x = block(x, layer_past, False)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss, (tuple(new_kv) if use_cache else None)

    def forward_cached(self, idx: torch.Tensor, past_kv=None):
        """Incremental forward pass for generation: returns ``(logits, new_kv)``.

        ``logits`` covers the ``idx`` positions (take ``[:, -1, :]`` to sample
        the next token); ``new_kv`` is the per-layer ``(k, v)`` cache to hand to
        the next call.  Feeding one token at a time with this cache produces the
        same distribution as re-running the whole window, for O(T) instead of
        O(T^2) attention work per step.
        """
        logits, _, kv = self._forward_impl(idx, None, past_kv, use_cache=True)
        return logits, kv

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with temperature scaling and top-k filtering."""
        rng = torch.Generator(device=idx.device)
        if seed is not None:
            rng.manual_seed(seed)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)[0][:, -1, :]  # (B, V)
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def configure_optimizer(self, weight_decay: float, learning_rate: float, betas: tuple[float, float]):
        """AdamW with separate weight-decay groups (nanoGPT style)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:  # weight matrices decay, biases/LayerNorm don't
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)
