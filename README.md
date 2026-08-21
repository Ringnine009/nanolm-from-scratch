# NanoLM — a from-scratch GPT, fine-tuned with self-implemented LoRA, served locally

**Train a small GPT from zero → instruction-fine-tune it with your own LoRA →
chat with it over a local FastAPI + SSE web UI — with zero API cost and zero
cloud dependency.**

NanoLM is a hands-on demonstration of the full language-model stack: a
byte-level BPE tokenizer trained on the corpus, a GPT-style transformer built
from scratch in PyTorch (embedding, causal multi-head attention, feed-forward,
LayerNorm, GPT-2 style weight init), a pretraining loop (AdamW, warmup +
cosine schedule, gradient clipping, bf16 autocast, checkpoint/resume), a
self-implemented LoRA fine-tuning pass on a mushroom-safety QA instruction
set, and a FastAPI server that streams answers token-by-token over
Server-Sent-Events.

**Real-world scenario:** a zero-API-cost offline domain chatbot for mushroom
safety — designed to slot into the offline tier of the *MycoGuard* mushroom
safety assistant. Everything runs on a laptop GPU with no external service.

> ⚠️ **Safety note:** NanoLM is an *educational* demonstration of deep-learning
> techniques. It is **not** a mushroom identification tool and must never be
> used for real foraging or medical decisions.

---

## Why does this exist / what does it prove?

| | |
|---|---|
| **Model internals** | Full GPT implementation from scratch: you can read every line — no `transformers` model classes, no hidden attention. |
| **Tokenizer** | A self-implemented byte-level BPE (subword-nmt style merge learning), trained on the corpus — no `tiktoken` dependency. |
| **Fine-tuning** | LoRA written from scratch: low-rank adapters injected into attention projections, base weights frozen, adapters saved as a tiny standalone file and mergeable back into the base. |
| **Serving** | FastAPI + SSE streaming chat with a small web UI, plus an interactive CLI. |
| **Reproducibility** | One command each for corpus → tokenizer → pretrain → finetune → serve. |

## Repository layout

```
nanollm/
├── nanollm/
│   ├── tokenizer.py     # byte-level BPE: training, encode/decode, special tokens
│   ├── data.py          # token loading, pretrain batching, instruction datasets
│   ├── model.py         # GPT from scratch (attention, MLP, LayerNorm, init, sampling)
│   ├── train.py         # pretraining loop (AdamW, cosine LR, grad clip, ckpt, resume)
│   ├── sample.py        # CLI sampling (temperature / top-k)
│   ├── lora.py          # LoRALinear injection, save/load/merge
│   ├── finetune.py      # LoRA instruction fine-tuning
│   ├── merge.py         # fold LoRA into the base checkpoint
│   ├── chat.py          # interactive CLI chat
│   └── server/          # FastAPI + SSE chat server + web UI (static/)
├── scripts/
│   ├── kb.py            # mushroom-safety knowledge base (species, toxins, rules…)
│   ├── build_corpus.py  # synthetic corpus + best-effort Wikipedia/Gutenberg fetch
│   ├── fetch_wikipedia_hf.py  # mushroom articles via HuggingFace datasets mirror
│   ├── build_qa.py      # instruction QA set (generated + manually proofread)
│   ├── prepare_data.py  # train tokenizer, tokenize corpus → train.bin/val.bin
│   └── plot_loss.py     # loss-curve figure from the CSV logs
├── data/
│   ├── corpus/          # corpus sources (see NOTICE.md for licensing)
│   └── qa/              # qa_train.jsonl / qa_val.jsonl
├── configs/             # default pretrain / lora configs
├── tests/               # pytest suite (tokenizer, model, sampling, LoRA, smoke, API)
└── docs/results.md      # training results, loss curves, before/after samples
```

## How the pieces work

### 1. Byte-level BPE tokenizer (self-implemented)

We chose to **implement BPE from scratch** rather than reuse `tiktoken`:

- it keeps the whole pipeline self-contained and readable (a core goal of the
  project);
- byte-level base vocabulary (256 bytes) means *any* UTF-8 text can be
  encoded — there is no out-of-vocabulary token;
- a smaller vocabulary (12k vs. GPT-2's 50k) shrinks the embedding matrix,
  which matters for a 5–30M-parameter model.

Merge learning follows the classic subword-nmt algorithm (repeatedly merge the
most frequent adjacent pair) with GPT-2-style pre-tokenization
(`'s|'t|…| ?\p{L}+|…` via the `regex` package) so merges stay inside words and
whitespace round-trips exactly. Three special tokens (`<|user|>`,
`<|assistant|>`, `<|end|>`) are reserved at the end of the vocabulary for the
instruction format.

### 2. GPT from scratch

A standard GPT-2-style decoder-only transformer (nanoGPT-inspired; see
Credits), with every component written by hand:

- token + learned positional embeddings (tied input/output embeddings);
- N blocks of LayerNorm → causal multi-head self-attention → MLP (GELU);
- causal masking so token *i* can only attend to tokens ≤ *i*;
- GPT-2 weight init (N(0, 0.02), residual projections scaled by
  1/√(2·N_layers));
- sampling with temperature scaling and top-k filtering.

The default config is ~28M parameters (7 layers × 512 hidden, 12k vocab,
256-token context). `scaled_dot_product_attention` is used as an optional fused
kernel; the manual attention path is kept and a test verifies they match.

### 3. Pretraining

`python -m nanollm.train` runs AdamW with separate decay/no-decay groups,
linear warmup + cosine decay, gradient clipping, and bf16 autocast on CUDA.
Checkpoints (model + optimizer + step) are saved every evaluation; the run
auto-resumes from `latest.ckpt` and honors a hard wall-clock budget
(`--max-minutes`). Loss and token throughput are logged to
`out/pretrain/losses.csv`; sample text is generated periodically so progress
is visible in the log.

### 4. LoRA fine-tuning (self-implemented)

`LoRALinear` wraps the attention projections (`c_attn`, `c_proj`, optionally
`fc`, `proj`): the base weights are frozen and a low-rank update
`(α/r)·(x Aᵀ) Bᵀ` is added. Only A/B are trained, so fine-tuning touches a
few hundred thousand parameters instead of the whole model. Adapters are saved
independently (`out/lora/lora.pt`, a few hundred KB) and can be:

- reloaded on top of the untouched base checkpoint, or
- merged back into the base (`python -m nanollm.merge`) to produce a single
  plain model for serving.

Only the assistant portion of each QA example is supervised (targets before
`<|assistant|>` are masked with −100).

### 5. Serving

`python -m nanollm.server.app --model checkpoints/merged.pt` starts FastAPI on
`http://127.0.0.1:8000` with:

- `POST /api/chat` — SSE stream of tokens (temperature / top-k configurable);
- `GET /api/health` — model status;
- `GET /` — a small chat UI (vanilla JS, streams into the page).

`python -m nanollm.chat --ckpt checkpoints/merged.pt` gives the same model in
the terminal.

---

## Quick start

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# GPU wheels for NVIDIA Blackwell (RTX 5060 etc.), CUDA 12.8:
pip install torch --index-url https://download.pytorch.org/whl/cu128
# China mirror tip: if download.pytorch.org is slow/unreachable, grab the
# cu128 wheel directly from https://mirrors.aliyun.com/pytorch-wheels/cu128/
# (e.g. torch-2.11.0+cu128-cp312-cp312-win_amd64.whl) and
# `pip install <downloaded.whl>`.

# 1. build the corpus (synthetic guaranteed; Wikipedia/Gutenberg best-effort)
python scripts/build_corpus.py
python scripts/fetch_wikipedia_hf.py          # optional: more real Wikipedia prose
python scripts/build_corpus.py --no-download # re-merge after fetching

# 2. train the tokenizer and tokenize the corpus
python scripts/prepare_data.py --vocab-size 12000

# 3. pretrain the GPT (time-boxed; resumes from latest.ckpt automatically)
python -m nanollm.train --data-dir data/processed --out-dir out/pretrain \
    --max-minutes 110 --batch-size 32 --block-size 256

# 4. LoRA instruction fine-tuning
python scripts/build_qa.py
python -m nanollm.finetune --base-ckpt out/pretrain/best.ckpt --out-dir out/lora

# 5. merge and serve
python -m nanollm.merge --base-ckpt out/pretrain/best.ckpt --lora out/lora/lora_best.pt \
    --out checkpoints/merged.pt
python -m nanollm.server.app --model checkpoints/merged.pt --port 8000
# open http://127.0.0.1:8000  — or chat in the terminal:
python -m nanollm.chat --ckpt checkpoints/merged.pt
```

Sampling standalone:

```bash
python -m nanollm.sample --ckpt out/pretrain/best.ckpt \
    --prompt "The death cap mushroom" --max-new-tokens 100 --temperature 0.8 --top-k 40
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers: tokenizer round-trips & specials; model shapes, causal masking,
tied embeddings, manual-vs-SDPA attention equivalence; sampling determinism
& top-k; training smoke (loss decreases, checkpoint resume); LoRA
save/load/merge round-trips and trainability; FastAPI/SSE API behaviour.

## Results

See [`docs/results.md`](docs/results.md) for the training curves, loss
numbers and sample text before/after LoRA fine-tuning. Summary figures are in
`out/figures/loss_curve.png`.

## Known limitations

- Trained on a tiny domain corpus (~0.15–2 MB); the model is a *demonstration*
  of mechanics, not a capable assistant. Answers are often plausible but
  unreliable.
- English-only; the QA set is small (~450 pairs).
- The pretrained model is sensitive to sampling settings; long generations can
  drift off-topic.
- **Never use outputs for real mushroom identification or medical decisions.**

## Credits & acknowledgements

- **Architecture inspiration:** [nanoGPT](https://github.com/karpathy/nanoGPT)
  (Andrej Karpathy, MIT) — the project follows nanoGPT's general structure
  (embedding/attention/MLP/LayerNorm, GPT-2 init, AdamW groups, checkpoint
  format). All code here is written from scratch for this project.
- **Tokenizer:** self-implemented byte-level BPE; pre-tokenization regex from
  the GPT-2/tiktoken approach (MIT, OpenAI).
- **Corpus:** see `data/corpus/NOTICE.md` — original synthetic text (this
  repo), Wikipedia extracts (CC BY-SA 4.0) when fetched, and public-domain
  Project Gutenberg books when fetched.
- **LoRA:** inspired by *LoRA: Low-Rank Adaptation of Large Language Models*
  (Hu et al., 2021, arXiv:2106.09685); implementation is original.

## License

MIT — see [LICENSE](LICENSE).

---

## 中文摘要

**NanoLM：从零训练并部署一个轻量对话模型。** 用 PyTorch 从零实现了一条完整的语言模型流水线：

1. **分词器**：自实现的 byte-level BPE（在语料上训练 11,741 次合并，词表 12,000），无需 tiktoken。
2. **模型**：从零手写 GPT（embedding / 因果多头注意力 / 前馈 / LayerNorm / GPT-2 式初始化 / 权重共享），约 **28.3M 参数**（7 层 × 512 维），支持手动注意力与融合内核（SDPA）两种实现并可验证等价。
3. **预训练**：在 1.5MB 蘑菇安全领域语料（自建知识库 + 维基百科公开文本）上训练 6,000 步（约 28 分钟，RTX 5060 Laptop），loss 从 9.50 稳定下降到 1.03（val 最低 5.02），含 AdamW、warmup+cosine 调度、梯度裁剪、bf16、checkpoint 断点续训。
4. **LoRA 微调**：自实现低秩适配器（注入 attention/MLP 全投影层，冻结基座权重，仅训练 1.6% 参数），用 442 条蘑菇安全问答对做指令微调（val loss 4.03 → 0.84），适配器独立保存（1.8MB）并可合并回基座。
5. **推理**：FastAPI + SSE 流式聊天 + 轻量网页 UI + 命令行聊天，完全本地、零 API 成本，可作为蘑菇安全助手的离线层。

**效果对比**：微调前模型无法理解指令（输出重复乱码）；微调后能输出结构化、事实正确的答案（如"毒鹅膏含 α-鹅膏蕈碱，6-24 小时后出现延迟中毒症状，应立即联系中毒控制中心"）。

**运行**：见上方英文 Quick start（`build_corpus.py` → `prepare_data.py` → `nanollm.train` → `nanollm.finetune` → `nanollm.merge` → `nanollm.server.app`）。

**安全声明**：本项目仅为深度学习教学演示，输出不可用于真实蘑菇辨识或医疗决策。
