# NanoLM — results

> Filled in after the actual training runs. All numbers below are real values
> from the logged runs on this machine (RTX 5060 Laptop 8GB, Python 3.12).

## Setup

- Corpus: `<size> KB` of mushroom-safety text
  (synthetic `<x> KB` + Wikipedia `<y> KB`).
- Tokenizer: self-implemented byte-level BPE, `<vocab>` vocab
  (`<merges>` merges learned).
- Model: GPT, `<n_layer>` layers × `<n_embd>` hidden × `<n_head>` heads,
  `<params>M` parameters, block size 256.
- Pretraining: `<steps>` steps, batch 32×256, AdamW lr `<lr>` (warmup + cosine),
  bf16, `~<minutes>` min on GPU.

## Pretraining loss curve

| step | train loss | val loss | tokens/s | wall (min) |
|------|-----------|----------|----------|------------|
| 0    | ~10.9     | —        | —        | 0          |
| …    | …         | …        | …        | …          |
| end  | …         | …        | …        | …          |

Figure: `out/figures/loss_curve.png`

## LoRA fine-tuning

- QA set: `<n_train>` train / `<n_val>` val instruction pairs.
- Adapters: r=8, α=16 on attention + MLP projections;
  trainable `<x>`% of `<params>`M params.
- `<steps>` steps, final train/val loss `<…>`.

## Sampling: before vs after LoRA

Prompt: `"Is the death cap mushroom poisonous?"`

**Before fine-tuning (base pretrained model):**
```
<raw output>
```

**After LoRA fine-tuning (merged model):**
```
<raw output>
```

## Reproduce

See README "Quick start". The full logs live in `out/pretrain/train.log` and
`out/lora/` (not committed; regenerate with the commands in the README).
