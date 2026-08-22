# NanoLM — results

All numbers below are from actual runs on this machine:
**RTX 5060 Laptop 8GB · Python 3.12 · torch 2.11.0+cu128** (see
`out/pretrain/train.log` and `out/lora/lora_losses.csv`; logs are regenerable
via the README commands, they are not committed).

## Setup

- **Corpus:** 1.50 MB of mushroom-safety text = 242 KB synthetic fact sheets
  (this repo, `scripts/kb.py`) + 1.13 MB Simple-English Wikipedia article
  extracts (Apache-2.0, fetched via the HF mirror, see
  `data/corpus/NOTICE.md`).
- **Tokenizer:** self-implemented byte-level BPE, vocab **12,000**
  (11,741 merges learned on the corpus, ~110 s), 373,167 tokens total
  (369,436 train / 3,731 val).
- **Model:** from-scratch GPT — 7 layers × 512 hidden × 8 heads,
  **28.30M parameters**, block size 256, tied embeddings.
- **Pretraining:** 6,000 steps, batch 32×256, AdamW lr 6e-4 (warmup 300 +
  cosine to 6e-5), weight decay 0.1, grad clip 1.0, bf16 autocast.
  Wall time **28.1 min**, throughput 29–87 K tokens/s, peak VRAM ~2.9 GB.
- **LoRA fine-tuning:** r=8, α=16 on all attention+MLP projections
  (28 adapters, 0.459M trainable = **1.60%** of the model), lr 3e-4,
  QA set 442 train / 49 val pairs, 1,680 steps ≈ 3 min.
  *Honest note:* the QA set is mostly **template-generated** from `kb.py`
  (~410 pairs) plus ~34 **hand-written** pairs (spot-checked by the
  maintainer, **not** a specialist review). The template-generated pairs were
  not individually fact-checked against a mushroom expert.

## Pretraining loss curve

| step | train loss | val loss | notes |
|------|-----------|----------|-------|
| 0    | 9.50      | 9.50     | ≈ ln(12000), expected random-init |
| 300  | 4.89      | 5.86     | fast first drop |
| 600  | 3.29      | 5.39     | |
| 900  | 2.01      | 5.14     | |
| 1200 | 1.03      | **5.02** | **best val** (checkpoint `best.ckpt`) |
| 3000 | 0.35      | 6.19     | overfitting on the tiny corpus |
| 6000 | 0.10      | 6.90     | end of run |

The classic small-data signature: the model quickly memorizes the training
text (train loss → 0.1) while validation loss bottoms out early. We fine-tune
from the best-val checkpoint.

## LoRA fine-tuning (from best.ckpt)

| metric | start | end |
|--------|-------|-----|
| train loss | 3.78 | 0.31 |
| val loss | 4.03 | **0.84** |

Adapters are 1.8 MB standalone (`out/lora/lora_best.pt`) vs a 113 MB base
checkpoint; after merging, `checkpoints/merged.pt` is a single 28.3M-param
plain model ready to serve.

## Sampling: before vs after LoRA

Prompt: `"Is the death cap mushroom poisonous?"` (wrapped in the
`<|user|>…<|assistant|>` instruction format; temperature 0.7, top-k 40, seed 3).

**Before fine-tuning (base pretrained model, `best.ckpt`):**
```
before the death when the only a mushroom market. It is usually does not made
by a group of a poisonous or a full color. It can be too small, but this is too
much more than any way when it is no symptoms of a doctor or too much when it
is too much to learn, and help with a person feels. The symptoms of a person
feels when they have a person with the same or see if they have to make them.
They may have a person. They may make a person, or too much person with any
```
→ repetitive, incoherent, does not answer the question.

**After LoRA fine-tuning (merged model, served by the FastAPI server):**
```
the death cap is deadly poisonous and should never be eaten. Amatoxins, chiefly
alpha-amanitin, which block rna polymerase ii and destroy liver cells. Delayed
poisoning: 6-24 hours after eating, violent cramps, vomiting and watery
diarrhea begin, followed by a deceptive improvement and then liver and kidney
failure. Contact a poison control center immediately if it is eaten.
```
→ structured, factual, follows the instruction format.

**More after-LoRA samples** (same settings):
- "Are morels safe to eat?" → *"…only after thorough cooking. Raw or
  undercooked morels contain heat-sensitive compounds that cause nausea,
  vomiting and cramps."*
- "How can I tell a chanterelle from a jack-o'-lantern mushroom?" → *"…the
  jack-o'-lantern mushroom is poisonous and should never be eaten… violent
  vomiting and diarrhea starting 30 minutes to 3 hours after eating."*
- "What should I do if someone eats a poisonous mushroom?" → *"…violent
  vomiting and diarrhea a few hours after eating. The toxins involved are
  gastrointestinal irritants. Seek medical help promptly if symptoms appear."*

## Held-out evaluation (merged model)

A **disjoint held-out set** of 44 mushroom-safety QA items was built by hand
(`scripts/build_eval_set.py`): every question is phrased differently from all
491 train/val questions, and the word-set Jaccard similarity to the nearest
training question is < 0.75 (checked programmatically in `tests/test_eval.py`).
Answers are generated with the unified generation module (temp 0.7, top-k 40,
repetition penalty 1.15, no-repeat 4-gram, `<|end|>` stop, seeded per item).
Scoring is keyword-based (any / all expected keywords appear, case-insensitive).

| metric | value |
|--------|-------|
| **hit (any expected keyword)** | **38.6%** (17/44) |
| **hit (all expected keywords)** | **9.1%** (4/44) |

Per category (hit_any / hit_all):

| category | n | hit_any | hit_all |
|----------|---|---------|---------|
| edibility yes/no | 10 | 80.0% | 20.0% |
| identification | 8 | 12.5% | 12.5% |
| habitat | 8 | 25.0% | 0.0% |
| symptoms | 8 | 37.5% | 12.5% |
| first-aid / general | 10 | 30.0% | 0.0% |

Honest reading: the LoRA-finetuned model reliably answers **yes/no edibility**
questions (80% contain the expected fact), but **specific factual recall is
weak** (e.g. "what color are the gills", "which toxin") — 27/44 answers
contain none of the expected keywords. The model is a mechanism/method
demonstration, not a reliable QA system. Full per-item results:
`out/eval_results.json` (regenerate with `python scripts/evaluate.py`).

## Reproduce

See README "Quick start". Figures: `out/figures/loss_curve.png`
(pretraining + LoRA curves). Full logs: `out/pretrain/train.log`,
`out/pretrain/losses.csv`, `out/lora/lora_losses.csv`.

> ⚠️ Educational demonstration only — never use NanoLM outputs for real
> mushroom identification or medical decisions.
