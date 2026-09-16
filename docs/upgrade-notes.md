# Upgrade notes — evaluation protocol v2, KV-cache decode, trainer/security fixes

Every item below follows the same shape: **what was wrong → the failing test
that pinned it (run and seen to fail) → what changed → the number that came
out → what it is worth saying about it in an interview**.

Baseline before this work: `pytest tests/ -q` → **54 passed**. After:
**82 passed** (54 original + 28 new). No original test was weakened; the two
test-side changes are called out explicitly at the end.

---

## A1. D2 — substring "hits" inflated the held-out score

**Problem.** `scripts/evaluate.py` scored with `kw in answer.lower()`. Pure
substrings counted as knowledge:

| gold keyword | answer word that "hit" it | verdict |
|--------------|---------------------------|---------|
| `cook` | *cooking* | inflection, not the word |
| `no` | *pois**no**us*, *k**no**wn* | unrelated |
| `no` | *North America* | unrelated |
| `edible` | ***in**edible* | **dangerous: the opposite meaning** |
| `ring` | *sp**ring*** | unrelated |
| `pine` | *p**ine*** in *pink* | unrelated |

Measured on the **published** result file (`out/eval_results.json`): v1 counted
17/44 = 38.6%; the same answers under word-boundary matching give 14/44 =
**31.8%** — i.e. 3 of the 17 published hits were pure noise.

**Failing test.** `tests/test_eval_protocol.py::test_substring_is_not_a_keyword_hit`,
`::test_unrelated_substring_is_not_a_keyword_hit`,
`::test_true_word_hit_still_counts` (the last one is the reverse check: a real
standalone `cook` must still count, so the rule is not tightened past truth).
Initially red: `ModuleNotFoundError: nanollm.eval_protocol`.

**Change.** New module `nanollm/eval_protocol.py`: keywords are token sequences
matched on word boundaries; gold may declare explicit surface variants with
`a|b|c` (nothing is stemmed silently, and the eval-set keywords were left
untouched). `scripts/evaluate.py` now uses it.

**Numbers.** v1 substring scoring on the new answers: 43.2 / 38.6 / 38.6 %
(mean 40.2%). v2 strict: 36.4 / 29.5 / 25.0 % (mean **30.3%**). A labelled
sensitivity view (`keyword_hit_inflected`, reported separately, never as the
headline) restores regular inflections only — `vomit`←*vomiting* yes,
`edible`←*inedible* no, `no`←*known* no — and gives mean **34.8%**.

**Talking point.** "My first number was 38.6%; a word-boundary audit showed 3 of
17 hits were substrings, one of them `edible` inside *inedible* — the metric was
rewarding an answer that said the opposite. The honest number is 30.3%, and I
kept the deprecated metric next to it so the change is auditable instead of
silent."

## A2. D3 — `polarity` was written but never read

**Problem.** All 44 items carry `polarity: yes/no` (10 of them set). The v1
evaluator never read the field, so "is the death cap safe to eat?" answered with
"yes, it is edible" scored exactly like a correct refusal.

**Failing test.** `tests/test_eval_protocol.py::test_reversed_polarity_is_not_correct`
(red before the module existed) plus the reverse checks
`::test_aligned_polarity_still_scores_correct`,
`::test_negated_affirmation_counts_as_negative_stance`
("*not edible* must not read as *edible*"), and
`::test_unspecified_polarity_is_never_penalised`.

**Change.** `detect_polarity()` classifies an answer's stance with negation
scope (a negation within three tokens before an affirmation flips it), and
`score_item()` reports `hit`, `polarity_ok` and `correct = hit and not
contradicted`. An answer with no stance cue is recorded as *unverified* — never
counted as a violation, and never silently as a correct stance.

**Numbers.** In the real runs, v2 flags **5 polarity violations across 3 runs**
(e.g. *"Are golden chanterelles good to eat?"* → *"…must never grow, and never
be eaten"*, and *"Has the yellow knight been found dangerous to eat?"* →
*"…formerly considered edible"*). Mean `correct` = 29.5% vs mean `hit` = 30.3%.

**Talking point.** "The eval set already had a polarity field; the evaluator
ignored it, so flipping the medical advice was free. It now costs the item —
and I report `correct` separately from `hit` so recall and stance never get
mixed into one number."

## A3. D1 — the score depended on the device

**Problem.** `torch.Generator(device=device)` inside `generate_tokens`: the same
seed produces a different random stream on CPU and CUDA. With the real
checkpoint and seed 3:

```
CPU : ", the death cap is deadly poisonous. Amatoxins, chiefly"
CUDA: " regions is edible, the only after thorough cooking. Gastrointestinal irritants"
```

The published 38.6% was a CUDA artefact; the same script and checkpoint on CPU
scored 52.3%.

**Failing test.** `tests/test_reproducibility.py::test_same_seed_gives_identical_tokens_on_cpu_and_cuda`
(red: the outputs differed) and `::test_greedy_temperature_zero_is_argmax`
(red with `RuntimeError: probability tensor contains either inf, nan or element
< 0` — `temperature=0` divided by zero in v1).

**Change.** The random draw comes from a **CPU** generator (`torch.rand(1,
generator=rng)`), then a `torch.searchsorted` on the cumulative distribution —
on whichever device the logits live. `temperature <= 0` now means greedy argmax.
The decode benchmark then exposed a second defect in this new path: a bf16
cumsum stalls below 1.0 on CUDA (`cdf[-1] = 0.99609375`), `searchsorted` returns
`vocab_size`, and the CUDA index kernel aborts with *"vectorized gather kernel
index out of bounds"*. Fixed by normalising in float32 and clamping
(`_sample_from_cdf`), with `tests/test_sampling_bounds.py` pinning it — including
a test that first proves the unclamped computation really does go out of range
on the fixture, so the regression test cannot go vacuous.

**Numbers.** CPU and CUDA now agree completely: **132/132 byte-identical
answers** (3 runs × 44 items), identical scores 30.3%. `reproduce_v1_eval.py`
re-runs the *old* code from commit `0c4ed5d` and reproduces the audit exactly:
CUDA 38.6%, CPU 52.3%.

**Talking point.** "Reproducibility wasn't a documentation problem, it was a
`torch.Generator(device=…)` call: the eval was measuring the RNG device as much
as the model. After moving randomness to a CPU stream and sampling by inverse
CDF, 132/132 answers are byte-identical across devices — and the same change
surfaced a bf16 out-of-vocabulary crash that only appeared once the cache made
bf16 decoding fast enough to reach it."

## A4. D5 — no baseline, no interval

**Problem.** n=44 (95% CI ≈ ±14 pp) was reported as a point estimate and
compared against nothing. There was no floor, no retrieval baseline, no seed
variation.

**Failing test.** `tests/test_eval_protocol.py::test_report_includes_baselines_and_ci`,
`::test_keyword_lookup_baseline_is_deterministic_and_uses_train_answers` (red:
no such functions existed).

**Change.** `build_report()` always carries (a) the empty-answer floor,
(b) a **keyword-lookup baseline** — every eval question answered with the stored
answer of the most similar *training* question — and (c) Wilson 95% intervals;
`--seeds 0,1,2` adds per-seed values, mean, spread and a pooled interval.

**Numbers.** empty 0.0%; keyword lookup **59.1%** (correct 56.8%); model 30.3%
(95% CI [23.1, 38.6], per-seed spread 4.7 pp).

**Talking point.** "The baseline was the uncomfortable part: a lookup table over
the fine-tuning answers scores 59.1% against the model's 30.3%. Combined with
133 epochs over 8.7% of a Chinchilla budget, the honest conclusion is that the
fine-tune taught answer *format* over memorized phrasing, not knowledge — and
the model only looked useful because no baseline was measuring that."

## A5. D4 — what is still held out (documented, not fixed)

The 44 items hold out question **phrasings**: 25 of the 26 entities also appear
in the fine-tuning QA. A knowledge-level split needs re-fine-tuning on
unseen-entity QA, which is out of scope here; `docs/results.md` states the
limitation explicitly and the keyword-lookup baseline quantifies it.

## B1. D8 — KV cache + vectorised post-processing

**Problem.** No KV cache (every step re-ran the whole window, O(T²) attention),
and the logit post-processing was pure Python: `_no_repeat_mask` looped over all
12,000 vocabulary entries, `_apply_repetition_penalty` indexed a GPU tensor once
per context token — each `row[tid] > 0` comparison synchronising the device.
Measured: repetition penalty **11.77 ms/step on CUDA** (the audit's ≈12 ms claim,
reproduced), i.e. more than the forward pass itself.

**Failing tests.** `tests/test_kv_cache.py`:
`::test_forward_cached_matches_full_forward_logits` (red: `AttributeError:
'GPT' object has no attribute 'forward_cached'`),
`::test_greedy_token_ids_identical_with_and_without_cache` — greedy decoding must
be **token-for-token identical**, including across the sliding window when the
sequence exceeds `block_size` — and
`::test_generate_tokens_cache_matches_uncached_with_sampling` for the shipped
path with penalty + no-repeat + top-k sampling.

**Change.** `CausalSelfAttention`/`Block`/`GPT` accept `past_kv` and return the
updated cache (`GPT.forward_cached`); the cached path uses an explicit boolean
mask for non-square (T>1 with history) attention and re-prefills when the window
slides, so positions restart exactly like the uncached path. `generate_tokens`
prefills once and then feeds single tokens; post-processing is vectorised
(`unfold`-based n-gram banning, one gather/scatter for the penalty).

**Numbers** (`scripts/benchmark_generation.py`, batch 1, 128 tokens, best of 6
round-robin rounds):

| configuration | CPU fp32 | CUDA fp32 | CUDA bf16 |
|---------------|----------|-----------|-----------|
| v1 (`0c4ed5d`) | 77.5 tok/s | 82.3 tok/s | 96.2 tok/s |
| vectorised post-processing only | 86.7 (1.12×) | 266.4 (3.24×) | 290.8 (3.02×) |
| **KV cache + vectorised** | **205.0 (2.65×)** | **301.1 (3.66×)** | **288.1 (3.00×)** |
| KV cache, no post-processing | 217.1 | 351.2 | 317.2 |

Post-processing per step: CUDA 14.41 → 0.50 ms; CPU 2.56 → 0.10 ms.

**Honest reading.** bf16 does not beat fp32 at batch 1 (288 vs 301 tok/s):
single-sequence decode is latency-bound. With the cache, post-processing is
~0.5 ms/step, so the remaining cost is the forward pass — the next step would be
CUDA graphs or batched serving, not more Python tuning.

**Talking point.** "The speed-up is only worth anything because cached and
uncached decoding are proven token-for-token identical — including when the
context window slides. And the benchmark found a real bug in my own new code:
bf16 cumsum on CUDA stalls below 1.0, so `searchsorted` returned `vocab_size`
and the index kernel aborted; the fix is a float32 normalisation plus a clamp,
pinned by a test that first proves the old path really goes out of range."

## C1. D9 — `train.py` had a dead condition and could publish unvalidated weights

**Problem.** `if best_val < ckpt["best_val"] or not (out_dir / "best.ckpt").exists()`
compares `best_val` with itself (the dead half) — leaving "best.ckpt does not
exist yet" as the only live branch. A run that never evaluated (no steps, or a
resume between two eval boundaries) therefore wrote an **unevaluated** model to
`best.ckpt`, which then becomes the LoRA base checkpoint.

**Failing tests.** `tests/test_train_best_ckpt.py::test_never_publishes_an_unevaluated_best_ckpt`
(red: `best.ckpt` existed with `step == 0` and no validation signal) and
`::test_final_eval_runs_when_no_in_loop_eval_fires` (red: no evaluation happened
at all in a resume run that took steps).

**Change.** Track whether any evaluation ran; if the loop ends after steps
without one, evaluate once at the end; never write `best.ckpt` without a
validation signal (it logs an explicit warning instead), and keep `latest.ckpt`
as the unconditional final state. The dead comparison is gone.

**Talking point.** "`best_val < best_val` is always false, so the only live
branch was *the file doesn't exist* — the trainer would happily name an
unevaluated checkpoint 'best'. It now runs a final evaluation when the loop
never evaluated, and refuses to publish an unvalidated file."

## C2. D11 — licence inconsistency

`docs/results.md` described the Wikipedia extracts as "Apache-2.0" while
`data/corpus/NOTICE.md` and the `corpus.txt` header say **CC BY-SA 4.0**. The
docs now say CC BY-SA 4.0 and record that the earlier revision was wrong.
While checking numbers against the logs, one more mismatch was fixed: the
Chinese summary claimed the pretraining loss fell to **1.03**, whereas
`out/pretrain/losses.csv` ends at 0.10 (step 5900: 0.1206) — it now says 0.10.

## C3. Missing figures: epochs and Chinchilla budget

Added to `docs/results.md` with formulas: 6,000 × 32 × 256 = **49.15 M tokens**
over 369,436 training tokens = **133.0 epochs**, and **8.7%** of a
Chinchilla-optimal budget (20 × 28,302,848 = 566 M). This is the small-data
high-epoch overfitting regime, visible in the twin curves (val bottoms at 5.02
at step 1200, then rises to 7.16).

## C4. Safe checkpoint loading (security)

**Problem.** This was not in the original audit: `scripts/evaluate.py` still
called `torch.load(..., weights_only=False)` — a second call site missed by
`e57bd9c`, which hardened the training path. Four scripts and six test modules
were affected. Unpickling an untrusted `.pt` file with `weights_only=False`
executes arbitrary code.

**Failing test.** `tests/test_safe_loading.py::test_no_call_site_loads_checkpoints_without_weights_only`
(red: it listed `scripts/evaluate.py`, `scripts/compare_samples.py`,
`scripts/benchmark_generation.py`, `scripts/reproduce_v1_eval.py`,
`tests/test_generation.py`) and
`::test_load_checkpoint_rejects_arbitrary_pickle`, which saves a checkpoint
whose payload is `os.system("echo pwned > pwned.txt")` and asserts the loader
refuses it (red: `ModuleNotFoundError: nanollm.checkpoints`).

**Change.** New `nanollm/checkpoints.py::load_checkpoint()` — the single place
checkpoints are read, always `weights_only=True`, with an optional
`required=(...)` key check. All four scripts now use it; the test modules pass
the flag explicitly. Every project checkpoint is a dict of tensors and plain
types, so `weights_only=True` loads them unchanged — nothing was loosened to
make a test pass, and the repo-wide scan enforces it going forward.

## C5. Dependency pinning (D12)

`requirements.txt` now pins the exact versions the published numbers were
produced with (torch 2.11.0, numpy 2.4.6, regex 2026.2.28, fastapi 0.141.1,
uvicorn 0.50.2, pydantic 2.13.4, httpx 0.28.1, requests 2.32.5, matplotlib
3.10.8, pytest 9.1.1), with a comment on installing the `+cu128` wheel.

---

## Test-suite changes (and why they are not weakening)

- **54 → 82 tests.** Every new test module was written and executed before its
  implementation and failed first: `test_eval_protocol.py`
  (`ModuleNotFoundError: nanollm.eval_protocol`), `test_kv_cache.py`
  (5 failures: `AttributeError: … 'forward_cached'`, plus the `temperature=0`
  division-by-zero crash), `test_reproducibility.py` (device divergence +
  the same crash), `test_train_best_ckpt.py` (`best.ckpt` written with
  `step == 0`; no final evaluation), `test_safe_loading.py`
  (`ModuleNotFoundError: nanollm.checkpoints` + a repo scan listing four unsafe
  scripts), `test_sampling_bounds.py` (`ImportError: _sample_from_cdf`).
  `test_sampling_bounds.py` was later *strengthened* after the fixture turned
  out not to reproduce the hazard on CPU (the bf16 cumsum stall is CUDA-only);
  the final version proves the unclamped path really does sample out of range
  on the fixture before asserting the fixed path stays in range.
- Pre-existing tests were touched in two ways only: flipping
  `weights_only=False` → `True` in fixture loading (`tests/test_generation.py`,
  `tests/test_train_smoke.py`, `tests/test_kv_cache.py`,
  `tests/test_reproducibility.py`, `tests/test_lora_roundtrip.py`) — no
  assertion changed — and correcting three assertions that were wrong about
  the fixture data, each of which made the check stricter:
  1. `test_true_word_hit_still_counts` used an answer without the gold word
     (`"No, never eat them raw."` for gold `cook`), so it tested nothing; it now
     uses `"No, cook them first."`;
  2. the n=44 interval expectation was moved onto `wilson_ci(22, 44)` (the
     k=0 report has a 0.0–8.0% interval, which is correct but not what the
     width assertion was about);
  3. the v1-comparison test asserted against eval item 0 instead of item 6
     (the morels question, whose gold keywords are `["cook", "raw", "nausea"]`);
     it now checks that the item still hits v2 through the real word `raw`
     while v1 additionally credited the substring inside *cooking*.
  `substring_only_hits` gained a `surfaces` field (which word produced the
  substring match) — an added evidence trail, and the corresponding assertion
  was updated to expect it.
