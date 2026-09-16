"""Sampling must never produce an out-of-vocabulary token id.

Found by the decode benchmark: CUDA's bfloat16 ``cumsum`` accumulates in bf16,
so the cumulative distribution stalls on the bf16 grid below 1.0
(``cdf[-1] == 0.99609375`` on the real checkpoint).  The ~0.4% of uniform draws
above that truncated tail made ``searchsorted`` return ``vocab_size`` - one past
the end of the embedding, which the CUDA index kernel rejects with
"vectorized gather kernel index out of bounds".  (CPU ``cumsum`` accumulates in
fp32, so the same code merely produced an ``<unk>`` token there.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.generation import _sample_from_cdf, _sample_token  # noqa: E402

VOCAB = 256


def test_inverse_cdf_sampling_clamps_a_truncated_cdf():
    """Device-independent guard for the fix: a cumulative sum that ends below
    1.0 must clamp to the last token, never return one index past the end."""
    cdf = torch.tensor([0.3, 0.6, 0.99609375])  # exactly what bf16 rounding yields
    assert _sample_from_cdf(cdf, torch.tensor([0.1])) == 0
    assert _sample_from_cdf(cdf, torch.tensor([0.5])) == 1
    assert _sample_from_cdf(cdf, torch.tensor([0.999])) == 2   # never 3
    assert _sample_from_cdf(cdf, torch.tensor([1.0])) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the bf16 truncation is CUDA-specific")
def test_bf16_sampling_on_cuda_never_returns_an_out_of_vocab_id():
    """Reproduces the benchmark crash: bf16 rows on CUDA, where the unclamped
    computation really does return an id past the end of the vocabulary."""
    torch.manual_seed(0)
    rows = [torch.randn(1, VOCAB, device="cuda").to(torch.bfloat16) for _ in range(20)]

    rng = torch.Generator(device="cpu")
    legacy_out_of_range = 0
    truncated_rows = 0
    for row in rows:
        cdf = torch.softmax(row, dim=-1).cumsum(-1)[0]      # the bf16 cdf v0.1 used
        truncated_rows += int(cdf[-1].item() < 1.0)
        for seed in range(50):
            rng.manual_seed(seed)
            u = torch.rand(1, generator=rng)
            legacy_out_of_range += int(int(torch.searchsorted(cdf, u.to(cdf.device)).item()) >= VOCAB)
    # the fixture must genuinely exercise the hazard, or the test below proves nothing
    assert truncated_rows > 0, "CUDA bf16 cumsum no longer truncates on this fixture"
    assert legacy_out_of_range > 0, "fixture no longer reproduces out-of-range sampling"

    for row in rows:
        for seed in range(50):
            rng.manual_seed(seed)
            nid = _sample_token(row, temperature=1.0, top_k=50, rng=rng)
            assert 0 <= nid < VOCAB, f"sampled id {nid} outside the vocabulary"


def test_sampling_stays_in_range_for_greedy_and_low_precision_rows():
    """Same invariant for every decoding configuration (runs on any device)."""
    torch.manual_seed(0)
    rng = torch.Generator(device="cpu")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        row = torch.randn(1, VOCAB).to(dtype)
        assert _sample_token(row, temperature=0.0, top_k=None, rng=rng) in range(VOCAB)
        for seed in range(50):
            rng.manual_seed(seed)
            assert _sample_token(row, temperature=0.7, top_k=10, rng=rng) in range(VOCAB)
            assert _sample_token(row, temperature=1.0, top_k=None, rng=rng) in range(VOCAB)
