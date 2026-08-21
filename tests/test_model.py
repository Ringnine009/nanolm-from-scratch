"""Tests for the from-scratch GPT model."""

import torch

from nanollm.model import GPT, GPTConfig


def make_gpt(**overrides):
    cfg = GPTConfig(
        vocab_size=1024,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_sdpa=overrides.pop("use_sdpa", True),
        **overrides,
    )
    return GPT(cfg)


def test_forward_backward_shapes():
    model = make_gpt()
    x = torch.randint(0, 1024, (4, 32))
    y = torch.randint(0, 1024, (4, 32))
    logits, loss = model(x, y)
    assert logits.shape == (4, 32, 1024)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    # every parameter has a gradient
    for p in model.parameters():
        assert p.grad is not None, p.shape


def test_default_config_parameter_count_in_range():
    from nanollm.model import default_config
    model = GPT(default_config())
    n = sum(p.numel() for p in model.parameters())
    assert 5_000_000 <= n <= 30_000_000, n


def test_causal_mask_future_tokens_do_not_affect_past_logits():
    model = make_gpt()
    model.eval()
    x = torch.randint(0, 1024, (1, 40))
    with torch.no_grad():
        logits_a = model(x)[0]
    # perturb the future half of the sequence
    x_pert = x.clone()
    x_pert[:, 20:] = torch.randint(0, 1024, (1, 20))
    with torch.no_grad():
        logits_b = model(x_pert)[0]
    assert torch.allclose(logits_a[:, :20], logits_b[:, :20], atol=1e-5)
    assert not torch.allclose(logits_a[:, 20:], logits_b[:, 20:], atol=1e-5)


def test_embeddings_tied():
    model = make_gpt()
    assert model.lm_head.weight is model.wte.weight


def test_manual_attention_matches_sdpa():
    torch.manual_seed(0)
    m1 = make_gpt(use_sdpa=False)
    m2 = make_gpt(use_sdpa=True)
    m2.load_state_dict(m1.state_dict())
    m1.eval(); m2.eval()
    x = torch.randint(0, 1024, (2, 30))
    with torch.no_grad():
        l1 = m1(x)[0]
        l2 = m2(x)[0]
    assert torch.allclose(l1, l2, atol=1e-4)


def test_generate_shape_and_determinism():
    model = make_gpt()
    model.eval()
    x = torch.randint(0, 1024, (1, 10))
    g1 = model.generate(x, max_new_tokens=20, temperature=1.0, top_k=50, seed=42)
    g2 = model.generate(x, max_new_tokens=20, temperature=1.0, top_k=50, seed=42)
    assert g1.shape == (1, 30)
    assert torch.equal(g1, g2)  # same seed -> identical output


def test_generate_topk_one_is_argmax():
    model = make_gpt()
    model.eval()
    x = torch.randint(0, 1024, (1, 10))
    g = model.generate(x, max_new_tokens=15, temperature=1.0, top_k=1, seed=0)
    with torch.no_grad():
        logits = model(x)[0][0, -1]
    assert g[0, 10] == logits.argmax().item()


def test_residual_proj_init_scaled():
    model = make_gpt()
    for block in model.h:
        # GPT-2 style: residual projections are scaled by 1/sqrt(2*n_layer)
        std = block.attn.c_proj.weight.std().item()
        assert std < 0.02, std
