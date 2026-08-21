"""Tests for the self-implemented LoRA adapter injection."""

import torch

from nanollm.lora import inject_lora, load_lora, merge_lora, save_lora
from nanollm.model import GPT, GPTConfig


def make_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=1024, block_size=64, n_layer=2, n_head=2, n_embd=64))


def test_inject_lora_freezes_base_and_enables_adapters():
    model = make_model()
    base_params = {n: p.clone() for n, p in model.named_parameters()}
    lora_modules = inject_lora(model, r=4, alpha=8)

    assert len(lora_modules) > 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad, name
    # base weights untouched by injection
    for n, p in model.named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, base_params[n]), n


def test_lora_forward_matches_manual_math():
    model = make_model()
    inject_lora(model, r=4, alpha=8)
    model.eval()
    x = torch.randint(0, 1024, (2, 20))

    with torch.no_grad():
        y_lora = model(x)[0]

    # manual: for every LoRA-wrapped linear, W' = W + (alpha/r) * B @ A
    model2 = make_model()
    with torch.no_grad():
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                target = model2
                for part in name.split(".")[:-1]:
                    target = getattr(target, part)
                linear = getattr(target, name.split(".")[-1])
                scale = mod.lora_alpha / mod.lora_r
                delta = mod.lora_B.weight.data @ mod.lora_A.weight.data * scale
                linear.weight.data.add_(delta)
        y_manual = model2(x)[0]
    assert torch.allclose(y_lora, y_manual, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    model1 = make_model()
    inject_lora(model1, r=4, alpha=16)
    model1.eval()
    path = tmp_path / "lora.pt"
    save_lora(model1, path, base_ckpt="dummy.ckpt", r=4, alpha=16)

    model2 = make_model()
    load_lora(model2, path)
    model2.eval()

    x = torch.randint(0, 1024, (2, 20))
    with torch.no_grad():
        y1 = model1(x)[0]
        y2 = model2(x)[0]
    assert torch.allclose(y1, y2, atol=1e-5)
    # loading restored trainability status
    lora_params = [p for n, p in model2.named_parameters() if "lora_" in n]
    assert lora_params and all(p.requires_grad for p in lora_params)
    base_params = [p for n, p in model2.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad for p in base_params)


def test_merge_lora_equals_lora_forward(tmp_path):
    model = make_model()
    inject_lora(model, r=8, alpha=16)
    model.eval()
    x = torch.randint(0, 1024, (2, 20))
    with torch.no_grad():
        y_lora = model(x)[0]

    merge_lora(model)
    model.eval()
    with torch.no_grad():
        y_merged = model(x)[0]
    assert torch.allclose(y_lora, y_merged, atol=1e-5)

    # after merging the model is a plain GPT again (no lora modules, all trainable)
    assert not any("lora_" in n for n, _ in model.named_parameters())
    assert all(p.requires_grad for p in model.parameters())
    # and it can be saved / reloaded as a normal state dict
    torch.save({"model": model.state_dict(), "config": model.config.__dict__}, tmp_path / "merged.pt")


def test_lora_saves_independent_small_file(tmp_path):
    model = make_model()
    inject_lora(model, r=4, alpha=8)
    path = tmp_path / "lora.pt"
    save_lora(model, path, base_ckpt="base.ckpt", r=4, alpha=8)
    payload = torch.load(path, weights_only=False)
    n_full = sum(p.numel() for p in model.parameters())
    n_lora = sum(v["A"].numel() + v["B"].numel() for v in payload["adapters"].values())
    assert n_lora < n_full // 20  # adapters are a tiny fraction of the model
    assert payload["config"]["r"] == 4
