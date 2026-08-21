"""End-to-end training smoke test: loss must decrease; checkpoint resume works."""

import torch

from nanollm.model import GPT, GPTConfig


def _read_loss(csv_path, field="train_loss"):
    rows = []
    with open(csv_path, newline="") as f:
        import csv
        for r in csv.DictReader(f):
            rows.append(float(r[field]))
    return rows


def test_training_loss_decreases(tiny_checkpoint):
    import nanollm.train as train_mod

    csv_path = tiny_checkpoint.parent / "losses.csv"
    losses = _read_loss(csv_path)
    assert len(losses) >= 5, losses
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    # and the final logged value is well below the initial (random-init ~ ln(V))
    assert losses[-1] < losses[0] - 0.1


def test_checkpoint_contains_model_and_config(tiny_checkpoint, tiny_tokenizer):
    tok, _ = tiny_tokenizer
    ckpt = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    assert "model" in ckpt and "optimizer" in ckpt and "config" in ckpt
    assert ckpt["config"]["vocab_size"] == 400
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    # forward works and is deterministic
    model.eval()
    x = torch.randint(0, 400, (1, 16))
    with torch.no_grad():
        y1 = model(x)[0]
        y2 = model(x)[0]
    assert torch.equal(y1, y2)


def test_resume_continues_training(tiny_checkpoint, tiny_bin_data, tmp_path):
    """Save at step N, resume, and verify the step counter advances."""
    from nanollm.train import main as train_main

    data_dir, tok_path = tiny_bin_data
    out2 = tmp_path / "out2"
    ckpt1 = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    step1 = ckpt1["step"]

    train_main([
        "--data-dir", str(data_dir),
        "--out-dir", str(out2),
        "--tokenizer", str(tok_path),
        "--device", "cpu",
        "--dtype", "fp32",
        "--vocab-size", "400",
        "--block-size", "32",
        "--n-layer", "2",
        "--n-head", "2",
        "--n-embd", "32",
        "--batch-size", "4",
        "--max-steps", "60",
        "--eval-interval", "20",
        "--eval-iters", "2",
        "--log-interval", "100",
        "--sample-interval", "100000",
        "--lr", "3e-3",
        "--warmup-steps", "5",
        "--init-from", str(tiny_checkpoint),
        "--max-minutes", "30",
    ])
    ckpt2 = torch.load(out2 / "latest.ckpt", map_location="cpu", weights_only=False)
    assert ckpt2["step"] == 60
    assert ckpt2["step"] > step1
