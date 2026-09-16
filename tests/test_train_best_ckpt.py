"""``best.ckpt`` must always be a validation-selected checkpoint
(audit finding D9: ``if best_val < ckpt["best_val"]`` is a dead comparison with
itself, so the only live branch was "best.ckpt does not exist yet" - which
published *unevaluated* weights under the name "best").
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nanollm.train import main as train_main  # noqa: E402


def _base_args(data_dir, tok_path, out_dir, max_steps, eval_interval):
    return [
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        "--tokenizer", str(tok_path),
        "--device", "cpu",
        "--dtype", "fp32",
        "--vocab-size", "400",
        "--block-size", "32",
        "--n-layer", "2",
        "--n-head", "2",
        "--n-embd", "32",
        "--batch-size", "4",
        "--max-steps", str(max_steps),
        "--eval-interval", str(eval_interval),
        "--eval-iters", "2",
        "--log-interval", "5",
        "--sample-interval", "100000",
        "--lr", "3e-3",
        "--warmup-steps", "5",
        "--max-minutes", "30",
    ]


def test_never_publishes_an_unevaluated_best_ckpt(tiny_bin_data, tmp_path):
    """A run that takes no step (and therefore never evaluates) must not leave
    an untrained model behind under the name ``best.ckpt``."""
    data_dir, tok_path = tiny_bin_data
    out = tmp_path / "out"
    train_main(_base_args(data_dir, tok_path, out, max_steps=0, eval_interval=10))

    assert (out / "latest.ckpt").exists(), "final state must still be checkpointed"
    best = out / "best.ckpt"
    if best.exists():
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        assert ckpt["step"] > 0, "best.ckpt holds unevaluated step-0 weights"
        assert math.isfinite(ckpt["best_val"]), "best.ckpt has no validation signal"


def test_final_eval_runs_when_no_in_loop_eval_fires(tiny_checkpoint, tiny_bin_data, tmp_path):
    """Resuming at a step that is not a multiple of eval_interval used to run to
    the end without a single evaluation; the trainer must evaluate once at the
    end instead of publishing unvalidated weights."""
    data_dir, tok_path = tiny_bin_data
    out = tmp_path / "out"
    args = _base_args(data_dir, tok_path, out, max_steps=41, eval_interval=1000)
    args += ["--init-from", str(tiny_checkpoint)]
    train_main(args)

    log = (out / "train.log").read_text(encoding="utf-8")
    assert "final eval" in log, "no evaluation happened at all in this run"
    ckpt = torch.load(out / "best.ckpt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 41
    assert math.isfinite(ckpt["best_val"])
