"""FastAPI chat server with Server-Sent-Events streaming.

Run:
    python -m nanollm.server.app --model checkpoints/merged.pt --port 8000
    # with the before/after comparison backed by real checkpoints:
    python -m nanollm.server.app --model checkpoints/merged.pt \
        --base-ckpt out/pretrain/best.ckpt --port 8000

Endpoints:
    GET  /                chat UI (static)
    GET  /api/health      model status
    POST /api/chat        streaming SSE chat (JSON body, see ChatRequest)
    GET  /api/comparison  before/after fine-tuning answers (same question)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import threading
from pathlib import Path
from typing import Literal, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanollm.generation import END_MARKER, drop_leading_junk, generate_answer, generate_tokens
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer

STATIC_DIR = Path(__file__).parent / "static"

COMPARISON_QUESTION = "Is the death cap mushroom poisonous?"

# Fallback examples: real outputs sampled from the actual checkpoints during
# the training run (documented in docs/results.md). Used when the server runs
# without --base-ckpt (or the base checkpoint file is unavailable).
COMPARISON_FALLBACK = {
    "before": (
        "before the death when the only a mushroom market. It is usually does not made by "
        "a group of a poisonous or a full color. It can be too small, but this is too much "
        "more than any way when it is no symptoms of a doctor or too much when it is too much "
        "to learn, and help with a person feels."
    ),
    "after": (
        "the death cap is deadly poisonous and should never be eaten. Amatoxins, chiefly "
        "alpha-amanitin, which block rna polymerase ii and destroy liver cells. Delayed "
        "poisoning: 6-24 hours after eating, violent cramps, vomiting and watery diarrhea "
        "begin, followed by a deceptive improvement and then liver and kidney failure. "
        "Contact a poison control center immediately if it is eaten."
    ),
}


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User question (or full prompt)")
    max_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=1, le=500)
    repetition_penalty: float = Field(1.15, ge=1.0, le=3.0)
    no_repeat_ngram_size: int = Field(4, ge=0, le=16)
    wrap_instructions: bool = Field(True, description="wrap prompt in <|user|>/<|assistant|>")
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="recent conversation history (oldest first); backward compatible",
    )


class ModelRuntime:
    """Holds the loaded model + tokenizer; swapped in at startup."""

    def __init__(self):
        self.model: Optional[GPT] = None
        self.base_model: Optional[GPT] = None  # base (pre-LoRA) model for comparison
        self.tokenizer: Optional[BPETokenizer] = None
        self.device = "cpu"
        self._comparison: Optional[dict] = None

    def _load_one(self, ckpt_path: str) -> GPT:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    def load(self, ckpt_path: str, tokenizer_path: str, device: str, base_ckpt: Optional[str] = None):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.model = self._load_one(ckpt_path)
        self.base_model = None
        if base_ckpt and Path(base_ckpt).exists():
            self.base_model = self._load_one(base_ckpt)
        self._comparison = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None


runtime = ModelRuntime()
app = FastAPI(title="NanoLM", version="0.1.0")


@app.get("/api/health")
def health():
    if not runtime.ready:
        return {"status": "loading"}
    return {
        "status": "ok",
        "device": runtime.device,
        "params": sum(p.numel() for p in runtime.model.parameters()),
        "vocab_size": runtime.model.config.vocab_size,
    }


def build_prompt(req: ChatRequest) -> str:
    """Build the model prompt from history + current question.

    Instruction format: ``<|user|>…<|assistant|>…`` turns are concatenated in
    order, ending with the current question.  In raw mode (``wrap_instructions
    = False``) the turns are joined with newlines instead.
    """
    if not req.history:
        if req.wrap_instructions and "<|user|>" not in req.prompt:
            return f"<|user|>{req.prompt}<|assistant|>"
        return req.prompt
    parts: list[str] = []
    for h in req.history:
        if req.wrap_instructions:
            tag = "<|user|>" if h.role == "user" else "<|assistant|>"
            parts.append(tag + h.content)
        else:
            parts.append(h.content)
    if req.wrap_instructions:
        parts.append(f"<|user|>{req.prompt}<|assistant|>")
    else:
        parts.append(req.prompt)
    return "".join(parts)


def build_chat_prompt(req: ChatRequest, tokenizer, block_size: int, reserve_tokens: int = 16) -> str:
    """Like ``build_prompt`` but drops the OLDEST history turns until the
    encoded prompt fits ``block_size - reserve_tokens``.  The current question
    is never dropped."""
    budget = block_size - reserve_tokens

    def fits(history: list[HistoryMessage]) -> bool:
        req.history = history
        return len(tokenizer.encode(build_prompt(req))) <= budget

    history = list(req.history)
    while len(history) > 1 and not fits(history):
        history = history[1:]  # drop the oldest turn
    req.history = history
    return build_prompt(req)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not runtime.ready:
        return StreamingResponse(iter(["event: error\ndata: {\"error\": \"model not loaded\"}\n\n"]),
                                 media_type="text/event-stream")
    prompt = build_chat_prompt(req, runtime.tokenizer, runtime.model.config.block_size)
    stop_ids = {runtime.tokenizer.special_to_id.get(END_MARKER, -1)}
    stop_ids.discard(-1)

    # True token-level streaming: the model runs in a worker thread and each
    # generated token is delivered to the SSE stream as soon as it is produced
    # (via call_soon_threadsafe so the event loop is never blocked).
    queue: asyncio.Queue = asyncio.Queue()

    def worker(loop: asyncio.AbstractEventLoop):
        try:
            stream = drop_leading_junk(generate_tokens(
                runtime.model, runtime.tokenizer, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature, top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                stop_ids=stop_ids,
            ))
            for tok in stream:
                loop.call_soon_threadsafe(queue.put_nowait, ("token", tok))
        except Exception as exc:  # surface generation errors to the client
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    loop = asyncio.get_running_loop()
    thread = threading.Thread(target=worker, args=(loop,), daemon=True)
    thread.start()

    async def stream():
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                yield f"data: {json.dumps({'error': payload}, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps({'token': payload}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------- #
# training info endpoint
# --------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]  # repository root
LOSS_CSV_PATH = ROOT_DIR / "out" / "pretrain" / "losses.csv"
EVAL_RESULTS_PATH = ROOT_DIR / "out" / "eval_results.json"

# Real run values (see docs/results.md); the loss fallback is the actual
# losses.csv sampled at the evaluation steps of the real training run.
TRAINING_HYPERPARAMS = {
    "training": {
        "steps": 6000, "learning_rate": 6e-4, "min_learning_rate": 6e-5,
        "warmup_steps": 300, "batch_size": 32, "block_size": 256,
        "grad_clip": 1.0, "weight_decay": 0.1, "dtype": "bf16",
        "corpus_mb": 1.5, "tokens": 373167, "wall_minutes": 28.1, "best_val": 5.02,
    },
    "lora": {
        "r": 8, "alpha": 16, "dropout": 0.05, "trainable_pct": 1.60,
        "train_pairs": 442, "val_pairs": 49, "steps": 1680, "val_loss": 0.84,
    },
}

EVALUATION_SUMMARY = {
    "n": 44, "hit_any": 0.386, "hit_all": 0.091,
    "categories": {
        "edibility": {"n": 10, "hit_any": 0.80},
        "identification": {"n": 8, "hit_any": 0.125},
        "habitat": {"n": 8, "hit_any": 0.25},
        "symptoms": {"n": 8, "hit_any": 0.375},
        "first_aid": {"n": 10, "hit_any": 0.30},
    },
    "source": "docs/results.md (real held-out run, regenerable via scripts/evaluate.py)",
}

ARCHIVED_LOSS = {
    "steps": [0, 300, 600, 900, 1200, 1500, 3000, 5700],
    "train_loss": [9.49, 4.87, 3.64, 1.98, 1.41, 0.73, 0.25, 0.12],
    "val_loss": [9.50, 5.86, 5.39, 5.14, 5.02, 5.51, 6.62, 7.16],
}


def load_loss_curve() -> dict:
    """Loss data from out/pretrain/losses.csv when available, else archived."""
    if LOSS_CSV_PATH.exists():
        steps, train_loss, val_loss = [], [], []
        with open(LOSS_CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                train_loss.append(float(row["train_loss"]))
                v = row.get("val_loss", "")
                val_loss.append(float(v) if v not in ("", "nan") else None)
        return {"steps": steps, "train_loss": train_loss, "val_loss": val_loss,
                "source": "out/pretrain/losses.csv"}
    return {**ARCHIVED_LOSS, "source": "archived sample"}


def model_hyperparams() -> dict:
    cfg = runtime.model.config if runtime.model is not None else None
    if cfg is None:
        return {"layers": None, "heads": None, "embed_dim": None, "block_size": None,
                "vocab_size": None, "params": None}
    return {
        "layers": cfg.n_layer, "heads": cfg.n_head, "embed_dim": cfg.n_embd,
        "block_size": cfg.block_size, "vocab_size": cfg.vocab_size,
        "params": sum(p.numel() for p in runtime.model.parameters()),
    }


@app.get("/api/training")
def training():
    """Training info for the 'Training' tab: loss curve, hyperparameters,
    and the held-out evaluation summary."""
    loss = load_loss_curve()
    ev = dict(EVALUATION_SUMMARY)
    if EVAL_RESULTS_PATH.exists():  # prefer live evaluation results
        try:
            live = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
            ev["hit_any"] = live["hit_any"]
            ev["hit_all"] = live["hit_all"]
            ev["categories"] = live["categories"]
            ev["source"] = "out/eval_results.json (regenerated by scripts/evaluate.py)"
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return {
        "loss": loss,
        "hyperparams": {
            "model": model_hyperparams(),
            "training": TRAINING_HYPERPARAMS["training"],
            "lora": TRAINING_HYPERPARAMS["lora"],
        },
        "evaluation": ev,
    }


@app.get("/api/comparison")
def comparison():
    """Before/after fine-tuning answers for the same question.

    With ``--base-ckpt`` provided, both answers are regenerated live from the
    real checkpoints (seeded, deterministic).  Otherwise the fallback examples
    from ``docs/results.md`` (real sampled outputs from the training run) are
    returned.  The result is computed once and cached.
    """
    if not runtime.ready:
        return {"error": "model not loaded"}
    if runtime._comparison is not None:
        return runtime._comparison

    if runtime.base_model is not None:
        before = generate_answer(runtime.base_model, runtime.tokenizer, COMPARISON_QUESTION, seed=3)
        after = generate_answer(runtime.model, runtime.tokenizer, COMPARISON_QUESTION, seed=3)
        source = ("Answers regenerated live from the real checkpoints "
                  "(before = base pretrained model, after = LoRA-merged model; "
                  "temperature 0.7, top-k 40, seed 3).")
    else:
        before = COMPARISON_FALLBACK["before"]
        after = COMPARISON_FALLBACK["after"]
        source = ("Example outputs sampled from the real checkpoints during the training "
                  "run (see docs/results.md). Start the server with --base-ckpt to "
                  "regenerate them live.")

    runtime._comparison = {"question": COMPARISON_QUESTION, "before": before, "after": after, "source": source}
    return runtime._comparison


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main(argv=None):
    p = argparse.ArgumentParser(description="NanoLM inference server")
    p.add_argument("--model", type=str, required=True, help="merged .pt or .ckpt")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--base-ckpt", type=str, default=None,
                   help="base (pre-LoRA) checkpoint for the /api/comparison endpoint")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    import uvicorn
    runtime.load(args.model, args.tokenizer, args.device, args.base_ckpt)
    print(f"[server] model loaded: {args.model} | device={runtime.device} | "
          f"base_ckpt={'yes' if runtime.base_model is not None else 'no (fallback comparison)'} | "
          f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    sys.exit(main())
