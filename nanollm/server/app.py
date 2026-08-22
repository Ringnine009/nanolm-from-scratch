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
import json
import sys
import threading
from pathlib import Path
from typing import Optional

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


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User question (or full prompt)")
    max_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=1, le=500)
    repetition_penalty: float = Field(1.15, ge=1.0, le=3.0)
    no_repeat_ngram_size: int = Field(4, ge=0, le=16)
    wrap_instructions: bool = Field(True, description="wrap prompt in <|user|>/<|assistant|>")


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
    if req.wrap_instructions and "<|user|>" not in req.prompt:
        return f"<|user|>{req.prompt}<|assistant|>"
    return req.prompt


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not runtime.ready:
        return StreamingResponse(iter(["event: error\ndata: {\"error\": \"model not loaded\"}\n\n"]),
                                 media_type="text/event-stream")
    prompt = build_prompt(req)
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
