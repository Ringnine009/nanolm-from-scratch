"""FastAPI chat server with Server-Sent-Events streaming.

Run:
    python -m nanollm.server.app --model checkpoints/merged.pt --port 8000

Endpoints:
    GET  /                chat UI (static)
    GET  /api/health      model status
    POST /api/chat        streaming SSE chat (JSON body, see ChatRequest)
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

from nanollm.generation import END_MARKER, drop_leading_junk, generate_tokens
from nanollm.model import GPT, GPTConfig
from nanollm.tokenizer import BPETokenizer

STATIC_DIR = Path(__file__).parent / "static"


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
        self.tokenizer: Optional[BPETokenizer] = None
        self.device = "cpu"

    def load(self, ckpt_path: str, tokenizer_path: str, device: str):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        self.model = GPT(config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

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


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main(argv=None):
    p = argparse.ArgumentParser(description="NanoLM inference server")
    p.add_argument("--model", type=str, required=True, help="merged .pt or .ckpt")
    p.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    import uvicorn
    runtime.load(args.model, args.tokenizer, args.device)
    print(f"[server] model loaded: {args.model} | device={runtime.device} | "
          f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    sys.exit(main())
