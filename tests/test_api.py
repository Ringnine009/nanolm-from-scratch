"""API tests for the FastAPI + SSE chat server."""

import json

import pytest
from fastapi.testclient import TestClient

from nanollm.server.app import app, runtime


@pytest.fixture(scope="module")
def client(tiny_checkpoint, tiny_tokenizer):
    tok, tok_path = tiny_tokenizer
    runtime.load(str(tiny_checkpoint), str(tok_path), device="cpu")
    with TestClient(app) as c:
        yield c
    runtime.model = None
    runtime.tokenizer = None
    runtime.base_model = None
    runtime._comparison = None


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vocab_size"] == 400


def test_chat_returns_sse_stream(client):
    r = client.post("/api/chat", json={"prompt": "Is the death cap poisonous?"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    text = r.text
    assert "data:" in text
    tokens = [json.loads(line[6:])["token"] for line in text.splitlines() if line.startswith("data: ")]
    assert tokens, "no tokens streamed"
    assert tokens[-1] == "[DONE]"
    # the visible streamed content should be non-trivial
    body = "".join(t for t in tokens if t != "[DONE]")
    assert len(body) > 0


def test_chat_wraps_instructions_by_default(client):
    r = client.post("/api/chat", json={"prompt": "hello", "max_tokens": 8})
    assert r.status_code == 200
    tokens = [json.loads(line[6:])["token"] for line in r.text.splitlines() if line.startswith("data: ")]
    assert any(t != "[DONE]" for t in tokens)


def test_chat_validation(client):
    r = client.post("/api/chat", json={"prompt": "", "max_tokens": 99999})
    assert r.status_code == 422  # pydantic validation


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "NanoLM" in r.text
    # the v4 UI ships: tabs, generation-parameter sliders, comparison pane
    assert "Before/After fine-tuning" in r.text
    assert "Generation parameters" in r.text
    assert "sl-temperature" in r.text
    # v5: i18n (EN default text + zh dictionary + lang switch) and live stats
    assert "data-i18n" in r.text
    assert "中文" in r.text
    assert "stats_done" in r.text
    assert "lang-zh" in r.text
    # v6: multi-turn history + Training tab (SVG loss curve, hyperparams)
    assert "history" in r.text
    assert "tab-training" in r.text
    assert "tr-svg" in r.text


def test_comparison_returns_structured_json(client):
    # no --base-ckpt in tests -> fallback examples from docs/results.md
    r = client.get("/api/comparison")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"question", "before", "after", "source"}
    assert body["question"]
    assert len(body["before"]) > 20
    assert len(body["after"]) > 20
    assert "checkpoints" in body["source"] or "results.md" in body["source"]
    # cached on repeat calls
    r2 = client.get("/api/comparison")
    assert r2.json() == body


def test_training_returns_structured_json(client):
    r = client.get("/api/training")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"loss", "hyperparams", "evaluation"}
    # loss arrays present (from the archived sample when no CSV is available)
    assert len(body["loss"]["steps"]) >= 8
    assert len(body["loss"]["train_loss"]) == len(body["loss"]["steps"])
    assert body["loss"]["source"] in ("out/pretrain/losses.csv", "archived sample")
    # hyperparams have the three sections and the real run values
    hp = body["hyperparams"]
    assert set(hp) == {"model", "training", "lora"}
    assert hp["model"]["layers"] > 0 and hp["model"]["vocab_size"] > 0
    assert hp["training"]["steps"] == 6000
    assert hp["lora"]["r"] == 8 and hp["lora"]["alpha"] == 16
    # evaluation numbers are the real held-out results
    ev = body["evaluation"]
    assert ev["n"] == 44
    assert abs(ev["hit_any"] - 0.386) < 0.01
    assert abs(ev["hit_all"] - 0.091) < 0.01
    assert abs(ev["categories"]["edibility"]["hit_any"] - 0.80) < 0.01


def test_chat_accepts_history_field(client):
    r = client.post("/api/chat", json={
        "prompt": "What about the destroying angel?",
        "history": [
            {"role": "user", "content": "Is the death cap poisonous?"},
            {"role": "assistant", "content": "Yes, it is deadly poisonous."},
        ],
        "max_tokens": 8,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
