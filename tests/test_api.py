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
