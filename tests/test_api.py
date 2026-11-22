import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        c.post  # noqa: B018 - touch attribute for clarity
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    assert "/decode" in schema["paths"]
    body = schema["components"]["schemas"]["DecodeRequest"]
    assert set(body["required"]) == {"prompt_token_ids"}


def test_decode_example_all_accepted(client):
    r = client.post("/decode", json={
        "prompt_token_ids": [2, 34, 24, 11],
        "max_new_tokens": 16,
        "draft_block_size": 4,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["greedy"]["continuation_token_ids"] == (
        data["speculative"]["continuation_token_ids"]
    )
    assert data["comparison"]["outputs_identical"] is True
    assert data["comparison"]["target_forward_calls_saved"] > 0
    assert all(x["all_drafts_accepted"]
               for x in data["speculative"]["rounds"])


def test_decode_example_mid_block_rejection(client):
    r = client.post("/decode", json={
        "prompt_token_ids": [2, 22, 30, 11],
        "max_new_tokens": 16,
        "draft_block_size": 4,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["greedy"]["continuation_token_ids"] == (
        data["speculative"]["continuation_token_ids"]
    )
    assert any(not x["all_drafts_accepted"] and x["accepted_count"] > 0
               for x in data["speculative"]["rounds"])


def test_rejects_out_of_vocab(client):
    r = client.post("/decode", json={
        "prompt_token_ids": [2, 64],
        "max_new_tokens": 4,
        "draft_block_size": 2,
    })
    assert r.status_code == 422


def test_rejects_empty_prompt(client):
    r = client.post("/decode", json={
        "prompt_token_ids": [],
        "max_new_tokens": 4,
        "draft_block_size": 2,
    })
    assert r.status_code == 422


def test_rejects_bad_block_size(client):
    r = client.post("/decode", json={
        "prompt_token_ids": [2, 5],
        "max_new_tokens": 4,
        "draft_block_size": 0,
    })
    assert r.status_code == 422
