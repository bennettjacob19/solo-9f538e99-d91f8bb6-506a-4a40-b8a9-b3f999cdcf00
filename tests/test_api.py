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


BATCH = {
    "prompts": [
        {"prompt_token_ids": [2, 34, 24, 11], "max_new_tokens": 16},
        {"prompt_token_ids": [2], "max_new_tokens": 5},
        {"prompt_token_ids": [2, 62, 32, 11], "max_new_tokens": 24},
        {"prompt_token_ids": [9, 31, 42, 7], "max_new_tokens": 7},
    ],
    "attention_score_budget": 1_000_000,
}


def test_batch_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    assert "/decode/batch" in schema["paths"]
    body = schema["components"]["schemas"]["BatchGreedyRequest"]
    assert set(body["required"]) == {"prompts", "attention_score_budget"}


def test_batch_matches_decode_greedy(client):
    r = client.post("/decode/batch", json=BATCH)
    assert r.status_code == 200
    data = r.json()
    assert [s["sample_index"] for s in data["samples"]] == [0, 1, 2, 3]
    for sample in data["samples"]:
        single = client.post("/decode", json={
            "prompt_token_ids": sample["prompt_token_ids"],
            "max_new_tokens": sample["max_new_tokens"],
            "draft_block_size": 2,
        }).json()["greedy"]
        assert sample["continuation_token_ids"] == (
            single["continuation_token_ids"]
        )
        assert sample["full_sequence_token_ids"] == (
            single["full_sequence_token_ids"]
        )
        assert sample["stop_reason"] == single["stop_reason"]


def test_batch_eos_sample_leaves_batch(client):
    data = client.post("/decode/batch", json=BATCH).json()
    samples = data["samples"]
    assert samples[2]["stop_reason"] == "eos"
    assert samples[2]["continuation_token_ids"][-1] == 1
    assert samples[0]["stop_reason"] == "max_new_tokens"
    assert len(samples[0]["continuation_token_ids"]) == 16
    # After the EOS sample stops, forwards no longer contain it.
    stopped_at = len(samples[2]["continuation_token_ids"])
    assert any(2 not in f["sample_indices"] for f in data["forwards"])
    assert stopped_at < 24


def test_batch_reports_forward_ledger(client):
    data = client.post("/decode/batch", json=BATCH).json()
    assert data["total_forwards"] == len(data["forwards"])
    assert data["total_attention_score_elements"] == sum(
        f["attention_score_elements"] for f in data["forwards"]
    )
    for f in data["forwards"]:
        assert f["attention_score_elements"] == (
            f["batch_size"] * f["n_heads"] * f["query_positions"]
            * f["key_positions"]
        )
        assert f["attention_score_elements"] <= 1_000_000
    assert [f["forward_index"] for f in data["forwards"]] == list(
        range(len(data["forwards"]))
    )
    # The prefill is shared by all four prompts despite their different sizes.
    prefill = data["forwards"][0]
    assert prefill["phase"] == "prefill"
    assert sorted(prefill["sample_indices"]) == [0, 1, 2, 3]


def test_batch_long_sample_runs_alone(client):
    r = client.post("/decode/batch", json={
        "prompts": [
            {"prompt_token_ids": [2], "max_new_tokens": 4},
            {"prompt_token_ids": [2] + list(range(3, 25)), "max_new_tokens": 4},
        ],
        "attention_score_budget": 2304,
    })
    assert r.status_code == 200
    data = r.json()
    prefills = [f for f in data["forwards"] if f["phase"] == "prefill"]
    assert any(f["sample_indices"] == [1] for f in prefills)
    assert any(f["sample_indices"] == [0] for f in prefills)
    for sample in data["samples"]:
        single = client.post("/decode", json={
            "prompt_token_ids": sample["prompt_token_ids"],
            "max_new_tokens": sample["max_new_tokens"],
        }).json()["greedy"]
        assert sample["continuation_token_ids"] == (
            single["continuation_token_ids"]
        )


def test_batch_rejects_budget_too_small_with_sample_index(client):
    r = client.post("/decode/batch", json={
        "prompts": [
            {"prompt_token_ids": [2], "max_new_tokens": 2},
            {"prompt_token_ids": [2, 5, 7, 11], "max_new_tokens": 2},
        ],
        "attention_score_budget": 32,
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "attention_score_budget_exceeded"
    assert detail["sample_index"] == 1
    assert detail["phase"] == "prefill"
    assert detail["required_attention_score_elements"] == 64


def test_batch_rejects_bad_size(client):
    r = client.post("/decode/batch", json={
        "prompts": [{"prompt_token_ids": [2], "max_new_tokens": 2}] * 9,
        "attention_score_budget": 1000,
    })
    assert r.status_code == 422
    r = client.post("/decode/batch", json={
        "prompts": [], "attention_score_budget": 1000,
    })
    assert r.status_code == 422


def test_batch_rejects_bad_tokens_and_caps(client):
    r = client.post("/decode/batch", json={
        "prompts": [{"prompt_token_ids": [2, 64], "max_new_tokens": 2}],
        "attention_score_budget": 1000,
    })
    assert r.status_code == 422
    r = client.post("/decode/batch", json={
        "prompts": [{"prompt_token_ids": [2], "max_new_tokens": 0}],
        "attention_score_budget": 1000,
    })
    assert r.status_code == 422
    r = client.post("/decode/batch", json={
        "prompts": [{"prompt_token_ids": [2], "max_new_tokens": 2}],
        "attention_score_budget": 0,
    })
    assert r.status_code == 422
