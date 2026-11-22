# Speculative Decoding Reference API

Side-by-side, deterministic comparison of:

* **speculative decoding** — a small draft model proposes a block of tokens,
  and one target-model forward pass verifies the whole block; and
* **token-by-token greedy decoding** with the target model alone.

Both paths use the target model's KV cache over the confirmed prefix and are
guaranteed to produce the same continuation. The API reports the exact number
of target forward passes each path needed.

## Quick start

```bash
docker compose up --build
# open http://localhost:8000/docs
```

or locally:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest -q
python examples/run_examples.py            # in-process report
python examples/run_examples.py --url http://127.0.0.1:8000   # via HTTP
```

## Inputs (at `/docs`, `POST /decode`)

| field | meaning | constraints |
|---|---|---|
| `prompt_token_ids` | prompt token ids | 1–256 ids, each in `[0, 63]` (token 1 = `<eos>`, 2 = `<bos>`) |
| `max_new_tokens` | continuation cap | 1–128 |
| `draft_block_size` | draft tokens proposed per round | 1–32 |

Response contains `greedy` and `speculative` objects with
`continuation_token_ids`, `full_sequence_token_ids`,
`target_forward_calls`, `target_positions_evaluated` and `stop_reason`, plus a
`comparison` block (`outputs_identical`, `target_forward_calls_saved`,
`speedup`). The speculative object also records every round: proposed draft
tokens, accepted count, corrected token, and emitted tokens.

Other endpoints: `GET /health`, `GET /models` (architecture, parameter count,
per-model weight digest).

## Models

Two deterministic single-layer pre-norm causal-attention transformers in
float64 NumPy, sharing one 64-token vocabulary:

* **target** — `d_model=32`, 4 heads, `d_ff=64`, fixed seed `20240517`.
* **draft** — same attention/embedding weights, FFN narrowed from 64 to 24
  hidden channels. The narrow FFN is distilled at import time from the target
  on a fixed synthetic corpus: channels are ranked by
  `||w_in[:, j]|| · ||w_out[j, :]||`, and the kept `w_out` rows are refitted by
  ridge least squares against the target's own FFN outputs. Tiny seeded
  jitter keeps the draft imperfect (≈60% per-position greedy agreement). It
  uses 62.5% fewer FFN MACs per token (1536 vs 4096).

No sampling, no dropout, no external model files: weights are derived from
fixed seeds, so both models and all outputs are bit-for-bit reproducible.

## Decoding algorithm

Both decoders prefill the prompt and then extend the target KV cache; a
confirmed position is never evaluated twice.

Speculative round:

1. The draft proposes up to `draft_block_size` tokens, extending its own KV
   cache one token at a time.
2. **One** target forward pass runs over the pending token (the last confirmed
   token, not yet in the target cache) plus the whole draft block. The cache
   yields one target prediction per draft position.
3. Walk the block: while the target's prediction equals the draft token,
   accept it. At the first mismatch, emit the target's own token and discard
   that position and the rest of the block.
4. If every draft token was accepted, the final row of the same forward pass
   supplies a free bonus token (unless the continuation cap is reached).
5. Stop if `<eos>` was committed or the cap was hit. Otherwise truncate both
   KV caches back to the corrected prefix and append only the newly committed
   token to the draft cache, then continue.

Because a draft token is committed only when it equals the target's argmax,
the emitted sequence is exactly the target's greedy sequence. The API
double-checks this per request and returns HTTP 500 if the two ever disagree.

## Sample runs (also in `examples/run_examples.py`)

**All drafts accepted — fewer target forwards.**

`{"prompt_token_ids": [2, 34, 24, 11], "max_new_tokens": 16, "draft_block_size": 4}`

```
outputs identical: True
target forwards: greedy=16 speculative=5 (saved 11, 3.2x)
draft tokens accepted: 13/13
continuation: [20, 25, 7, 44, 19, 27, 52, 44, 19, 25, 7, 44, 19, 27, 46, 19]
  round 0: draft=[20,25,7,44] accepted=4 corrected=None emitted=[20,25,7,44,19]
  round 1: draft=[27,52,44,19] accepted=4 corrected=None emitted=[27,52,44,19,25]
  round 2: draft=[7,44,19,27] accepted=4 corrected=None emitted=[7,44,19,27,46]
  round 3: draft=[19] accepted=1 corrected=None emitted=[19]
```

**Draft block rejected mid-way, generation continues.**

`{"prompt_token_ids": [2, 22, 30, 11], "max_new_tokens": 16, "draft_block_size": 4}`

```
outputs identical: True
target forwards: greedy=16 speculative=5 (saved 11)
draft tokens accepted: 12/16
continuation: [25, 20, 52, 25, 44, 56, 22, 20, 12, 2, 48, 33, 48, 33, 48, 19]
  round 0: draft=[25,20,52,44] accepted=3 corrected=25 emitted=[25,20,52,25]
  round 1: draft=[44,22,20,12] accepted=1 corrected=56 emitted=[44,56]
  round 2: draft=[22,20,12,2] accepted=4 corrected=None emitted=[22,20,12,2,48]
  round 3: draft=[33,48,33,48] accepted=4 corrected=None emitted=[33,48,33,48,19]
```

Round 0 guesses `44` where the target wants `25`; the guess is replaced, the
rest of that block is dropped, and round 1 starts from the corrected sequence.

**EOS stop** (extra example):

`{"prompt_token_ids": [2, 62, 32, 11], "max_new_tokens": 24, "draft_block_size": 4}`

```
greedy and speculative both stop at token 1 (<eos>) after 14 tokens
target forwards: greedy=14 speculative=9
```

## Tests

`pytest -q` runs 224 tests: greedy vs speculative equality across prompts,
block sizes and caps; speculative and greedy vs a brute-force no-cache
reference; KV-cache reuse vs full recomputation; cache truncation paths;
EOS handling; forward-call accounting; API validation and both sample inputs.

## Known limitations

* CPU NumPy only; a "forward call" is the counted unit, not wall-clock latency.
  The draft's extra work is not included in the speedup number.
* The models are randomly initialised and distilled only by the offline FFN
  fit above; they are a correctness harness, not a real language model.
* `max_pos=512`; requests exceeding it return HTTP 422.
* The app keeps no state between requests; there is no streaming endpoint.

## Layout

```
app/model.py      NumPy causal LM, KV cache, draft distillation
app/decoding.py   greedy + speculative decoders and traces
app/main.py       FastAPI app (/decode, /models, /health, /docs)
tests/            decoding + API tests
examples/         runnable sample report
```
