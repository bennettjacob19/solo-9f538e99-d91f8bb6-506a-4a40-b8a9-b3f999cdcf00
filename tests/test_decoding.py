import numpy as np
import pytest

from app.decoding import greedy_decode, speculative_decode
from app.model import DRAFT_MODEL, TARGET_MODEL, VOCAB_SIZE, CausalLM, ModelError

PROMPTS = [
    [2, 5, 7, 11],
    [2],
    [5, 5, 5, 5],
    [2, 19, 19, 19],
    [9, 31, 42, 7],
    [3, 4, 5, 6, 7, 8],
    [2, 3, 50, 11],
    [2, 3, 4, 11],
    [2, 34, 24, 11],
    [2, 22, 30, 11],
]


def test_determinism():
    a = greedy_decode(TARGET_MODEL, [2, 5, 7, 11], 12)
    b = greedy_decode(TARGET_MODEL, [2, 5, 7, 11], 12)
    assert a.continuation == b.continuation
    assert a.target_forward_calls == b.target_forward_calls


def test_kv_cache_matches_full_recompute():
    prompt = [2, 5, 7, 11, 19, 3]
    full, _ = TARGET_MODEL.forward(prompt)
    _, cache = TARGET_MODEL.forward(prompt[:3])
    tail, _ = TARGET_MODEL.forward(prompt[3:], cache)
    restarted, _ = TARGET_MODEL.forward(prompt[3:],
                                        TARGET_MODEL.forward(prompt[:3])[1])
    assert np.allclose(tail, full[3:], atol=1e-12)
    assert np.allclose(restarted, tail, atol=1e-12)


def test_cache_rejects_bad_shape():
    cache = (
        np.zeros((2, 99, 8)),
        np.zeros((2, 99, 8)),
    )
    with pytest.raises(ModelError):
        TARGET_MODEL.forward([5], cache)


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("max_new", [1, 2, 5, 16])
@pytest.mark.parametrize("block", [1, 2, 3, 4, 8])
def test_speculative_matches_greedy(prompt, max_new, block):
    greedy = greedy_decode(TARGET_MODEL, list(prompt), max_new)
    spec = speculative_decode(TARGET_MODEL, DRAFT_MODEL, list(prompt),
                              max_new, block)
    assert spec.continuation == greedy.continuation
    assert spec.full_sequence == greedy.full_sequence
    assert spec.stop_reason == greedy.stop_reason
    assert len(spec.continuation) <= max_new


def brute_force_greedy(prompt, max_new):
    """Reference decoder: no cache at all, re-run the prompt every step."""
    seq = list(prompt)
    out = []
    for _ in range(max_new):
        logits, _ = TARGET_MODEL.forward(seq)
        token = int(np.argmax(logits[-1]))
        out.append(token)
        seq.append(token)
        if token == 1:
            break
    return out


@pytest.mark.parametrize("prompt, max_new, block", [
    ([2, 34, 24, 11], 16, 4),
    ([2, 22, 30, 11], 16, 4),
    ([2, 62, 32, 11], 24, 4),
    ([7, 7, 7], 5, 1),
    ([63, 0, 1], 3, 2),
])
def test_both_decoders_match_bruteforce(prompt, max_new, block):
    reference = brute_force_greedy(prompt, max_new)
    greedy = greedy_decode(TARGET_MODEL, prompt, max_new)
    spec = speculative_decode(TARGET_MODEL, DRAFT_MODEL, prompt, max_new, block)
    assert greedy.continuation == reference
    assert spec.continuation == reference


def test_greedy_forward_counts():
    # One prefill plus one call per token after the first; the final token is
    # read off the last call's logits without an extra forward.
    result = greedy_decode(TARGET_MODEL, [2, 5, 7, 11], 10)
    assert result.target_forward_calls == 10
    assert result.target_positions_evaluated == 4 + 9


def test_speculative_uses_one_target_call_per_round():
    result = speculative_decode(TARGET_MODEL, DRAFT_MODEL, [2, 34, 24, 11],
                                16, 4)
    assert result.target_forward_calls == 1 + len(result.rounds)
    for r in result.rounds:
        assert r.target_forward_calls == 1


def test_all_accepted_reduces_target_forwards():
    greedy = greedy_decode(TARGET_MODEL, [2, 34, 24, 11], 16)
    spec = speculative_decode(TARGET_MODEL, DRAFT_MODEL, [2, 34, 24, 11], 16, 4)
    assert spec.continuation == greedy.continuation
    assert all(r.corrected_token is None for r in spec.rounds)
    assert len(spec.rounds) >= 2
    assert spec.target_forward_calls < greedy.target_forward_calls
    # A fully accepted block of 4 yields 5 tokens from one target call.
    assert spec.rounds[0].emitted_tokens == spec.rounds[0].draft_tokens + [
        spec.continuation[4]
    ]


def test_mid_block_rejection_continues():
    greedy = greedy_decode(TARGET_MODEL, [2, 22, 30, 11], 16)
    spec = speculative_decode(TARGET_MODEL, DRAFT_MODEL, [2, 22, 30, 11], 16, 4)
    assert spec.continuation == greedy.continuation
    rejected = [r for r in spec.rounds
                if r.corrected_token is not None and r.accepted_count > 0]
    assert rejected, "expected a partially accepted block"
    assert any(len(r.emitted_tokens) > 1 for r in rejected)
    # Later rounds still emit tokens after the correction.
    assert len(spec.rounds) > rejected[0].round_index + 1
    # The rejected tail never leaks into the output.
    assert spec.draft_tokens_accepted < spec.draft_tokens_proposed


def test_eos_stops_both_paths():
    prompt = [2, 62, 32, 11]
    greedy = greedy_decode(TARGET_MODEL, prompt, 24)
    spec = speculative_decode(TARGET_MODEL, DRAFT_MODEL, prompt, 24, 4)
    assert greedy.stop_reason == "eos"
    assert spec.stop_reason == "eos"
    assert spec.continuation == greedy.continuation
    assert spec.continuation.count(1) == 1
    assert spec.continuation[-1] == 1


def test_draft_is_cheaper():
    # Narrower FFN means fewer MACs per token, and fewer stored weights.
    assert DRAFT_MODEL.w_in.shape[1] < TARGET_MODEL.w_in.shape[1]
    assert DRAFT_MODEL.w_out.shape[0] < TARGET_MODEL.w_out.shape[0]
    assert DRAFT_MODEL.w_in.size + DRAFT_MODEL.w_out.size < (
        TARGET_MODEL.w_in.size + TARGET_MODEL.w_out.size
    )


def test_bad_token_ids():
    with pytest.raises(ModelError):
        TARGET_MODEL.forward([VOCAB_SIZE])


def test_max_pos_guard():
    small = CausalLM.distilled_draft("tiny", TARGET_MODEL, rank=8)
    assert small.max_pos == TARGET_MODEL.max_pos
