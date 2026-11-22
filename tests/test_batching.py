import numpy as np
import pytest

from app.batching import (
    AttentionBudgetError,
    BatchInputError,
    batched_greedy_decode,
)
from app.decoding import greedy_decode
from app.model import TARGET_MODEL, ModelError
from tests.test_decoding import PROMPTS


def _greedy(prompt, cap):
    return greedy_decode(TARGET_MODEL, list(prompt), cap)


@pytest.mark.parametrize("n", [1, 2, 3, 8])
def test_batch_matches_per_request_greedy(n):
    prompts = [list(PROMPTS[i % len(PROMPTS)]) for i in range(n)]
    caps = [1 + (i * 5) % 17 for i in range(n)]
    result = batched_greedy_decode(
        TARGET_MODEL, prompts, caps, attention_score_budget=10_000_000
    )
    assert [s.index for s in result.samples] == list(range(n))
    for state, prompt, cap in zip(result.samples, prompts, caps):
        reference = _greedy(prompt, cap)
        assert state.continuation == reference.continuation
        assert state.stop_reason == reference.stop_reason
        assert state.prompt == prompt
        assert state.max_new_tokens == cap


def test_mixed_length_batch_shares_forwards():
    prompts = [[2], [2, 5], [2, 5, 7, 11], [2, 22, 30, 11, 19, 3]]
    caps = [6, 5, 4, 3]
    result = batched_greedy_decode(
        TARGET_MODEL, prompts, caps, attention_score_budget=10_000_000
    )
    for state, prompt, cap in zip(result.samples, prompts, caps):
        reference = _greedy(prompt, cap)
        assert state.continuation == reference.continuation
        assert state.stop_reason == reference.stop_reason

    # The prefill forward holds all four different-length prompts at once, with
    # the query and key axes padded to the longest prompt.
    first = result.forwards[0]
    assert first.phase == "prefill"
    assert sorted(first.sample_indices) == [0, 1, 2, 3]
    assert first.query_positions == 6
    assert first.key_positions == 6


def test_decode_forwards_share_mixed_cache_lengths():
    prompts = [[2], [2, 5, 7, 11]]
    result = batched_greedy_decode(
        TARGET_MODEL, prompts, [4, 4], attention_score_budget=10_000_000
    )
    decode_forwards = [f for f in result.forwards if f.phase == "decode"]
    assert decode_forwards
    # Different cache lengths are padded into one decode forward.
    assert decode_forwards[0].sample_indices == [0, 1]
    assert decode_forwards[0].query_positions == 1
    assert decode_forwards[0].key_positions == 5
    for f in decode_forwards:
        assert f.attention_score_elements == (
            f.batch_size * f.n_heads * f.query_positions * f.key_positions
        )


def test_every_forward_stays_within_budget():
    prompts = [[2, 5, 7, 11], [2], [2, 22, 30, 11, 19], [7, 7, 7]]
    caps = [12, 8, 6, 5]
    for budget in [256, 512, 1024, 4096, 10_000_000]:
        result = batched_greedy_decode(TARGET_MODEL, prompts, caps, budget)
        assert all(f.attention_score_elements <= budget
                   for f in result.forwards)
        for state, prompt, cap in zip(result.samples, prompts, caps):
            reference = _greedy(prompt, cap)
            assert state.continuation == reference.continuation


def test_long_sample_does_not_block_short_ones():
    # A 24-token prompt forces 4*24*24 = 2304 score elements in prefill, so at
    # this budget it needs its own forward while the short prompts still share
    # one.
    prompts = [[2, 5], [2] + list(range(3, 25)), [7, 7, 7]]
    caps = [4, 4, 4]
    budget = 2304
    result = batched_greedy_decode(TARGET_MODEL, prompts, caps, budget)
    prefills = [f for f in result.forwards if f.phase == "prefill"]
    assert any(f.sample_indices == [1] for f in prefills)
    shared = [f for f in prefills if f.sample_indices == [0, 2]]
    assert shared and shared[0].batch_size == 2
    assert all(f.attention_score_elements <= budget for f in result.forwards)
    for state, prompt, cap in zip(result.samples, prompts, caps):
        reference = _greedy(prompt, cap)
        assert state.continuation == reference.continuation


def test_budget_too_small_for_a_single_sample_names_index():
    prompts = [[2], [2, 5, 7, 11], [2, 5]]
    with pytest.raises(AttentionBudgetError) as err:
        batched_greedy_decode(TARGET_MODEL, prompts, [2, 2, 2], 32)
    assert err.value.sample_index == 1
    assert err.value.phase == "prefill"
    assert err.value.required == 4 * 4 * 4


def test_budget_too_small_for_decode_names_index():
    # Prefill of a 4-token prompt costs 64, so a budget of 64 admits it; the
    # continuation eventually needs 4*(4+k) > 64 elements per decode forward.
    prompts = [[2], [2, 5, 7, 11]]
    with pytest.raises(AttentionBudgetError) as err:
        batched_greedy_decode(TARGET_MODEL, prompts, [2, 20], 64)
    assert err.value.sample_index == 1
    assert err.value.phase == "decode"


def test_eos_sample_leaves_while_others_continue():
    prompts = [[2, 62, 32, 11], [2, 34, 24, 11]]
    caps = [24, 24]
    result = batched_greedy_decode(
        TARGET_MODEL, prompts, caps, attention_score_budget=10_000_000
    )
    early, late = result.samples
    assert early.continuation == _greedy(prompts[0], 24).continuation
    assert late.continuation == _greedy(prompts[1], 24).continuation
    assert early.stop_reason == "eos"
    assert late.stop_reason == "max_new_tokens"

    # The sample that hit EOS stops appearing in later forwards while the
    # other one keeps decoding.
    first_forward_without_eos = next(
        i for i, f in enumerate(result.forwards) if 0 not in f.sample_indices
    )
    for f in result.forwards[first_forward_without_eos:]:
        assert f.sample_indices == [1]
    assert len(early.continuation) < len(late.continuation)


def test_per_sample_caps_are_independent():
    prompts = [[2, 34, 24, 11], [2, 22, 30, 11]]
    result = batched_greedy_decode(
        TARGET_MODEL, prompts, [3, 9], attention_score_budget=10_000_000
    )
    assert len(result.samples[0].continuation) == 3
    assert len(result.samples[1].continuation) == 9
    assert result.samples[0].stop_reason == "max_new_tokens"
    assert result.samples[1].stop_reason == "max_new_tokens"


def test_kv_cache_is_reused_per_sample():
    prompt = [2, 5, 7, 11, 19, 3]
    result = batched_greedy_decode(
        TARGET_MODEL, [prompt], [5], attention_score_budget=10_000_000
    )
    state = result.samples[0]
    # Prefill evaluates the prompt once; each further step evaluates one token.
    assert state.target_positions_evaluated == len(prompt) + 4
    assert state.target_forward_calls == 5
    # The reused cache covers the prompt plus every emitted token but the
    # final one: the last token is read off the previous forward's logits and
    # is never fed back (matching greedy_decode).
    k, v = state.cache
    assert k.shape[0] == len(prompt) + len(state.continuation) - 1
    assert v.shape[0] == k.shape[0]
    # And it agrees with a full recomputation of that prefix.
    evaluated = prompt + state.continuation[:-1]
    _ref_logits, ref_cache = TARGET_MODEL.forward(evaluated)
    assert np.allclose(k, ref_cache[0], atol=1e-12)
    assert np.allclose(v, ref_cache[1], atol=1e-12)


def test_cache_continuation_matches_flat_recompute():
    prompt = [2, 5, 7, 11, 19, 3]
    result = batched_greedy_decode(
        TARGET_MODEL, [prompt], [5], attention_score_budget=10_000_000
    )
    state = result.samples[0]
    full = prompt + state.continuation
    flat_logits, _ = TARGET_MODEL.forward(full)
    # Feeding the prompt through one decode-free path, then every generated
    # token one at a time from the cache, reproduces the flat logits.
    _, cache = TARGET_MODEL.forward(prompt)
    for offset, token in enumerate(state.continuation[:-1]):
        step_logits, cache = TARGET_MODEL.forward([token], cache)
        assert np.allclose(step_logits[-1], flat_logits[len(prompt) + offset],
                           atol=1e-12)


def test_forward_batch_matches_single_forward():
    prompts = [[2, 5, 7], [2, 34, 24, 11, 19], [7], [2, 22, 30, 11]]
    logits, caches, info = TARGET_MODEL.forward_batch(prompts)
    for i, prompt in enumerate(prompts):
        ref_logits, ref_cache = TARGET_MODEL.forward(prompt)
        assert np.allclose(logits[i], ref_logits, atol=1e-12)
        assert np.allclose(caches[i][0], ref_cache[0], atol=1e-12)
        assert np.allclose(caches[i][1], ref_cache[1], atol=1e-12)
    assert info.score_elements == (
        info.batch_size * info.n_heads * info.query_positions
        * info.key_positions
    )


def test_forward_batch_unequal_caches_and_new_lengths():
    cache_a = TARGET_MODEL.forward([2, 5, 7, 11])[1]
    cache_b = TARGET_MODEL.forward([2])[1]
    logits, caches, _ = TARGET_MODEL.forward_batch(
        [[9, 3], [4, 5, 6]], [cache_a, cache_b]
    )
    for logit, cache, tok, ref_cache in zip(
        logits, caches, [[9, 3], [4, 5, 6]], [cache_a, cache_b]
    ):
        ref_logits, ref = TARGET_MODEL.forward(tok, ref_cache)
        assert np.allclose(logit, ref_logits, atol=1e-12)
        assert np.allclose(cache[0], ref[0], atol=1e-12)


def test_forward_batch_validation():
    with pytest.raises(ModelError):
        TARGET_MODEL.forward_batch([])
    with pytest.raises(ModelError):
        TARGET_MODEL.forward_batch([[2], []])
    with pytest.raises(ModelError):
        TARGET_MODEL.forward_batch([[2], [64]])
    with pytest.raises(ModelError):
        TARGET_MODEL.forward_batch([[2]], [None, None])


def test_max_pos_overflow_names_sample():
    long_prompt = [2] * TARGET_MODEL.max_pos
    with pytest.raises(BatchInputError) as err:
        batched_greedy_decode(
            TARGET_MODEL, [[2], long_prompt], [1, 2], 10**9
        )
    assert err.value.sample_index == 1
    assert "sample 1" in str(err.value)


def test_decode_time_overflow_names_sample():
    # A prompt that fits, but whose continuation would run past max_pos.
    long_prompt = [2] * (TARGET_MODEL.max_pos - 2)
    with pytest.raises(BatchInputError) as err:
        batched_greedy_decode(TARGET_MODEL, [long_prompt], [5], 10**9)
    assert err.value.sample_index == 0
