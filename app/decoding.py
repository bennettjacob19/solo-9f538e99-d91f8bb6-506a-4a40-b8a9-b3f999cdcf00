"""Greedy and speculative decoding over the NumPy causal LMs.

Both decoders use the *target* model's KV cache for the confirmed prefix, so
no confirmed position is ever evaluated twice.  The only difference is how
many tokens one target forward pass yields:

* greedy:         1 forward -> 1 token
* speculative:    1 forward -> up to ``draft_block_size + 1`` tokens

Speculative decoding stays exact because a draft token is only committed when
it equals the target's own argmax, and the first disagreement is replaced by
the target's token.  The rejected tail of the block is discarded, the target
cache is truncated back to the confirmed prefix, and both models continue from
the corrected sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import EOS_ID, CausalLM


def _argmax(logits: np.ndarray) -> int:
    # np.argmax breaks ties by lowest index, which keeps this deterministic.
    return int(np.argmax(logits))


def _truncate(cache: tuple[np.ndarray, np.ndarray],
              length: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the first ``length`` cached positions."""
    k, v = cache
    if length < 0 or length > k.shape[0]:
        raise ValueError("cache truncation out of range")
    return k[:length], v[:length]


@dataclass
class RoundTrace:
    round_index: int
    draft_tokens: list[int]
    accepted_count: int
    corrected_token: int | None
    emitted_tokens: list[int]
    target_forward_calls: int = 1


@dataclass
class DecodeResult:
    continuation: list[int]
    full_sequence: list[int]
    target_forward_calls: int
    target_positions_evaluated: int
    stop_reason: str
    rounds: list[RoundTrace] = field(default_factory=list)
    draft_forward_calls: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0


def greedy_decode(
    model: CausalLM, prompt: list[int], max_new_tokens: int
) -> DecodeResult:
    """Target-only, token-by-token greedy decoding with a KV cache."""
    logits, cache = model.forward(prompt)
    calls = 1
    positions = len(prompt)
    next_logits = logits[-1]

    continuation: list[int] = []
    stop_reason = "max_new_tokens"
    while len(continuation) < max_new_tokens:
        token = _argmax(next_logits)
        continuation.append(token)
        if token == EOS_ID:
            stop_reason = "eos"
            break
        if len(continuation) == max_new_tokens:
            break
        # Only the new token is evaluated; the prompt and all earlier tokens
        # come from the reused KV cache.
        logits, cache = model.forward([token], cache)
        calls += 1
        positions += 1
        next_logits = logits[-1]

    return DecodeResult(
        continuation=continuation,
        full_sequence=prompt + continuation,
        target_forward_calls=calls,
        target_positions_evaluated=positions,
        stop_reason=stop_reason,
    )


def speculative_decode(
    target: CausalLM,
    draft: CausalLM,
    prompt: list[int],
    max_new_tokens: int,
    draft_block_size: int,
) -> DecodeResult:
    """Block-draft speculative decoding with greedy acceptance.

    Invariant: ``pending`` (if set) is the last token of the confirmed
    sequence, and the target cache covers exactly the confirmed tokens before
    it.  The next verification forward therefore starts with ``pending`` and
    evaluates the new draft block in the same pass.
    """
    if draft_block_size < 1:
        raise ValueError("draft_block_size must be >= 1")

    # Initial prefill.  Its last row verifies the first draft token of round 1
    # without any extra target call.
    prompt_logits, target_cache = target.forward(prompt)
    target_calls = 1
    target_positions = len(prompt)
    prefix_logits = prompt_logits[-1]
    pending: int | None = None

    draft_logits, draft_cache = draft.forward(prompt)
    draft_calls = 1
    draft_next_logits = draft_logits[-1]

    continuation: list[int] = []
    stop_reason = "max_new_tokens"
    rounds: list[RoundTrace] = []
    proposed = 0
    accepted_total = 0

    while len(continuation) < max_new_tokens:
        round_index = len(rounds)
        generated_before = len(continuation)
        remaining = max_new_tokens - generated_before
        block = min(draft_block_size, remaining)

        # --- draft: propose up to `block` tokens with the small model, reusing
        # --- its own KV cache for the confirmed prefix.
        draft_tokens: list[int] = []
        block_cache = draft_cache
        block_logits = draft_next_logits
        for _ in range(block):
            token = _argmax(block_logits)
            draft_tokens.append(token)
            if token == EOS_ID:
                break
            out_logits, block_cache = draft.forward([token], block_cache)
            draft_calls += 1
            block_logits = out_logits[-1]
        proposed += len(draft_tokens)

        # --- verify: one target forward over the pending token (if any) plus
        # --- the whole draft block.
        inputs = ([pending] if pending is not None else []) + draft_tokens
        verify_logits, verify_cache = target.forward(inputs, target_cache)
        target_calls += 1
        target_positions += len(inputs)
        offset = 1 if pending is not None else 0

        def prediction(index: int, base_logits: np.ndarray,
                       block_logits: np.ndarray, shift: int) -> int:
            """Target's greedy token after confirmed + draft[:index]."""
            row = index + shift - 1
            if row < 0:
                return _argmax(base_logits)
            return _argmax(block_logits[row])

        accepted = 0
        while (accepted < len(draft_tokens)
               and prediction(accepted, prefix_logits, verify_logits, offset)
               == draft_tokens[accepted]):
            accepted += 1
        accepted_total += accepted
        # The last row of the verification pass predicts the token after the
        # whole block; when every draft matched it is the free "bonus" token.
        bonus_token = _argmax(verify_logits[-1])

        if accepted == len(draft_tokens):
            corrected = None
            emitted = list(draft_tokens)
            if len(continuation) + len(emitted) < max_new_tokens:
                emitted.append(bonus_token)
        else:
            corrected = prediction(accepted, prefix_logits, verify_logits,
                                   offset)
            emitted = draft_tokens[:accepted] + [corrected]

        # An EOS anywhere in the committed tokens ends generation exactly
        # where greedy decoding would have ended.
        eos_at = next((i for i, t in enumerate(emitted) if t == EOS_ID), None)
        if eos_at is not None:
            emitted = emitted[: eos_at + 1]
            stop_reason = "eos"
            stopping = True
        else:
            stopping = len(continuation) + len(emitted) >= max_new_tokens
            if stopping:
                stop_reason = "max_new_tokens"

        rounds.append(
            RoundTrace(
                round_index=round_index,
                draft_tokens=draft_tokens,
                accepted_count=accepted,
                corrected_token=corrected,
                emitted_tokens=emitted,
            )
        )
        continuation.extend(emitted)
        if stopping:
            break

        # --- resync both models from the corrected sequence.  Rejected draft
        # --- tokens are dropped from both caches; only the single newly
        # --- committed token (the bonus or the correction) is appended.
        pending = emitted[-1]
        # Absolute prefix length: the prompt plus every emitted token except
        # the one now held in `pending`.
        keep = len(prompt) + len(continuation) - 1
        target_cache = _truncate(verify_cache, keep)
        draft_cache = _truncate(block_cache, keep)
        out_logits, draft_cache = draft.forward([pending], draft_cache)
        draft_calls += 1
        draft_next_logits = out_logits[-1]

    return DecodeResult(
        continuation=continuation,
        full_sequence=prompt + continuation,
        target_forward_calls=target_calls,
        target_positions_evaluated=target_positions,
        stop_reason=stop_reason,
        rounds=rounds,
        draft_forward_calls=draft_calls,
        draft_tokens_proposed=proposed,
        draft_tokens_accepted=accepted_total,
    )
