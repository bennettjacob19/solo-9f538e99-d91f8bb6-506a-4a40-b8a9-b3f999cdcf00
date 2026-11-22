"""Batched target-only greedy continuation with a per-forward budget.

Several prompts of different lengths are decoded together.  A single model
forward can serve samples that are at different points in their own
continuation: padded query/key axes let the samples share one rectangular
attention computation while each keeps and extends its own KV cache at its own
absolute positions.

Budget
------
The caller supplies ``attention_score_budget``: an upper bound on the number
of attention-score elements materialised by *each* forward pass.  For a batch
of ``B`` samples with ``n_heads`` heads, ``Q`` padded query positions and
``K`` padded key positions that is exactly ``B * n_heads * Q * K``.  Padding is
counted, because the rectangular score tensor really does contain it.

Scheduling
----------
Every forward is packed from samples that individually fit.  Prefills are
scheduled shortest-first; if a long prompt does not fit next to the short ones,
it gets its own forward while the short prompts still share one.  The same
holds while decoding, where the padded cost is driven by the longest KV cache
in the group.  If even one sample alone exceeds the budget for a forward, the
whole batch is rejected with that sample's index (see
:class:`AttentionBudgetError`).

Correctness
-----------
Each sample follows exactly the emission rule of
:func:`app.decoding.greedy_decode`: take the argmax of its own latest logits,
stop on ``<eos>`` or when its own cap is reached, and only then evaluate the
newly committed token.  No sample ever sees another sample's tokens or cache,
so its continuation is the per-request greedy continuation; only the arithmetic
layout (padded batch) differs, which can move logits by ~1e-15.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import EOS_ID, CausalLM

MAX_BATCH_SAMPLES = 8


class BatchInputError(ValueError):
    """A batch sample cannot be evaluated (e.g. would exceed ``max_pos``)."""

    def __init__(self, sample_index: int, message: str):
        self.sample_index = sample_index
        super().__init__(f"sample {sample_index}: {message}")


class AttentionBudgetError(ValueError):
    """One sample alone needs more score elements than the budget allows.

    The whole batch is rejected; ``sample_index`` is the 0-based position of
    the offending sample in the request.
    """

    def __init__(self, *, sample_index: int, phase: str, required: int,
                 budget: int):
        self.sample_index = sample_index
        self.phase = phase
        self.required = required
        self.budget = budget
        super().__init__(
            f"sample {sample_index} alone requires {required} attention score "
            f"elements for a {phase} forward, which exceeds the budget of "
            f"{budget}"
        )


@dataclass
class BatchSampleState:
    index: int
    prompt: list[int]
    max_new_tokens: int
    continuation: list[int] = field(default_factory=list)
    cache: tuple[np.ndarray, np.ndarray] | None = None
    next_logits: np.ndarray | None = None
    stop_reason: str | None = None
    target_forward_calls: int = 0
    target_positions_evaluated: int = 0

    @property
    def done(self) -> bool:
        return self.stop_reason is not None


@dataclass
class ForwardRecord:
    forward_index: int
    phase: str
    sample_indices: list[int]
    batch_size: int
    query_positions: int
    key_positions: int
    n_heads: int
    attention_score_elements: int


@dataclass
class BatchDecodeResult:
    samples: list[BatchSampleState]
    forwards: list[ForwardRecord]

    @property
    def total_attention_score_elements(self) -> int:
        return sum(f.attention_score_elements for f in self.forwards)


def _pack_prefill(indices: list[int], lengths: list[int], n_heads: int,
                  budget: int) -> list[list[int]]:
    """Group prefill candidates shortest-first under the padded cost model."""
    order = sorted(indices, key=lambda i: (lengths[i], i))
    groups: list[list[int]] = []
    current: list[int] = []

    def check_single(i: int) -> None:
        length = lengths[i]
        required = n_heads * length * length
        if required > budget:
            raise AttentionBudgetError(
                sample_index=i, phase="prefill", required=required,
                budget=budget,
            )

    for i in order:
        if not current:
            check_single(i)
            current = [i]
            continue
        longest = max(lengths[i], max(lengths[j] for j in current))
        cost = (len(current) + 1) * n_heads * longest * longest
        if cost <= budget:
            current.append(i)
        else:
            groups.append(current)
            check_single(i)
            current = [i]
    if current:
        groups.append(current)
    return groups


def _pack_decode(indices: list[int], pasts: list[int], n_heads: int,
                 budget: int) -> list[list[int]]:
    """Group one-token decode candidates by padded cache length."""
    order = sorted(indices, key=lambda i: (pasts[i], i))
    groups: list[list[int]] = []
    current: list[int] = []

    def check_single(i: int) -> None:
        required = n_heads * (pasts[i] + 1)
        if required > budget:
            raise AttentionBudgetError(
                sample_index=i, phase="decode", required=required,
                budget=budget,
            )

    for i in order:
        if not current:
            check_single(i)
            current = [i]
            continue
        longest = max(pasts[i], max(pasts[j] for j in current))
        cost = (len(current) + 1) * n_heads * (longest + 1)
        if cost <= budget:
            current.append(i)
        else:
            groups.append(current)
            check_single(i)
            current = [i]
    if current:
        groups.append(current)
    return groups


def batched_greedy_decode(
    model: CausalLM,
    prompts: list[list[int]],
    max_new_tokens: list[int],
    attention_score_budget: int,
) -> BatchDecodeResult:
    """Greedy-decode each prompt, sharing forwards across samples.

    ``prompts[i]`` is prefilled in full and continues at position
    ``len(prompts[i])``.  Results stay in submission order.  Raises
    :class:`AttentionBudgetError` if any single sample cannot fit one forward
    on its own, and :class:`BatchInputError` if a sample would overflow
    ``model.max_pos``.
    """
    n_samples = len(prompts)
    if n_samples == 0:
        raise ValueError("batched_greedy_decode requires at least one prompt")
    if len(max_new_tokens) != n_samples:
        raise ValueError("one max_new_tokens value per prompt")
    if attention_score_budget < 1:
        raise ValueError("attention_score_budget must be >= 1")

    n_heads = model.n_heads
    states = [
        BatchSampleState(index=i, prompt=list(prompts[i]),
                         max_new_tokens=int(max_new_tokens[i]))
        for i in range(n_samples)
    ]
    forwards: list[ForwardRecord] = []

    # The prefill cost is known exactly (each prompt is prefilled in full), and
    # _pack_prefill raises before any forward runs if one sample alone does not
    # fit.  Decode cost grows with the cache, so it is checked when the decode
    # forward is actually scheduled: if a sample's cache makes even a solo
    # decode step exceed the budget, the batch aborts with that sample's index.

    def run(group: list[int], tokens: list[list[int]],
            phase: str) -> None:
        for i, tok in zip(group, tokens):
            past = 0 if states[i].cache is None else states[i].cache[0].shape[0]
            total = past + len(tok)
            if total > model.max_pos:
                raise BatchInputError(
                    i, f"sequence length {total} exceeds max_pos="
                       f"{model.max_pos} during {phase}"
                )
        logits, caches, info = model.forward_batch(
            tokens, [states[i].cache for i in group]
        )
        if info.score_elements > attention_score_budget:
            # Defensive: the packing functions use the same formula.
            raise RuntimeError(
                "packed forward exceeded the attention score budget"
            )
        forwards.append(
            ForwardRecord(
                forward_index=len(forwards),
                phase=phase,
                sample_indices=list(group),
                batch_size=info.batch_size,
                query_positions=info.query_positions,
                key_positions=info.key_positions,
                n_heads=info.n_heads,
                attention_score_elements=info.score_elements,
            )
        )
        for slot, i in enumerate(group):
            state = states[i]
            state.cache = caches[slot]
            state.next_logits = logits[slot][-1]
            state.target_forward_calls += 1
            state.target_positions_evaluated += len(tokens[slot])

    # --- prefill every prompt in full, shortest-first and budget-packed.
    lengths = [len(p) for p in prompts]
    for group in _pack_prefill(list(range(n_samples)), lengths, n_heads,
                               attention_score_budget):
        run(group, [states[i].prompt for i in group], "prefill")

    # --- step-wise continuation.  All still-active samples emit one token per
    # --- round and the samples that continue are batched in one forward.
    while True:
        pending: list[tuple[int, int]] = []  # (sample index, token to evaluate)
        for state in states:
            if state.done:
                continue
            token = int(np.argmax(state.next_logits))
            state.continuation.append(token)
            if token == EOS_ID:
                state.stop_reason = "eos"
                continue
            if len(state.continuation) >= state.max_new_tokens:
                state.stop_reason = "max_new_tokens"
                continue
            pending.append((state.index, token))
        if not pending:
            break

        jobs = [i for i, _ in pending]
        token_for = dict(pending)
        pasts = [0] * n_samples
        for i in jobs:
            pasts[i] = states[i].cache[0].shape[0]
        for group in _pack_decode(jobs, pasts, n_heads,
                                  attention_score_budget):
            run(group, [[token_for[i]] for i in group], "decode")

    return BatchDecodeResult(samples=states, forwards=forwards)
