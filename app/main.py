"""FastAPI app exposing side-by-side greedy vs speculative decoding."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .batching import (
    MAX_BATCH_SAMPLES,
    AttentionBudgetError,
    BatchInputError,
    batched_greedy_decode,
)
from .decoding import DecodeResult, greedy_decode, speculative_decode
from .model import (
    DRAFT_MODEL,
    EOS_ID,
    TARGET_MODEL,
    VOCAB_SIZE,
    VOCAB_WORDS,
    ModelError,
    model_digest,
)

MAX_PROMPT_TOKENS = 256
MAX_BATCH_MAX_NEW_TOKENS = 128
MAX_ATTENTION_SCORE_BUDGET = 50_000_000

app = FastAPI(
    title="Speculative Decoding Reference API",
    version=__version__,
    description=(
        "Compares **speculative decoding** against **token-by-token greedy "
        "decoding** with two deterministic single-layer causal-attention "
        "NumPy models that share a 64-token vocabulary. Both decoders reuse "
        "the target model's KV cache for the confirmed prefix, so no confirmed "
        "position is ever evaluated twice. Speculative decoding verifies one "
        "whole draft block in a single target forward pass; on the first "
        "mismatch it keeps the accepted prefix, emits the target's own token "
        "instead, and both models continue from the corrected sequence.\n\n"
        "Both outputs are guaranteed identical, and generation stops at <eos> "
        "or at `max_new_tokens`. The response reports how many target forward "
        "passes each path actually made.\n\n"
        "Sample inputs: **all drafts accepted** "
        "`{\"prompt_token_ids\": [2, 34, 24, 11], \"max_new_tokens\": 16, "
        "\"draft_block_size\": 4}`; **mid-block rejection** "
        "`{\"prompt_token_ids\": [2, 22, 30, 11], \"max_new_tokens\": 16, "
        "\"draft_block_size\": 4}`."
    ),
)


class DecodeRequest(BaseModel):
    prompt_token_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_TOKENS,
        description="Prompt token ids in [0, 63]. Token 1 is <eos>, 2 is <bos>.",
        examples=[[2, 34, 24, 11]],
    )
    max_new_tokens: int = Field(
        16,
        ge=1,
        le=128,
        description="Maximum number of continuation tokens to generate.",
    )
    draft_block_size: int = Field(
        4,
        ge=1,
        le=32,
        description="Number of tokens the draft model proposes per round.",
    )

    @field_validator("prompt_token_ids")
    @classmethod
    def check_ids(cls, ids: list[int]) -> list[int]:
        bad = [t for t in ids if t < 0 or t >= VOCAB_SIZE]
        if bad:
            raise ValueError(
                f"token ids must be in [0, {VOCAB_SIZE - 1}], got {bad[:5]}"
            )
        return ids


class Round(BaseModel):
    round_index: int
    draft_tokens: list[int]
    accepted_count: int
    corrected_token: int | None
    emitted_tokens: list[int]
    target_forward_calls: int
    all_drafts_accepted: bool


class DecoderOutput(BaseModel):
    continuation_token_ids: list[int]
    continuation_words: list[str]
    full_sequence_token_ids: list[int]
    target_forward_calls: int
    target_positions_evaluated: int
    stop_reason: str


class GreedyOutput(DecoderOutput):
    pass


class SpeculativeOutput(DecoderOutput):
    draft_forward_calls: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    acceptance_rate: float | None
    rounds: list[Round]


class ModelInfo(BaseModel):
    name: str
    digest: str


class Comparison(BaseModel):
    outputs_identical: bool
    target_forward_calls_saved: int
    speedup: float | None
    note: str


class DecodeResponse(BaseModel):
    prompt_token_ids: list[int]
    max_new_tokens: int
    draft_block_size: int
    eos_token_id: int
    models: dict
    greedy: GreedyOutput
    speculative: SpeculativeOutput
    comparison: Comparison


class BatchPrompt(BaseModel):
    prompt_token_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_TOKENS,
        description="Prompt token ids in [0, 63]. Token 1 is <eos>, 2 is <bos>.",
        examples=[[2, 34, 24, 11]],
    )
    max_new_tokens: int = Field(
        16,
        ge=1,
        le=MAX_BATCH_MAX_NEW_TOKENS,
        description="Per-sample continuation cap.",
    )

    @field_validator("prompt_token_ids")
    @classmethod
    def check_ids(cls, ids: list[int]) -> list[int]:
        bad = [t for t in ids if t < 0 or t >= VOCAB_SIZE]
        if bad:
            raise ValueError(
                f"token ids must be in [0, {VOCAB_SIZE - 1}], got {bad[:5]}"
            )
        return ids


class BatchGreedyRequest(BaseModel):
    prompts: list[BatchPrompt] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SAMPLES,
        description="Up to 8 prompts; each is prefilled in full and decoded "
                    "with its own cap.",
    )
    attention_score_budget: int = Field(
        ...,
        ge=1,
        le=MAX_ATTENTION_SCORE_BUDGET,
        description=(
            "Maximum attention-score elements one forward pass may "
            "materialise: batch_size * n_heads * padded_query_positions * "
            "padded_key_positions, counting padding."
        ),
        examples=[1_000_000],
    )


class BatchSampleOutput(BaseModel):
    sample_index: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    continuation_token_ids: list[int]
    continuation_words: list[str]
    full_sequence_token_ids: list[int]
    stop_reason: str
    target_forward_calls: int
    target_positions_evaluated: int


class BatchForwardOutput(BaseModel):
    forward_index: int
    phase: str = Field(description="'prefill' or 'decode'.")
    sample_indices: list[int] = Field(
        description="Submission-order indexes of the samples in this forward."
    )
    batch_size: int
    query_positions: int
    key_positions: int
    n_heads: int
    attention_score_elements: int


class BatchGreedyResponse(BaseModel):
    attention_score_budget: int
    eos_token_id: int
    samples: list[BatchSampleOutput]
    forwards: list[BatchForwardOutput]
    total_forwards: int
    total_attention_score_elements: int
    equals_single_decode_greedy: bool
    note: str


def _greedy_payload(result: DecodeResult) -> dict:
    return {
        "continuation_token_ids": result.continuation,
        "continuation_words": [VOCAB_WORDS[t] for t in result.continuation],
        "full_sequence_token_ids": result.full_sequence,
        "target_forward_calls": result.target_forward_calls,
        "target_positions_evaluated": result.target_positions_evaluated,
        "stop_reason": result.stop_reason,
    }


def _speculative_payload(result: DecodeResult) -> dict:
    return {
        "continuation_token_ids": result.continuation,
        "continuation_words": [VOCAB_WORDS[t] for t in result.continuation],
        "full_sequence_token_ids": result.full_sequence,
        "target_forward_calls": result.target_forward_calls,
        "target_positions_evaluated": result.target_positions_evaluated,
        "stop_reason": result.stop_reason,
        "draft_forward_calls": result.draft_forward_calls,
        "draft_tokens_proposed": result.draft_tokens_proposed,
        "draft_tokens_accepted": result.draft_tokens_accepted,
        "acceptance_rate": (
            result.draft_tokens_accepted / result.draft_tokens_proposed
            if result.draft_tokens_proposed
            else None
        ),
        "rounds": [
            {
                "round_index": r.round_index,
                "draft_tokens": r.draft_tokens,
                "accepted_count": r.accepted_count,
                "corrected_token": r.corrected_token,
                "emitted_tokens": r.emitted_tokens,
                "target_forward_calls": r.target_forward_calls,
                "all_drafts_accepted": r.corrected_token is None,
            }
            for r in result.rounds
        ],
    }


def _param_count(model) -> int:
    return int(
        model.embed.size + model.pos.size + model.w_q.size + model.w_k.size
        + model.w_v.size + model.w_o.size + model.w_in.size
        + model.w_out.size + model.gamma_1.size + model.gamma_2.size
        + model.embed_out.size
    )


def _model_payload(model) -> dict:
    return {
        "name": model.name,
        "layers": 1,
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "d_ff": int(model.w_in.shape[1]),
        "ffn_macs_per_token": int(2 * model.w_in.size),
        "params": _param_count(model),
        "digest": model_digest(model),
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": EOS_ID,
        "target_model": TARGET_MODEL.name,
        "draft_model": DRAFT_MODEL.name,
    }


@app.get("/models")
def models() -> dict:
    return {
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": EOS_ID,
        "target": _model_payload(TARGET_MODEL),
        "draft": _model_payload(DRAFT_MODEL),
    }


@app.post("/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest) -> DecodeResponse:
    try:
        greedy = greedy_decode(TARGET_MODEL, req.prompt_token_ids,
                               req.max_new_tokens)
        spec = speculative_decode(
            TARGET_MODEL,
            DRAFT_MODEL,
            req.prompt_token_ids,
            req.max_new_tokens,
            req.draft_block_size,
        )
    except ModelError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if greedy.continuation != spec.continuation:
        # Never expected: a mismatch means speculative decoding is no longer
        # exact, so report it loudly instead of returning a wrong answer.
        raise HTTPException(
            status_code=500,
            detail={
                "error": "decoders disagree",
                "greedy": greedy.continuation,
                "speculative": spec.continuation,
            },
        )

    saved = greedy.target_forward_calls - spec.target_forward_calls
    return DecodeResponse(
        prompt_token_ids=req.prompt_token_ids,
        max_new_tokens=req.max_new_tokens,
        draft_block_size=req.draft_block_size,
        eos_token_id=EOS_ID,
        models={
            "target": ModelInfo(name=TARGET_MODEL.name,
                                digest=model_digest(TARGET_MODEL)),
            "draft": ModelInfo(name=DRAFT_MODEL.name,
                               digest=model_digest(DRAFT_MODEL)),
            "shared_vocab_size": VOCAB_SIZE,
        },
        greedy=GreedyOutput(**_greedy_payload(greedy)),
        speculative=SpeculativeOutput(**_speculative_payload(spec)),
        comparison=Comparison(
            outputs_identical=True,
            target_forward_calls_saved=saved,
            speedup=round(greedy.target_forward_calls
                          / spec.target_forward_calls, 4),
            note=(
                "Identical outputs by construction. Both paths reuse the "
                "target KV cache over the confirmed prefix; speculative "
                "decoding just verifies a whole draft block per target call."
            ),
        ),
    )


@app.post(
    "/decode/batch",
    response_model=BatchGreedyResponse,
    summary="Batched target-only greedy continuation under a score budget",
    description=(
        "Decode up to 8 prompts together. Every prompt is prefilled in full; "
        "prefill and continuation both batch samples of different lengths "
        "into one model forward. Each forward's attention-score elements "
        "(batch * heads * padded queries * padded keys, padding included) "
        "stay within `attention_score_budget`; a long sample that does not "
        "fit is given its own forward while the short ones are still batched. "
        "Each sample reuses its own KV cache and continues at its own "
        "positions, leaves the batch on <eos> or at its cap, and its output "
        "equals calling POST /decode and reading `greedy` for that prompt. "
        "If one sample cannot fit a forward even alone, the batch returns 422 "
        "with its index."
    ),
)
def decode_batch(req: BatchGreedyRequest) -> BatchGreedyResponse:
    prompts = [p.prompt_token_ids for p in req.prompts]
    caps = [p.max_new_tokens for p in req.prompts]
    try:
        result = batched_greedy_decode(
            TARGET_MODEL, prompts, caps, req.attention_score_budget
        )
    except AttentionBudgetError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "attention_score_budget_exceeded",
                "message": str(exc),
                "sample_index": exc.sample_index,
                "phase": exc.phase,
                "required_attention_score_elements": exc.required,
                "attention_score_budget": exc.budget,
            },
        ) from exc
    except BatchInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "sample_too_long",
                "message": str(exc),
                "sample_index": exc.sample_index,
                "max_pos": TARGET_MODEL.max_pos,
            },
        ) from exc
    except ModelError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    samples = [
        BatchSampleOutput(
            sample_index=s.index,
            prompt_token_ids=s.prompt,
            max_new_tokens=s.max_new_tokens,
            continuation_token_ids=s.continuation,
            continuation_words=[VOCAB_WORDS[t] for t in s.continuation],
            full_sequence_token_ids=s.prompt + s.continuation,
            stop_reason=s.stop_reason or "max_new_tokens",
            target_forward_calls=s.target_forward_calls,
            target_positions_evaluated=s.target_positions_evaluated,
        )
        for s in result.samples
    ]
    forwards = [
        BatchForwardOutput(
            forward_index=f.forward_index,
            phase=f.phase,
            sample_indices=f.sample_indices,
            batch_size=f.batch_size,
            query_positions=f.query_positions,
            key_positions=f.key_positions,
            n_heads=f.n_heads,
            attention_score_elements=f.attention_score_elements,
        )
        for f in result.forwards
    ]
    return BatchGreedyResponse(
        attention_score_budget=req.attention_score_budget,
        eos_token_id=EOS_ID,
        samples=samples,
        forwards=forwards,
        total_forwards=len(forwards),
        total_attention_score_elements=(
            result.total_attention_score_elements
        ),
        equals_single_decode_greedy=True,
        note=(
            "Each sample is the greedy continuation POST /decode returns for "
            "that prompt: its own KV cache, its own positions, full prompt "
            "prefill, and the same <eos>/cap stop rule. Batching changes only "
            "the arithmetic layout, which can move logits by ~1e-15."
        ),
    )
