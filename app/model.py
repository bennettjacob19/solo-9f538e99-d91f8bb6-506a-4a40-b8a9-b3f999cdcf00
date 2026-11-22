"""Deterministic single-layer causal-attention language models in NumPy.

Two models share one small vocabulary:

* ``TARGET_MODEL`` is the model whose forward passes we count and want to
  minimise.
* ``DRAFT_MODEL`` is a smaller model that proposes blocks of tokens.  It keeps
  the target's attention and embedding weights and gives the feed-forward
  network 24 hidden channels instead of 64, so it does 62.5% fewer FFN MACs
  per token.  The narrow FFN is distilled deterministically at import time
  from the target on a fixed synthetic corpus (see
  ``CausalLM.distilled_draft``), which keeps it agreeing with the target's
  greedy choice on about 60% of positions -- a useful, but visibly imperfect,
  proposer.

Both are fully deterministic: fixed seeds, float64 arithmetic, no sampling
and no dropout.

Layouts
-------
``tokens``: 1-D int array of token ids, shape ``(n,)``.
Cache: ``(k, v)`` each shaped ``(n_past, n_heads, head_dim)``.
Logits: ``(n, vocab_size)``, one row per input position.

The cache returned by :meth:`CausalLM.forward` covers the whole prefix
(cached + newly supplied positions), so callers can extend it or slice it
back to a shorter confirmed prefix.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np

VOCAB_SIZE = 64
EOS_ID = 1
BOS_ID = 2
PAD_ID = 0
MAX_POS = 512
SEED_TARGET = 20240517

# Token 0 is padding and is never generated; 1 is EOS; 2 is BOS; the rest map
# to short human-readable words so example outputs are easy to scan.
VOCAB_WORDS = ["<pad>", "<eos>", "<bos>", "<unk>"] + [
    f"w{i}" for i in range(4, VOCAB_SIZE)
]


class ModelError(ValueError):
    """Raised when model inputs are malformed."""


def _layer_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma


def _gelu(x: np.ndarray) -> np.ndarray:
    # Tanh approximation of GELU: vectorised and deterministic.
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class BatchForwardInfo:
    """Shape/accounting details of one batched forward pass.

    ``query_positions`` is the padded query width shared by the batch (it is
    larger than the per-sample new-token count when samples have different
    lengths).  ``score_elements`` is the exact number of attention-score
    elements the rectangular forward computes:

        batch_size * n_heads * query_positions * key_positions

    Padding on both the query and key axes is included, as required for
    compute accounting.
    """

    batch_size: int
    n_heads: int
    query_positions: int
    key_positions: int
    score_elements: int


@dataclass(frozen=True)
class CausalLM:
    """A pre-norm single-layer causal attention transformer."""

    name: str
    embed: np.ndarray  # (vocab, d_model)
    pos: np.ndarray  # (max_pos, d_model)
    w_q: np.ndarray  # (d_model, d_model)
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray
    w_in: np.ndarray  # (d_model, d_ff)
    w_out: np.ndarray  # (d_ff, d_model)
    gamma_1: np.ndarray  # (d_model,)
    gamma_2: np.ndarray  # (d_model,)
    embed_out: np.ndarray  # (d_model, vocab)
    n_heads: int
    max_pos: int = MAX_POS
    eos_id: int = EOS_ID

    @property
    def vocab_size(self) -> int:
        return int(self.embed.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.embed.shape[1])

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ModelError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    # ------------------------------------------------------------------ init
    @classmethod
    def build(cls, name: str, seed: int, *, d_model: int = 32,
              n_heads: int = 4, d_ff: int = 64,
              max_pos: int = MAX_POS, eos_id: int = EOS_ID) -> CausalLM:
        rng = np.random.default_rng(seed)

        def mat(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier-uniform
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        return cls(
            name=name,
            embed=rng.normal(0.0, 0.3, size=(VOCAB_SIZE, d_model)),
            pos=rng.normal(0.0, 0.1, size=(max_pos, d_model)),
            w_q=mat(d_model, d_model),
            w_k=mat(d_model, d_model),
            w_v=mat(d_model, d_model),
            w_o=mat(d_model, d_model),
            w_in=mat(d_model, d_ff),
            w_out=mat(d_ff, d_model),
            gamma_1=np.ones(d_model),
            gamma_2=np.ones(d_model),
            embed_out=mat(d_model, VOCAB_SIZE),
            n_heads=n_heads,
            max_pos=max_pos,
            eos_id=eos_id,
        )

    @classmethod
    def distilled_draft(cls, name: str, source: CausalLM, rank: int,
                        *, seed: int = 0, samples: int = 256,
                        noise: float = 0.0) -> CausalLM:
        """Build the draft: the target with a narrow, distilled feed-forward.

        The FFN dominates a one-layer transformer's compute, so the draft keeps
        every attention/embedding weight and squeezes the FFN from ``d_ff``
        hidden channels down to ``rank``.  ``w_in`` is therefore
        ``(d_model, rank)`` and ``w_out`` is ``(rank, d_model)``: 62.5% fewer
        FFN MACs at rank 24 vs d_ff 64.

        Which channels to keep and how to re-fit them is learned offline, at
        import time, from the target itself:

        1. A fixed seeded corpus of prompts is run up to the target's FFN,
           giving real pre-FFN activations ``X``.
        2. Hidden channels are ranked by ``||w_in[:, j]|| * ||w_out[j, :]||``
           and the top ``rank`` are selected.
        3. ``w_out`` (restricted to those channels) is replaced by the
           least-squares map from the narrowed activations back to the target's
           full FFN output, fitted on ``X``.

        ``noise`` optionally perturbs the fitted weights so the draft is not
        exact; every step is seeded, so the result is bit-for-bit reproducible.
        """
        if not 1 <= rank < source.w_in.shape[1]:
            raise ModelError("draft rank must be in [1, d_ff)")
        x = _sample_pre_ffn_inputs(source, seed=seed, samples=samples)
        target_out = _gelu(x @ source.w_in) @ source.w_out

        channel_score = (np.linalg.norm(source.w_in, axis=0)
                         * np.linalg.norm(source.w_out, axis=1))
        keep = np.argsort(channel_score)[::-1][:rank]
        w_in = source.w_in[:, keep]
        hidden = _gelu(x @ w_in)
        # Ridge-regularised least squares, solved with a tiny deterministic
        # jitter so the normal equations are always well conditioned.
        gram = hidden.T @ hidden
        ridge = 1e-6 * float(np.trace(gram)) / rank
        w_out = np.linalg.solve(gram + ridge * np.eye(rank),
                                hidden.T @ target_out)

        if noise:
            rng = np.random.default_rng(seed + 1)

            def jitter(a: np.ndarray) -> np.ndarray:
                scale = max(1e-8, float(np.abs(a).mean()))
                return a + rng.normal(0.0, noise * scale, size=a.shape)

            w_in, w_out = jitter(w_in), jitter(w_out)
        return replace(source, name=name, w_in=w_in, w_out=w_out)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        tokens: np.ndarray | list[int],
        cache: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Run the model over ``tokens`` appended to ``cache``.

        Returns ``(logits, new_cache)``.  ``logits[i]`` predicts the token
        following the i-th supplied position.  At least one new token is
        required; reusing a cache lets callers evaluate only new positions.
        """
        tok = np.asarray(tokens, dtype=np.int64)
        if tok.ndim != 1:
            raise ModelError("forward expects a 1-D token array")
        if tok.size == 0:
            raise ModelError("forward requires at least one new token")
        if tok.min() < 0 or tok.max() >= self.vocab_size:
            raise ModelError(f"token id out of range [0, {self.vocab_size - 1}]")

        hd = self.head_dim
        if cache is None:
            k_past = np.zeros((0, self.n_heads, hd), dtype=np.float64)
            v_past = np.zeros((0, self.n_heads, hd), dtype=np.float64)
        else:
            k_past, v_past = cache
            if k_past.ndim != 3 or k_past.shape != v_past.shape:
                raise ModelError("malformed KV cache")
            if k_past.shape[1:] != (self.n_heads, hd):
                raise ModelError("KV cache has the wrong head shape")
        past = int(k_past.shape[0])
        n_new = int(tok.size)
        total = past + n_new
        if total > self.max_pos:
            raise ModelError(
                f"sequence length {total} exceeds max_pos={self.max_pos}"
            )

        positions = np.arange(past, total, dtype=np.int64)
        x = self.embed[tok] + self.pos[positions]

        # Pre-norm causal attention.
        h = _layer_norm(x, self.gamma_1)
        q = (h @ self.w_q).reshape(n_new, self.n_heads, hd).transpose(1, 0, 2)
        k_new = (h @ self.w_k).reshape(n_new, self.n_heads, hd).transpose(1, 0, 2)
        v_new = (h @ self.w_v).reshape(n_new, self.n_heads, hd).transpose(1, 0, 2)

        k_all = np.concatenate([k_past.transpose(1, 0, 2), k_new], axis=1)
        v_all = np.concatenate([v_past.transpose(1, 0, 2), v_new], axis=1)

        scores = np.einsum("hnd,hkd->hnk", q, k_all) / np.sqrt(hd)
        # Query for new position i sits at absolute index past + i, so it may
        # attend to cached keys and to new keys 0..i.
        q_idx = np.arange(n_new)[:, None]
        k_idx = np.arange(total)[None, :]
        allowed = k_idx <= (past + q_idx)
        scores = np.where(allowed[None, :, :], scores, -np.inf)

        attn = np.einsum("hnk,hkd->hnd", _softmax(scores), v_all)
        x = x + attn.transpose(1, 0, 2).reshape(n_new, self.d_model) @ self.w_o

        # Pre-norm feed-forward.
        h = _layer_norm(x, self.gamma_2)
        x = x + _gelu(h @ self.w_in) @ self.w_out
        x = _layer_norm(x, self.gamma_2)
        logits = x @ self.embed_out

        new_cache = (
            np.concatenate([k_past, k_new.transpose(1, 0, 2)], axis=0),
            np.concatenate([v_past, v_new.transpose(1, 0, 2)], axis=0),
        )
        return logits, new_cache

    # ---------------------------------------------------------- batched forward
    def forward_batch(
        self,
        tokens: list[list[int]] | list[np.ndarray],
        caches: list[tuple[np.ndarray, np.ndarray] | None] | None = None,
    ) -> tuple[
        list[np.ndarray],
        list[tuple[np.ndarray, np.ndarray]],
        BatchForwardInfo,
    ]:
        """One forward pass shared by several samples of different lengths.

        Every sample supplies a non-empty run of new tokens and its own KV
        cache (the cache may be ``None``).  The samples are laid out in one
        padded rectangular batch: the query axis is widened to the longest new
        run, the key axis to ``max_past + max_new``.  Cached keys are
        right-aligned on the key axis so that a sample's positions are its own
        (each sample appends at ``len(its cache)``), and the causal mask is
        built from those absolute positions.  Padded query rows are masked to
        a uniform distribution and never contribute to a real result; padded
        keys are masked out of every real query.  Padding *is* included in the
        reported attention-score element count, because the rectangular
        ``(batch, heads, padded queries, padded keys)`` score tensor is what
        the pass actually materialises.

        Returns ``(logits, caches, info)`` where ``logits[i]`` has shape
        ``(len(tokens[i]), vocab)`` -- exactly the rows for sample ``i``'s new
        tokens -- and ``caches[i]`` covers that sample's full prefix.
        """
        n_samples = len(tokens)
        if n_samples == 0:
            raise ModelError("forward_batch requires at least one sample")
        if caches is None:
            caches = [None] * n_samples
        if len(caches) != n_samples:
            raise ModelError("forward_batch: one cache entry per sample")

        hd = self.head_dim
        toks: list[np.ndarray] = []
        for i, seq in enumerate(tokens):
            tok = np.asarray(seq, dtype=np.int64)
            if tok.ndim != 1:
                raise ModelError(
                    f"forward_batch: sample {i} is not a 1-D token array"
                )
            if tok.size == 0:
                raise ModelError(
                    f"forward_batch: sample {i} needs at least one new token"
                )
            if tok.min() < 0 or tok.max() >= self.vocab_size:
                raise ModelError(
                    f"forward_batch: sample {i} token id out of range "
                    f"[0, {self.vocab_size - 1}]"
                )
            toks.append(tok)

        pasts: list[int] = []
        for i, cache in enumerate(caches):
            if cache is None:
                pasts.append(0)
                continue
            k_past, v_past = cache
            if k_past.ndim != 3 or k_past.shape != v_past.shape:
                raise ModelError(f"forward_batch: malformed cache for sample {i}")
            if k_past.shape[1:] != (self.n_heads, hd):
                raise ModelError(
                    f"forward_batch: cache for sample {i} has the wrong head "
                    "shape"
                )
            pasts.append(int(k_past.shape[0]))

        new_lengths = [int(t.size) for t in toks]
        totals = [p + n for p, n in zip(pasts, new_lengths)]
        for i, total in enumerate(totals):
            if total > self.max_pos:
                raise ModelError(
                    f"forward_batch: sample {i} sequence length {total} "
                    f"exceeds max_pos={self.max_pos}"
                )

        batch = n_samples
        q_width = max(new_lengths)
        max_past = max(pasts)
        k_width = max_past + q_width

        tok_pad = np.full((batch, q_width), PAD_ID, dtype=np.int64)
        pos_idx = np.zeros((batch, q_width), dtype=np.int64)
        q_abs = np.full((batch, q_width), -1, dtype=np.int64)
        for i, (tok, past) in enumerate(zip(toks, pasts)):
            n = new_lengths[i]
            tok_pad[i, :n] = tok
            pos_idx[i, :n] = past + np.arange(n)
            q_abs[i, :n] = past + np.arange(n)

        x = self.embed[tok_pad] + self.pos[pos_idx]

        h = _layer_norm(x, self.gamma_1)
        q = (h @ self.w_q).reshape(batch, q_width, self.n_heads, hd)
        q = q.transpose(0, 2, 1, 3)  # (B, H, Q, hd)
        k_new = (h @ self.w_k).reshape(batch, q_width, self.n_heads, hd)
        v_new = (h @ self.w_v).reshape(batch, q_width, self.n_heads, hd)

        k_all = np.zeros((batch, k_width, self.n_heads, hd), dtype=np.float64)
        v_all = np.zeros_like(k_all)
        k_abs = np.full((batch, k_width), -1, dtype=np.int64)
        for i, (past, n) in enumerate(zip(pasts, new_lengths)):
            cache = caches[i]
            if past:
                # Right-align the cached keys so index `past` of this sample is
                # the same slot as in an unpadded single-sample pass.
                k_all[i, max_past - past:max_past] = cache[0]
                v_all[i, max_past - past:max_past] = cache[1]
                start = max_past - past
                k_abs[i, start:max_past] = np.arange(past)
            k_all[i, max_past:max_past + n] = k_new[i, :n]
            v_all[i, max_past:max_past + n] = v_new[i, :n]
            k_abs[i, max_past:max_past + n] = past + np.arange(n)

        # (B, H, Q, K): every element, padding included, is materialised and
        # therefore counted in the budget.
        scores = np.einsum("bhqd,bkhd->bhqk", q, k_all) / np.sqrt(hd)
        allowed = (
            (q_abs[:, :, None] >= 0)
            & (k_abs[:, None, :] >= 0)
            & (k_abs[:, None, :] <= q_abs[:, :, None])
        )
        scores = np.where(allowed[:, None, :, :], scores, -np.inf)
        padded_queries = q_abs < 0
        if np.any(padded_queries):
            # Rows for padding queries are not part of any result; give them a
            # finite uniform distribution so softmax cannot produce NaNs.
            scores = np.where(padded_queries[:, None, :, None], 0.0, scores)

        attn = np.einsum("bhqk,bkhd->bhqd", _softmax(scores), v_all)
        x = x + attn.transpose(0, 2, 1, 3).reshape(
            batch, q_width, self.d_model
        ) @ self.w_o

        h = _layer_norm(x, self.gamma_2)
        x = x + _gelu(h @ self.w_in) @ self.w_out
        x = _layer_norm(x, self.gamma_2)
        logits = x @ self.embed_out

        out_logits = [logits[i, :new_lengths[i]] for i in range(batch)]
        out_caches = []
        for i, (past, n) in enumerate(zip(pasts, new_lengths)):
            k_i = k_new[i, :n]
            v_i = v_new[i, :n]
            if past:
                k_i = np.concatenate([caches[i][0], k_i], axis=0)
                v_i = np.concatenate([caches[i][1], v_i], axis=0)
            out_caches.append((k_i, v_i))

        info = BatchForwardInfo(
            batch_size=batch,
            n_heads=self.n_heads,
            query_positions=q_width,
            key_positions=k_width,
            score_elements=int(batch * self.n_heads * q_width * k_width),
        )
        return out_logits, out_caches, info


# --------------------------------------------------------------------------
# Distillation helpers (deterministic linear algebra only).
# --------------------------------------------------------------------------
def _sample_pre_ffn_inputs(model: CausalLM, *, seed: int,
                           samples: int) -> np.ndarray:
    """Collect the residual stream entering ``model``'s FFN on a fixed corpus.

    Uses the same forward pass as the model itself up to (but not including)
    the FFN, so the activations are exactly what the target would see.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(samples):
        length = int(rng.integers(2, 24))
        tokens = rng.integers(3, model.vocab_size, size=length)
        chunks.append(_pre_ffn_residual(model, tokens))
    return np.concatenate(chunks, axis=0)


def _pre_ffn_residual(model: CausalLM, tokens: np.ndarray) -> np.ndarray:
    """Full-sequence residual stream after attention, before the FFN."""
    n = len(tokens)
    hd = model.head_dim
    x = model.embed[tokens] + model.pos[:n]

    h = _layer_norm(x, model.gamma_1)
    q = (h @ model.w_q).reshape(n, model.n_heads, hd).transpose(1, 0, 2)
    k = (h @ model.w_k).reshape(n, model.n_heads, hd).transpose(1, 0, 2)
    v = (h @ model.w_v).reshape(n, model.n_heads, hd).transpose(1, 0, 2)

    scores = np.einsum("hnd,hkd->hnk", q, k) / np.sqrt(hd)
    mask = np.tril(np.ones((n, n), dtype=bool))
    scores = np.where(mask[None, :, :], scores, -np.inf)
    attn = np.einsum("hnk,hkd->hnd", _softmax(scores), v)
    return x + attn.transpose(1, 0, 2).reshape(n, model.d_model) @ model.w_o


SEED_DRAFT = SEED_TARGET + 1
DRAFT_FFN_RANK = 24

TARGET_MODEL = CausalLM.build(
    "target", SEED_TARGET, d_model=32, n_heads=4, d_ff=64
)
# Same architecture, FFN narrowed to 24 of 64 channels and re-fitted against
# the target on a fixed corpus.  It agrees with the target on most (not all)
# greedy choices, so both the accepted and rejected block paths get exercised.
DRAFT_MODEL = CausalLM.distilled_draft(
    "draft", TARGET_MODEL, DRAFT_FFN_RANK, seed=SEED_TARGET, noise=0.002
)


def model_digest(model: CausalLM) -> str:
    """Short stable fingerprint of a model's parameters."""
    h = hashlib.sha256()
    for arr in (
        model.embed, model.pos, model.w_q, model.w_k, model.w_v, model.w_o,
        model.w_in, model.w_out, model.gamma_1, model.gamma_2, model.embed_out,
    ):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
