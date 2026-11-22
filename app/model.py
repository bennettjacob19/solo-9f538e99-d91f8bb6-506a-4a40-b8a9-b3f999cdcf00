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
