#!/usr/bin/env python3
"""Run the two promised scenarios and print a compact report.

Usage (from the repository root):

    python examples/run_examples.py           # call the decoders in-process
    python examples/run_examples.py --url URL # call a running API instead

The in-process mode needs only NumPy (no HTTP server).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running straight from a checkout: python examples/run_examples.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES = [
    ("all drafts accepted", [2, 34, 24, 11], 16, 4),
    ("draft block rejected mid-way", [2, 22, 30, 11], 16, 4),
    ("EOS stop", [2, 62, 32, 11], 24, 4),
]

# Batch scenario exercised after the single-request cases.
BATCH_PROMPTS = [
    ([2, 34, 24, 11], 16),          # short prompt, fills its cap
    ([2], 5),                       # shortest prompt
    ([2, 62, 32, 11], 24),          # hits <eos> early and leaves the batch
    ([2, 22, 30, 11, 19, 3, 8, 12, 5, 7, 44, 19], 6),  # long prompt
]
BATCH_BUDGET = 4096


def local_batch_report(prompts, caps, budget):
    from app.batching import batched_greedy_decode
    from app.decoding import greedy_decode
    from app.model import TARGET_MODEL

    result = batched_greedy_decode(TARGET_MODEL, prompts, caps, budget)
    per_request = [greedy_decode(TARGET_MODEL, p, c) for p, c in zip(prompts, caps)]
    return {
        "identical": all(
            s.continuation == g.continuation and s.stop_reason == g.stop_reason
            for s, g in zip(result.samples, per_request)
        ),
        "samples": [
            {
                "continuation": s.continuation,
                "stop_reason": s.stop_reason,
            }
            for s in result.samples
        ],
        "forwards": [
            {
                "phase": f.phase,
                "samples": f.sample_indices,
                "score_elements": f.attention_score_elements,
            }
            for f in result.forwards
        ],
    }


def remote_batch_report(url, prompts, caps, budget):
    import urllib.error
    import urllib.request

    body = json.dumps({
        "prompts": [
            {"prompt_token_ids": p, "max_new_tokens": c}
            for p, c in zip(prompts, caps)
        ],
        "attention_score_budget": budget,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/decode/batch", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        payload = json.load(exc)
        detail = payload.get("detail", payload)
        if isinstance(detail, str):
            detail = {"message": detail}
        return {"status": exc.code, "detail": detail}
    return {
        "identical": True,
        "samples": [
            {
                "continuation": s["continuation_token_ids"],
                "stop_reason": s["stop_reason"],
            }
            for s in data["samples"]
        ],
        "forwards": [
            {
                "phase": f["phase"],
                "samples": f["sample_indices"],
                "score_elements": f["attention_score_elements"],
            }
            for f in data["forwards"]
        ],
    }


def local_report(prompt: list[int], max_new: int, block: int) -> dict:
    from app.decoding import greedy_decode, speculative_decode
    from app.model import DRAFT_MODEL, TARGET_MODEL

    greedy = greedy_decode(TARGET_MODEL, prompt, max_new)
    spec = speculative_decode(TARGET_MODEL, DRAFT_MODEL, prompt, max_new, block)
    return {
        "identical": greedy.continuation == spec.continuation,
        "greedy_calls": greedy.target_forward_calls,
        "spec_calls": spec.target_forward_calls,
        "stop_reason": spec.stop_reason,
        "continuation": spec.continuation,
        "accepted": spec.draft_tokens_accepted,
        "proposed": spec.draft_tokens_proposed,
        "rounds": [
            {
                "draft": r.draft_tokens,
                "accepted": r.accepted_count,
                "corrected": r.corrected_token,
                "emitted": r.emitted_tokens,
            }
            for r in spec.rounds
        ],
    }


def remote_report(url: str, prompt: list[int], max_new: int, block: int) -> dict:
    import urllib.request

    body = json.dumps({
        "prompt_token_ids": prompt,
        "max_new_tokens": max_new,
        "draft_block_size": block,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/decode", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return {
        "identical": data["comparison"]["outputs_identical"],
        "greedy_calls": data["greedy"]["target_forward_calls"],
        "spec_calls": data["speculative"]["target_forward_calls"],
        "stop_reason": data["speculative"]["stop_reason"],
        "continuation": data["speculative"]["continuation_token_ids"],
        "accepted": data["speculative"]["draft_tokens_accepted"],
        "proposed": data["speculative"]["draft_tokens_proposed"],
        "rounds": [
            {
                "draft": r["draft_tokens"],
                "accepted": r["accepted_count"],
                "corrected": r["corrected_token"],
                "emitted": r["emitted_tokens"],
            }
            for r in data["speculative"]["rounds"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="base URL of a running API, e.g. "
                                     "http://127.0.0.1:8000")
    args = parser.parse_args()

    ok = True
    for name, prompt, max_new, block in CASES:
        if args.url:
            rep = remote_report(args.url, prompt, max_new, block)
        else:
            rep = local_report(prompt, max_new, block)
        ok &= rep["identical"]

        print(f"\n=== {name} ===")
        print(f"prompt={prompt} max_new_tokens={max_new} "
              f"draft_block_size={block}")
        print(f"outputs identical: {rep['identical']}")
        print(f"target forwards: greedy={rep['greedy_calls']} "
              f"speculative={rep['spec_calls']} "
              f"(saved {rep['greedy_calls'] - rep['spec_calls']})")
        print(f"draft tokens accepted: {rep['accepted']}/{rep['proposed']}")
        print(f"stop reason: {rep['stop_reason']}")
        print(f"continuation: {rep['continuation']}")
        for i, r in enumerate(rep["rounds"]):
            print(f"  round {i}: draft={r['draft']} accepted={r['accepted']} "
                  f"corrected={r['corrected']} emitted={r['emitted']}")

    # Batch: mixed lengths, an early EOS exit, then a budget that is too small.
    if args.url:
        rep = remote_batch_report(args.url, [p for p, _ in BATCH_PROMPTS],
                                  [c for _, c in BATCH_PROMPTS], BATCH_BUDGET)
    else:
        rep = local_batch_report([p for p, _ in BATCH_PROMPTS],
                                 [c for _, c in BATCH_PROMPTS], BATCH_BUDGET)

    print("\n=== batch: mixed lengths + early EOS ===")
    if "samples" not in rep:
        print(f"unexpected error: {rep}")
        return 1
    ok &= rep["identical"]
    print(f"outputs identical to per-request /decode greedy: {rep['identical']}")
    for i, (prompt, cap) in enumerate(BATCH_PROMPTS):
        sample = rep["samples"][i]
        print(f"  sample {i}: prompt_len={len(prompt)} cap={cap} "
              f"stop={sample['stop_reason']} len={len(sample['continuation'])}"
              f" continuation={sample['continuation']}")
    for f in rep["forwards"]:
        print(f"  forward {f['phase']:7s} samples={f['samples']} "
              f"score_elements={f['score_elements']}")

    # Budget small enough to reject sample 3 (a 12-token prompt needs
    # 4*12*12 = 576 elements for its own prefill) but large enough for the
    # short prompts to keep batching together.
    insufficient = 256
    if args.url:
        bad = remote_batch_report(args.url, [p for p, _ in BATCH_PROMPTS],
                                  [c for _, c in BATCH_PROMPTS], insufficient)
    else:
        from app.batching import AttentionBudgetError

        try:
            local_batch_report([p for p, _ in BATCH_PROMPTS],
                               [c for _, c in BATCH_PROMPTS], insufficient)
        except AttentionBudgetError as exc:
            bad = {"detail": {
                "error": "attention_score_budget_exceeded",
                "message": str(exc),
                "sample_index": exc.sample_index,
            }}
        else:
            bad = {}

    print(f"\n=== batch: insufficient budget ({insufficient}) ===")
    detail = bad.get("detail", {})
    print(f"rejected: {detail.get('error')} at sample_index="
          f"{detail.get('sample_index')}")
    print(f"message: {detail.get('message')}")

    # The same prompts under a budget that still separates the long sample but
    # lets it run on its own: the short samples remain batched.
    squeezed = 576
    if args.url:
        sq = remote_batch_report(args.url, [p for p, _ in BATCH_PROMPTS],
                                 [c for _, c in BATCH_PROMPTS], squeezed)
    else:
        sq = local_batch_report([p for p, _ in BATCH_PROMPTS],
                                [c for _, c in BATCH_PROMPTS], squeezed)

    print(f"\n=== batch: tight budget ({squeezed}), long sample runs alone ===")
    if "samples" in sq:
        ok &= sq["identical"]
        print(f"outputs identical: {sq['identical']}")
        print(f"prefill groups: "
              f"{[f['samples'] for f in sq['forwards'] if f['phase'] == 'prefill']}")
        print(f"max score elements in a forward: "
              f"{max(f['score_elements'] for f in sq['forwards'])}")
    else:
        print(f"unexpected error: {sq}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
