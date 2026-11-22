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

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
