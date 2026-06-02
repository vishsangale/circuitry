#!/usr/bin/env python3
"""Round-3 download warm-up: fetch OLMoE (MoE) + Gemma-2-2b (post-norm).

Gemma models are gated on HF; without an HF_TOKEN this will 401 — we detect
and report that cleanly so we know whether the Gemma test is runnable.
"""

from __future__ import annotations

import time

from huggingface_hub import snapshot_download

TARGETS = [
    ("allenai/OLMoE-1B-7B-0924", "MoE (64 experts, 8 active)"),
    ("google/gemma-2-2b", "post-norm (gated?)"),
]


def main() -> None:
    for repo, note in TARGETS:
        t0 = time.time()
        print(f"\n=== {repo}  [{note}] ===", flush=True)
        try:
            path = snapshot_download(
                repo,
                allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"],
            )
            print(f"  OK ({time.time()-t0:.0f}s) -> {path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("\nWARM-ROUND3 DONE", flush=True)


if __name__ == "__main__":
    main()
