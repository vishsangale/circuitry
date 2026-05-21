# scripts/parity_check.py
"""Numerical parity between circuitry and the in-tree mendu inspector.

Run in M2 (mendu cutover). The script trains a tiny canonical model under
both pipelines, captures all TB scalars from each, and asserts they agree
within the tolerances from docs/design.md §7 Phase M2:

  - Most metrics:        rtol=1e-5, atol=1e-7
  - SVD-derived metrics: rtol=1e-4   (effective_rank, condition_number,
                                       heavy_tail_alpha, singular_values)

M1 ships the harness; M2 wires it up against ~/workspace/mendu and ratchets
tolerances down if a metric is empirically tighter than expected.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_TOLERANCES = {
    "default": {"rtol": 1e-5, "atol": 1e-7},
    "svd": {"rtol": 1e-4, "atol": 1e-6},
}
SVD_METRICS = {
    "effective_rank",
    "condition_number",
    "heavy_tail_alpha",
    "singular_values",
    "stable_rank",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mendu-root", required=True,
                   help="path to ~/workspace/mendu")
    p.add_argument("--steps", type=int, default=20)
    _ = p.parse_args()
    print("parity_check.py: M2 harness — body fills in during mendu cutover.")
    print("Tolerances:", DEFAULT_TOLERANCES)
    print("SVD-bucket metrics:", sorted(SVD_METRICS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
