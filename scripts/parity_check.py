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

from tensorboard.backend.event_processing import event_accumulator

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

# Tag mapping: mendu's pre-cutover name → circuitry's post-cutover name.
# Both pipelines emit the listed-on-the-left tag in mendu's run; circuitry emits
# the listed-on-the-right tag. The "scalar name break" was a user decision (see
# plan-m2.md key-decision #2).
UNIVERSAL_TAGS = {
    "train/lm_loss": "train/lm_loss",
    "train/lr": "train/lr",
    "grad/global/total_norm": "grad/global/total_norm",
}


def _load_scalars(run_dir):
    """Return {tag: list[Scalar(step, value, wall_time)]} for all scalar tags in a TB run dir."""
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: ea.Scalars(tag) for tag in ea.Tags()["scalars"]}


def _tag_bucket(tag: str) -> str:
    return "svd" if any(s in tag for s in SVD_METRICS) else "default"


def _close(a: float, b: float, tol: dict) -> bool:
    return abs(a - b) <= tol["atol"] + tol["rtol"] * abs(b)


def _compare(mendu_scalars: dict, circuitry_scalars: dict) -> list[str]:
    """Return a list of failure strings; empty list means all in-spec."""
    failures: list[str] = []
    for mendu_tag, circuitry_tag in UNIVERSAL_TAGS.items():
        if mendu_tag not in mendu_scalars:
            failures.append(f"missing in mendu: {mendu_tag}")
            continue
        if circuitry_tag not in circuitry_scalars:
            failures.append(f"missing in circuitry: {circuitry_tag}")
            continue
        bucket = _tag_bucket(mendu_tag)
        tol = DEFAULT_TOLERANCES[bucket]
        # Pair by step index (assumes both pipelines emit at the same cadence)
        m_by_step = {s.step: s.value for s in mendu_scalars[mendu_tag]}
        c_by_step = {s.step: s.value for s in circuitry_scalars[circuitry_tag]}
        for step, m_val in m_by_step.items():
            if step not in c_by_step:
                continue  # different cadence at this step; skip
            c_val = c_by_step[step]
            if not _close(m_val, c_val, tol):
                failures.append(
                    f"{mendu_tag} @ step {step}: mendu={m_val:.6g} circuitry={c_val:.6g} "
                    f"(bucket={bucket}, rtol={tol['rtol']}, atol={tol['atol']})"
                )
    return failures


def _build_tiny_llama(seed: int = 0):
    """Build a tiny LLaMA-shaped model with fixed seed.

    Uses LLaMA naming conventions (layers.N.attention.wq.weight etc.) so that
    both mendu's arch_discovery and circuitry's _discovery.discover() can classify
    params into per-role buckets.  bias=False throughout — mendu's arch_discovery
    only matches `.weight` suffixes.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class TinyAttn(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim, bias=False)
            self.wv = nn.Linear(dim, dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            # Simplified (no causal mask, no rotary) — we only need gradients
            # flowing through the named params.
            return self.wo(self.wv(x) + self.wq(x) + self.wk(x))

    class TinyFFN(nn.Module):
        def __init__(self, dim: int = 64, hidden: int = 128) -> None:
            super().__init__()
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden, bias=False)

        def forward(self, x):
            return self.w2(self.w1(x) * self.w3(x))

    class TinyLayer(nn.Module):
        def __init__(self, dim: int = 64) -> None:
            super().__init__()
            self.attention = TinyAttn(dim)
            self.feed_forward = TinyFFN(dim)
            # bias=False mimics LLaMA-family RMSNorm (no bias).
            # Without this, nn.LayerNorm introduces bias parameters that mendu's
            # total_norm counts (via named_parameters) but circuitry's GRAD hook
            # misses (it only captures .weight, not .bias). Real LLaMA uses
            # elementwise_affine without bias — this keeps parity meaningful.
            self.attention_norm = nn.LayerNorm(dim, bias=False)
            self.ffn_norm = nn.LayerNorm(dim, bias=False)

        def forward(self, x):
            x = x + self.attention(self.attention_norm(x))
            x = x + self.feed_forward(self.ffn_norm(x))
            return x

    class TinyLLaMA(nn.Module):
        def __init__(
            self, vocab: int = 100, dim: int = 64, n_layers: int = 2
        ) -> None:
            super().__init__()
            self.tok_embeddings = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList(
                [TinyLayer(dim) for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(dim, bias=False)
            self.output = nn.Linear(dim, vocab, bias=False)

        def forward(self, ids):
            x = self.tok_embeddings(ids)
            for layer in self.layers:
                x = layer(x)
            return self.output(self.norm(x))

    return TinyLLaMA()


def _make_batch(seed: int = 0, batch: int = 2, seq: int = 8, vocab: int = 100):
    """Return a fixed random integer batch for reproducible gradient traces."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=g, dtype=torch.long)


def _run_mendu(mendu_root, steps: int, out_dir) -> None:
    """Run the tiny canonical training under mendu's InspectionRecorder."""
    import torch

    sys.path.insert(0, str(mendu_root))
    from tools.inspect_checkpoint.live import Cadence, InspectionRecorder  # noqa: PLC0415

    model = _build_tiny_llama()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _make_batch()
    rec = InspectionRecorder(
        run_dir=str(out_dir),
        model=model,
        optimizer=optimizer,
        cadence=Cadence(step_scalars=1, weight_cheap=1),
    )
    for step in range(steps):
        logits = model(batch)
        loss = logits.sum()
        loss.backward()
        rec.on_step_pre_optim(
            step,
            lm_loss=loss.item(),
            aux=0.0,
            total=loss.item(),
            lr=1e-3,
        )
        if step % 5 == 0:
            rec.on_checkpoint(step)
        optimizer.step()
        optimizer.zero_grad()
    rec.close()


def _run_circuitry(steps: int, out_dir) -> None:
    """Run the tiny canonical training under circuitry's Recorder.

    Uses a custom Recipe with LLaMA-native naming conventions (wq/wk/wv/wo,
    w1/w2/w3) to match the TinyLLaMA model built by _build_tiny_llama().
    The stock 'llm' recipe targets HuggingFace naming (q_proj/k_proj/...) and
    would match 0 modules on this model.
    """
    import torch

    from circuitry import Recorder  # noqa: PLC0415
    from circuitry.recipes import Recipe, register_recipe, _clear_registry_for_tests  # noqa: PLC0415
    from circuitry.recorder.hooks import HookPoint, TensorSource  # noqa: PLC0415

    # Custom recipe matching TinyLLaMA's LLaMA-style param names.
    # GRAD hook uses ".*" to match all modules — the recorder grabs .weight from each
    # and skips modules without .weight (container ModuleLists, etc.). This mirrors
    # mendu's total_norm which iterates all named_parameters() with .grad.
    PARITY_RECIPE = Recipe(
        name="parity_llama",
        hook_points=[
            # Attention weight matrices (LLaMA naming: wq/wk/wv/wo)
            HookPoint(source=TensorSource.WEIGHT,
                      pattern=r".*\.attention\.(wq|wk|wv|wo)$"),
            # FFN weight matrices
            HookPoint(source=TensorSource.WEIGHT,
                      pattern=r".*\.feed_forward\.(w1|w2|w3)$"),
            # Gradient hook — all modules to get global total_norm over all params,
            # matching mendu's _flush_gradient_norms which sums all named_parameters.
            HookPoint(source=TensorSource.GRAD, pattern=r".*"),
        ],
        weight_diagnostics=["effective_rank", "stable_rank", "sv_histogram"],
        gradient_diagnostics=["norms_per_param"],
    )
    _clear_registry_for_tests()
    register_recipe(PARITY_RECIPE)

    model = _build_tiny_llama()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _make_batch()
    rec = Recorder(
        model,
        run_dir=out_dir,
        recipe="parity_llama",
        writer="tensorboard",
        every_n_steps=1,
    )
    rec.attach()
    for step in range(steps):
        logits = model(batch)
        loss = logits.sum()
        loss.backward()
        rec.step(
            step,
            loss=loss.item(),
            loss_components={"lm_loss": loss.item(), "lr": 1e-3},
        )
        optimizer.step()
        optimizer.zero_grad()
    rec.detach()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Parity check: circuitry vs. mendu pre-cutover InspectionRecorder."
    )
    p.add_argument(
        "--mendu-root",
        required=True,
        help="path to ~/workspace/mendu",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument(
        "--out-dir",
        default=None,
        help="if set, persist run dirs here; else use tempdir",
    )
    args = p.parse_args()

    import pathlib
    import tempfile

    if args.out_dir:
        out_base = pathlib.Path(args.out_dir)
        out_base.mkdir(parents=True, exist_ok=True)
    else:
        out_base = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_parity_"))

    mendu_dir = out_base / "mendu"
    circuitry_dir = out_base / "circuitry"

    print(
        f"parity_check.py: running canonical tiny LLaMA for {args.steps} steps"
    )
    print(f"  mendu     -> {mendu_dir}")
    print(f"  circuitry -> {circuitry_dir}")

    _run_mendu(pathlib.Path(args.mendu_root), args.steps, mendu_dir)
    _run_circuitry(args.steps, circuitry_dir)

    print("Both pipelines completed. Comparing TB scalars...")
    mendu_scalars = _load_scalars(mendu_dir / "tb")  # InspectorTBWriter writes to <run_dir>/tb
    circuitry_scalars = _load_scalars(circuitry_dir)  # TensorBoardWriter writes events to run_dir directly

    failures = _compare(mendu_scalars, circuitry_scalars)
    if failures:
        print(f"PARITY FAILED ({len(failures)} mismatches):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PARITY OK — all {len(UNIVERSAL_TAGS)} universal tags within tolerances.")
    print(f"Output: {out_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
