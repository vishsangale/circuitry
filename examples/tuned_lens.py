# examples/tuned_lens.py
"""Runnable tuned-lens example (v1.10). ``python examples/tuned_lens.py``.

Fits a tuned lens (Belrose et al. 2023) on a tiny decoder, then attaches a
Recorder with the opt-in ``tuned_lens_kl`` diagnostic and shows that the tuned
lens reports a *lower* per-layer KL than the parameter-free logit lens — the
tuned affine absorbs the early/mid-layer basis mismatch.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from circuitry import HookPoint, Recipe, Recorder, TensorSource  # noqa: E402
from circuitry.core.lens import logit_lens_kl, tuned_lens_kl  # noqa: E402
from circuitry.tuned_lens import fit_tuned_lens  # noqa: E402
from circuitry.writers.base import RecordingWriter  # noqa: E402

D_MODEL, VOCAB = 16, 32


class Block(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(D_MODEL, D_MODEL, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(0.3 * torch.randn(D_MODEL, D_MODEL, generator=g))

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class TinyLM(nn.Module):
    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(i) for i in range(n_layers)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return self.lm_head(self.norm(x))


def main() -> None:
    torch.manual_seed(0)
    model = TinyLM(n_layers=4)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randn(4, 8, D_MODEL, generator=g) for _ in range(6)]

    # 1) Fit the tuned lens post-hoc (the only optimizer loop; outside training).
    lens = fit_tuned_lens(model, batches, steps=300, lr=5e-2)
    print(f"fitted tuned lens for layers {lens.layers} (d_model={lens.d_model})")
    # Persist + reload to show the serialization round-trip.
    lens.save(REPO / "tuned_lens.pt")

    # 2) Compare tuned vs logit lens directly on a held-out batch.
    held = torch.randn(2, 8, D_MODEL, generator=torch.Generator().manual_seed(7))
    res: dict[int, torch.Tensor] = {}
    handles = [
        blk.register_forward_hook(
            lambda _m, _i, out, idx=idx: res.__setitem__(idx, out))
        for idx, blk in enumerate(model.layers)
    ]
    model.eval()
    with torch.no_grad():
        model(held)
    for h in handles:
        h.remove()
    W = model.get_output_embeddings().weight.detach()
    final_logits = model.norm(res[3]) @ W.t()
    print("\nper-layer KL to final distribution (lower = prediction more formed):")
    print(f"{'layer':>5} | {'logit_lens':>11} | {'tuned_lens':>11}")
    for layer in lens.layers:
        A, b = lens.translator_for(layer)
        llk = logit_lens_kl(res[layer], W, final_logits, layer_norm=model.norm)
        tlk = tuned_lens_kl(res[layer], (A, b), W, final_logits, layer_norm=model.norm)
        print(f"{layer:>5} | {llk:>11.4f} | {tlk:>11.4f}")

    # 3) Wire it into the Recorder as the opt-in tuned_lens_kl diagnostic.
    recipe = Recipe(
        name="tuned-lens-demo",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"layers\.\d+$")],
        activation_diagnostics=["logit_lens_kl", "tuned_lens_kl"],
        tuned_lens=lens,
    )
    writer = RecordingWriter()
    rec = Recorder(model, REPO / "_demo_run", dataclasses.replace(recipe),
                   writer=writer, every_n_steps=1, strict=False)
    rec.attach()
    model(held)
    rec.step(0)
    rec.detach()
    emitted = sorted(t for t, _v, _s in writer.scalars if "tuned_lens_kl" in t)
    print(f"\nRecorder emitted: {emitted}")


if __name__ == "__main__":
    main()
