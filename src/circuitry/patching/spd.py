"""Stochastic Parameter Decomposition (SPD) — parameter-space interpretability.

Decomposes a target ``nn.Linear``'s weight into ``C`` rank-one subcomponents
trained to be *faithful* (they sum to the original weight), *minimal* (few
are causally important per input), and *replaceable* (ablating unimportant
ones leaves the model's output unchanged).  Bushnaq et al., "Stochastic
Parameter Decomposition" (arXiv:2506.20790; reference implementation
github.com/goodfire-ai/spd), the scalable successor to Attribution-based
Parameter Decomposition (APD, arXiv:2501.14926).

Mechanics (mirroring the reference ``ComponentLinear``):

- Subcomponents: ``V (d_in, C)`` and ``U (C, d_out)``; the component forward
  is ``((x @ V) * mask) @ U`` and ``V @ U`` reconstructs ``W.T``.
- Causal importance ``ci(x) ∈ (0, 1)^C`` is predicted from the module input
  (here: a small MLP — the original APD/SPD formulation; the reference repo
  uses a CI-transformer at LM scale).
- Stochastic masks ``m = ci + (1 − ci) · Uniform(0, 1)`` (uniform in
  ``[ci, 1]``): components the CI model marks unimportant are randomly
  ablated, and the masked model must still match the target model's output.
- Losses: faithfulness ``‖W.T − V @ U‖²`` + stochastic reconstruction
  (output divergence under masking; MSE by default, ``output_loss="kl"``
  for logit outputs) + importance minimality (``mean(ci)``; the reference
  adds a log-scaled term — simplification documented).

No hidden device moves; trains wherever the target module's weight lives.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SPDResult", "SPDRunner"]


class _ImportanceMLP(nn.Module):
    """Small MLP: module input → per-subcomponent causal importance in (0, 1)."""

    def __init__(self, d_in: int, n_components: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_components),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))


class SPDResult:
    """Trained decomposition from :class:`SPDRunner`.

    Attributes:
        U: ``(C, d_out)`` output transforms (detached).
        V: ``(d_in, C)`` input basis (detached).
        faithfulness_error: relative reconstruction error
            ``‖W.T − V @ U‖_F / ‖W‖_F`` at the end of training.
        losses: per-step ``{"faith": [...], "stoch": [...], "imp": [...]}``.
    """

    def __init__(
        self,
        U: Tensor,
        V: Tensor,
        importance_model: nn.Module,
        faithfulness_error: float,
        losses: dict[str, list[float]],
    ) -> None:
        self.U = U
        self.V = V
        self._importance_model = importance_model
        self.faithfulness_error = faithfulness_error
        self.losses = losses

    @property
    def n_components(self) -> int:
        return self.U.shape[0]

    def importance(self, x: Tensor) -> Tensor:
        """Causal importance ``ci(x) ∈ (0, 1)^C`` for module inputs *x* ``(..., d_in)``."""
        with torch.no_grad():
            return self._importance_model(x)

    def active_components(self, x: Tensor, *, threshold: float = 0.5) -> list[int]:
        """Components whose mean importance over *x* exceeds *threshold*."""
        mean_ci = self.importance(x).reshape(-1, self.n_components).mean(dim=0)
        return [c for c in range(self.n_components) if float(mean_ci[c]) > threshold]

    def component_weight(self, c: int) -> Tensor:
        """Rank-one weight of component *c* in the Linear's ``(d_out, d_in)`` orientation."""
        return torch.outer(self.U[c], self.V[:, c])

    def reconstructed_weight(self) -> Tensor:
        """``(d_out, d_in)`` sum of all components (≈ the original weight)."""
        return (self.V @ self.U).T

    def to_markdown(self, *, x: Tensor | None = None, top_k: int = 10) -> str:
        lines = ["## SPD Decomposition", ""]
        lines.append(f"- Components: {self.n_components}")
        lines.append(f"- Faithfulness error (relative): {self.faithfulness_error:.4g}")
        norms = [float(self.component_weight(c).norm()) for c in range(self.n_components)]
        ranked = sorted(range(self.n_components), key=lambda c: norms[c], reverse=True)
        header = "| component | ‖W_c‖_F |"
        sep = "| ---: | ---: |"
        mean_ci = None
        if x is not None:
            mean_ci = self.importance(x).reshape(-1, self.n_components).mean(dim=0)
            header += " mean importance |"
            sep += " ---: |"
        lines.append("")
        lines.append(f"### Top-{min(top_k, self.n_components)} Components by Norm")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for c in ranked[:top_k]:
            row = f"| {c} | {norms[c]:.4g} |"
            if mean_ci is not None:
                row += f" {float(mean_ci[c]):.4f} |"
            lines.append(row)
        return "\n".join(lines)


class SPDRunner:
    """Decompose one ``nn.Linear`` inside *model* into rank-one subcomponents.

    Hook-based: during training, a forward hook replaces the target module's
    output with the masked component forward (plus the original bias); the
    rest of the model is untouched and frozen.  No model surgery; the hook
    is removed when ``run()`` returns.

    Args:
        model: the model containing the target module.
        module: the ``nn.Linear`` to decompose.
        n_components: number of rank-one subcomponents ``C``.
        importance_hidden: hidden width of the causal-importance MLP.
        seed: seeds subcomponent init and mask sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        module: nn.Linear,
        *,
        n_components: int,
        importance_hidden: int = 64,
        seed: int = 0,
    ) -> None:
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"SPDRunner decomposes nn.Linear modules, got {type(module).__name__}"
            )
        self._model = model
        self._module = module
        self._C = n_components
        self._hidden = importance_hidden
        self._seed = seed

    def run(
        self,
        batches: Iterable[Any],
        *,
        n_steps: int = 500,
        lr: float = 1e-3,
        coeff_faith: float = 1.0,
        coeff_stoch: float = 1.0,
        coeff_imp: float = 1e-3,
        output_loss: str = "mse",
        forward_fn: Callable[[nn.Module, Any], Tensor] | None = None,
    ) -> SPDResult:
        """Train the decomposition.

        Args:
            batches: iterable of model inputs, cycled for *n_steps* (each
                element is passed to ``model(batch)`` or *forward_fn*).
            n_steps: optimizer iterations.
            lr: Adam learning rate.
            coeff_faith / coeff_stoch / coeff_imp: loss weights for
                faithfulness, stochastic reconstruction, and importance
                minimality.
            output_loss: ``"mse"`` (default; any output shape) or ``"kl"``
                (softmax-KL over the last dim — use for logit outputs).
            forward_fn: optional ``(model, batch) → output`` entry point for
                non-standard forwards.

        Returns:
            :class:`SPDResult`.
        """
        if output_loss not in ("mse", "kl"):
            raise ValueError(f"output_loss must be 'mse' or 'kl', got {output_loss!r}")
        batch_list = list(batches)
        if not batch_list:
            raise ValueError("batches is empty")

        W = self._module.weight  # (d_out, d_in)
        d_out, d_in = W.shape
        device, dtype = W.device, W.dtype
        gen = torch.Generator(device="cpu").manual_seed(self._seed)
        torch.manual_seed(self._seed)

        # Init so that V @ U starts near W.T / C-scaled randomness.
        V = nn.Parameter(torch.randn(d_in, self._C, device=device, dtype=dtype) / d_in**0.5)
        U = nn.Parameter(torch.randn(self._C, d_out, device=device, dtype=dtype) / self._C**0.5)
        ci_model = _ImportanceMLP(d_in, self._C, self._hidden).to(device=device, dtype=dtype)

        frozen = [(p, p.requires_grad) for p in self._model.parameters()]
        for p, _ in frozen:
            p.requires_grad_(False)

        # The hook computes the masked component forward from the module
        # input; the per-forward mask is staged in `state` by the train loop.
        state: dict[str, Tensor | None] = {"mask": None}

        def _component_hook(mod: nn.Module, inputs: tuple, output: Tensor) -> Tensor:
            mask = state["mask"]
            if mask is None:
                return output
            x = inputs[0]
            comp_acts = x @ V                       # (..., C)
            out = (comp_acts * mask) @ U            # (..., d_out)
            if mod.bias is not None:
                out = out + mod.bias
            return out

        handle = self._module.register_forward_hook(_component_hook)
        was_training = self._model.training
        opt = torch.optim.Adam([V, U, *ci_model.parameters()], lr=lr)
        losses: dict[str, list[float]] = {"faith": [], "stoch": [], "imp": []}

        # Capture the module input on the target pass so the CI model sees
        # the same activations the masked pass will.
        cap: dict[str, Tensor] = {}

        def _capture_hook(mod: nn.Module, inputs: tuple, output: Tensor) -> None:
            cap["x"] = inputs[0].detach()

        try:
            self._model.eval()
            for step in range(n_steps):
                batch = batch_list[step % len(batch_list)]

                # Target pass (hook inert: mask=None) — capture x and target out.
                state["mask"] = None
                cap_handle = self._module.register_forward_hook(_capture_hook)
                try:
                    with torch.no_grad():
                        target_out = (
                            forward_fn(self._model, batch) if forward_fn is not None
                            else self._model(batch)
                        )
                finally:
                    cap_handle.remove()
                x = cap["x"]

                # Stochastic mask: m = ci + (1 - ci) * U(0, 1)  — uniform in [ci, 1].
                ci = ci_model(x)                                        # (..., C)
                u = torch.rand(ci.shape, generator=gen).to(device=device, dtype=dtype)
                state["mask"] = ci + (1.0 - ci) * u

                masked_out = (
                    forward_fn(self._model, batch) if forward_fn is not None
                    else self._model(batch)
                )

                loss_faith = (W.T - V @ U).pow(2).mean()
                if output_loss == "kl":
                    loss_stoch = F.kl_div(
                        torch.log_softmax(masked_out, dim=-1),
                        torch.log_softmax(target_out, dim=-1),
                        log_target=True, reduction="batchmean",
                    )
                else:
                    loss_stoch = (masked_out - target_out).pow(2).mean()
                loss_imp = ci.mean()

                total = (
                    coeff_faith * loss_faith
                    + coeff_stoch * loss_stoch
                    + coeff_imp * loss_imp
                )
                opt.zero_grad()
                total.backward()
                opt.step()

                losses["faith"].append(float(loss_faith.item()))
                losses["stoch"].append(float(loss_stoch.item()))
                losses["imp"].append(float(loss_imp.item()))
        finally:
            handle.remove()
            state["mask"] = None
            for p, req in frozen:
                p.requires_grad_(req)
            self._model.train(was_training)

        with torch.no_grad():
            rel_err = float(
                (W.T - V @ U).norm() / W.norm().clamp_min(1e-12)
            )
        ci_model.eval()
        for p in ci_model.parameters():
            p.requires_grad_(False)
        return SPDResult(
            U=U.detach(),
            V=V.detach(),
            importance_model=ci_model,
            faithfulness_error=rel_err,
            losses=losses,
        )
