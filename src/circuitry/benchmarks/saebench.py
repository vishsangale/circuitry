"""SAEBench metric runner.

Karvonen et al. 2025. https://arxiv.org/abs/2503.09532
Implements analytically tractable SAEBench metrics on raw activation tensors.

All metrics work offline on CPU tensors — no dataset download or network
access required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from circuitry.sae.grad import sae_decompose


@dataclass
class SAEBenchResult:
    """Result container for SAEBench metric evaluation.

    Attributes:
        l0:                  Mean number of active features per token (L0 norm).
        explained_variance:  Fraction of activation variance explained by the
                             SAE reconstruction.  Clipped to ``[0, 1]``.
        mse:                 Mean squared reconstruction error.
        feature_density:     Fraction of features that activate (> 0) on at
                             least one example in the batch.
        sparse_probing_r2:   R² of a linear regression from SAE feature
                             activations to the original activations.  Proxy
                             for how well SAE features span the activation space.
        ce_loss_score:       ``(CE_clean - CE_sae) / CE_clean`` — requires a
                             language model; ``None`` when not computed.
    """

    l0: float
    explained_variance: float
    mse: float
    feature_density: float
    sparse_probing_r2: float
    ce_loss_score: float | None = None

    def summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        lines = ["SAEBench Metrics:"]
        lines.append(f"  L0 sparsity:        {self.l0:.2f}")
        lines.append(f"  Explained variance: {self.explained_variance:.4f}")
        lines.append(f"  MSE:                {self.mse:.6f}")
        lines.append(f"  Feature density:    {self.feature_density:.4f}")
        lines.append(f"  Sparse probing R²:  {self.sparse_probing_r2:.4f}")
        if self.ce_loss_score is not None:
            lines.append(f"  CE loss score:      {self.ce_loss_score:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def l0_sparsity(sae: Any, acts: Tensor) -> float:
    """Mean number of active features per token (L0 norm of feature activations).

    Args:
        sae:  SAE object with ``.encode(x)`` method.
        acts: ``(n, d_model)`` activation tensor.

    Returns:
        Mean L0 over the batch as a Python float.
    """
    f = sae.encode(acts)
    return (f > 0).float().sum(-1).mean().item()


def explained_variance(sae: Any, acts: Tensor) -> float:
    """Fraction of activation variance explained by the SAE reconstruction.

    ``EV = 1 - Var(x - x_hat) / Var(x)``.  Clipped to ``[0, 1]``.

    Args:
        sae:  SAE object with ``.encode(x)`` and ``.decode(f)`` methods.
        acts: ``(n, d_model)`` activation tensor.

    Returns:
        Explained variance in ``[0, 1]`` as a Python float.
    """
    with torch.no_grad():
        _f, x_hat, _eps = sae_decompose(sae, acts)
    residual = acts - x_hat
    acts_var = acts.var().item()
    if acts_var == 0.0:
        return 1.0
    ev = 1.0 - residual.var().item() / acts_var
    return float(max(0.0, min(1.0, ev)))


def reconstruction_mse(sae: Any, acts: Tensor) -> float:
    """Mean squared error of the SAE reconstruction: ``mean ||x - x_hat||²``.

    Args:
        sae:  SAE object with ``.encode(x)`` and ``.decode(f)`` methods.
        acts: ``(n, d_model)`` activation tensor.

    Returns:
        MSE as a Python float.
    """
    with torch.no_grad():
        _f, x_hat, _eps = sae_decompose(sae, acts)
    return (acts - x_hat).pow(2).mean().item()


def feature_density(sae: Any, acts: Tensor) -> float:
    """Fraction of features that activate (> 0) on at least one example.

    Args:
        sae:  SAE object with ``.encode(x)`` method.
        acts: ``(n, d_model)`` activation tensor.

    Returns:
        Feature density in ``[0, 1]`` as a Python float.
    """
    with torch.no_grad():
        f = sae.encode(acts)
    return (f > 0).any(0).float().mean().item()


def sparse_probing_r2(sae: Any, acts: Tensor) -> float:
    """R² of a linear regression from SAE feature activations to original activations.

    Fits ``torch.linalg.lstsq(f, acts)`` on the batch and computes R² on the
    same data (training R²).  This is a proxy for how well the SAE feature
    space linearly reconstructs the activation space.

    Args:
        sae:  SAE object with ``.encode(x)`` method.
        acts: ``(n, d_model)`` activation tensor.

    Returns:
        R² in ``[0, 1]`` as a Python float (clipped to 0 from below).
    """
    with torch.no_grad():
        f = sae.encode(acts)

    # Use float32 for numerical stability in lstsq
    f_f32   = f.float()
    acts_f32 = acts.float()

    result   = torch.linalg.lstsq(f_f32, acts_f32, driver="gelsd")
    solution = result.solution  # (d_hidden, d_model)

    acts_pred = f_f32 @ solution  # (n, d_model)
    ss_res = (acts_f32 - acts_pred).pow(2).sum().item()
    ss_tot = (acts_f32 - acts_f32.mean(0, keepdim=True)).pow(2).sum().item()

    if ss_tot == 0.0:
        return 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, min(1.0, r2)))


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

_ALL_TASKS = frozenset({"l0", "explained_variance", "mse", "feature_density", "sparse_probing_r2"})


def run_saebench(
    sae: Any,
    acts: Tensor,
    *,
    tasks: list[str] | None = None,
) -> SAEBenchResult:
    """Run SAEBench metrics on activation tensors.

    Args:
        sae:   SAE object with ``.encode(x)`` and ``.decode(f)`` methods.
        acts:  ``(n, d_model)`` activation tensor.
        tasks: Subset of metric names to run; ``None`` runs all.
                Valid names: ``"l0"``, ``"explained_variance"``, ``"mse"``,
                ``"feature_density"``, ``"sparse_probing_r2"``.

    Returns:
        :class:`SAEBenchResult` with computed metrics (uncomputed fields set
        to ``0.0`` and ``ce_loss_score`` set to ``None``).

    Raises:
        ValueError: If an unknown task name is supplied.
    """
    if tasks is None:
        tasks_set = _ALL_TASKS
    else:
        unknown = set(tasks) - _ALL_TASKS
        if unknown:
            raise ValueError(
                f"Unknown SAEBench task(s): {sorted(unknown)}. "
                f"Valid tasks: {sorted(_ALL_TASKS)}"
            )
        tasks_set = set(tasks)

    _l0   = l0_sparsity(sae, acts)          if "l0"                 in tasks_set else 0.0
    _ev   = explained_variance(sae, acts)   if "explained_variance" in tasks_set else 0.0
    _mse  = reconstruction_mse(sae, acts)   if "mse"                in tasks_set else 0.0
    _fd   = feature_density(sae, acts)      if "feature_density"    in tasks_set else 0.0
    _r2   = sparse_probing_r2(sae, acts)    if "sparse_probing_r2"  in tasks_set else 0.0

    return SAEBenchResult(
        l0=_l0,
        explained_variance=_ev,
        mse=_mse,
        feature_density=_fd,
        sparse_probing_r2=_r2,
        ce_loss_score=None,
    )
