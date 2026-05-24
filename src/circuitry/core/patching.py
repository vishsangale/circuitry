"""Pure metrics for activation patching. See design spec §5.

Tensor→float functions. No model execution, no I/O, no .cuda().
"""
from __future__ import annotations

import torch
from torch import Tensor


def logit_diff(logits: Tensor, correct: int, incorrect: int) -> float:
    """Difference in logits at correct vs incorrect token.

    Accepts (vocab,), (batch, vocab), or (batch, seq, vocab). For 3-D input,
    uses the last sequence position.
    """
    x = logits.detach().float()
    if x.ndim == 3:
        x = x[:, -1, :]
    if x.ndim == 1:
        return float((x[correct] - x[incorrect]).item())
    return float((x[:, correct] - x[:, incorrect]).mean().item())


def kl_divergence(
    p_logits: Tensor,
    q_logits: Tensor,
    *,
    chunk_size: int = 256,
) -> float:
    """KL(softmax(p) || softmax(q)), mean over leading dims. Chunked."""
    p = p_logits.detach().to(torch.float32)
    q = q_logits.detach().to(torch.float32)
    if p.ndim == 1:
        p = p.unsqueeze(0)
        q = q.unsqueeze(0)
    if p.ndim == 3:
        p = p.reshape(-1, p.shape[-1])
        q = q.reshape(-1, q.shape[-1])
    n = p.shape[0]
    if n == 0:
        return 0.0
    kl_sum = p.new_zeros(())
    for start in range(0, n, max(1, chunk_size)):
        pc = p[start : start + chunk_size]
        qc = q[start : start + chunk_size]
        log_p = torch.log_softmax(pc, dim=-1)
        log_q = torch.log_softmax(qc, dim=-1)
        kl_sum = kl_sum + (log_p.exp() * (log_p - log_q)).sum()
    return float((kl_sum / n).item())


def ce_loss(logits: Tensor, targets: Tensor) -> float:
    """Cross-entropy loss, mean over batch.

    For 3-D logits (batch, seq, vocab), uses the last sequence position.
    """
    x = logits.detach().float()
    if x.ndim == 3:
        x = x[:, -1, :]
    return float(torch.nn.functional.cross_entropy(x, targets).item())


# ---------------------------------------------------------------------------
# Differentiable (_t) variants — same math, NO .detach(), return grad-carrying
# scalar Tensor.  Used by EAP where gradients must flow back through the metric.
# ---------------------------------------------------------------------------


def logit_diff_t(logits: Tensor, correct: int, incorrect: int) -> Tensor:
    """Differentiable logit-diff: returns a scalar Tensor (no .detach())."""
    x = logits.float()
    if x.ndim == 3:
        x = x[:, -1, :]
    if x.ndim == 1:
        return x[correct] - x[incorrect]
    return (x[:, correct] - x[:, incorrect]).mean()


def kl_divergence_t(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """Differentiable KL(softmax(p) || softmax(q)), mean over tokens.

    No chunking (EAP batches are small) and no .detach().
    """
    p = p_logits.float()
    q = q_logits.float()
    if p.ndim == 1:
        p = p.unsqueeze(0)
        q = q.unsqueeze(0)
    if p.ndim == 3:
        p = p.reshape(-1, p.shape[-1])
        q = q.reshape(-1, q.shape[-1])
    n = p.shape[0]
    log_p = torch.log_softmax(p, dim=-1)
    log_q = torch.log_softmax(q, dim=-1)
    kl_sum = (log_p.exp() * (log_p - log_q)).sum()
    return kl_sum / max(n, 1)


def ce_loss_t(logits: Tensor, targets: Tensor) -> Tensor:
    """Differentiable cross-entropy loss (no .detach()), scalar Tensor."""
    x = logits.float()
    if x.ndim == 3:
        x = x[:, -1, :]
    return torch.nn.functional.cross_entropy(x, targets)
