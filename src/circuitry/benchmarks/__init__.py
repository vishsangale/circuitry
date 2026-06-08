"""circuitry.benchmarks — standardised benchmark task loaders and metric runners.

Provides:
- MIB task loaders (Mueller et al. ICML 2025, arxiv:2504.13151)
- SAEBench metric runner (Karvonen et al. 2025, arxiv:2503.09532)

All synthetic; no network access or dataset downloads required.
"""
from circuitry.benchmarks.mib import (
    MIBTask,
    load_ioi,
    load_greater_than,
)

from circuitry.benchmarks.saebench import (
    SAEBenchResult,
    l0_sparsity,
    explained_variance,
    reconstruction_mse,
    feature_density,
    sparse_probing_r2,
    run_saebench,
)

__all__ = [
    "MIBTask",
    "load_ioi",
    "load_greater_than",
    "SAEBenchResult",
    "l0_sparsity",
    "explained_variance",
    "reconstruction_mse",
    "feature_density",
    "sparse_probing_r2",
    "run_saebench",
]
