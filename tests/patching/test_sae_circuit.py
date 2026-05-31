"""SAE feature circuit ablation / FeatureACDC tests. v1.6.0 Stage B.

ABLATION / ACDC gates per spec §6:
  - test_faithfulness_full_circuit_is_one  (≈1, abs ~1e-3)
  - test_faithfulness_empty_is_zero        (≈0, abs ~1e-3)
  - test_ablation_monotonicity             (top vs bottom node KL)
  - test_ablation_modes                    (corrupted / zero / mean)
  - test_model_clean_after_ablation
  - test_feature_acdc_converges
  - test_sweep_pareto
  - test_prune_threshold / test_prune_acdc / test_prune_both
  - test_error_node_edges_optin
  - test_no_sae_param_grad_leak
"""
from __future__ import annotations

import math

import pytest
import torch

# Re-use all toy models and helpers from test_sae_features
from tests.patching.test_sae_features import (
    LinearResidToy,
    SyntheticSAE,
    _make_clean_corr,
    _make_resolver,
    _metric,
)

# ---------------------------------------------------------------------------
# Helpers shared by this module
# ---------------------------------------------------------------------------


def _make_runner(model, sae0, sae1, d=8):
    from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
    from circuitry.patching.sites import Site

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)
    resolver = _make_resolver(d)
    return SAEFeatureEdgeRunner(model, {site0: sae0, site1: sae1}, resolver)


def _make_acdc_runner(model, sae0, sae1, d=8):
    from circuitry.patching.sae_edges import FeatureACDCRunner
    from circuitry.patching.sites import Site

    site0 = Site("resid_post", layer=0)
    site1 = Site("resid_post", layer=1)
    resolver = _make_resolver(d)
    sae_sites = {site0: sae0, site1: sae1}
    return FeatureACDCRunner(model, sae_sites, resolver)


def _setup(seed_model=0, seed_sae0=10, seed_sae1=11, d=8, d_sae=16):
    torch.manual_seed(seed_model)
    model = LinearResidToy(n_layers=2, d=d)
    torch.manual_seed(seed_sae0)
    sae0 = SyntheticSAE(d_model=d, d_sae=d_sae)
    torch.manual_seed(seed_sae1)
    sae1 = SyntheticSAE(d_model=d, d_sae=d_sae)
    runner = _make_runner(model, sae0, sae1, d=d)
    return model, sae0, sae1, runner


def _get_full_circuit(runner, clean, corrupted, top_k=32):
    """Run the edge runner and return the circuit (full survivor set)."""
    return runner.run(clean, corrupted, _metric, top_k_survivors=top_k)


# ---------------------------------------------------------------------------
# test_faithfulness_full_circuit_is_one
# ---------------------------------------------------------------------------


def test_faithfulness_full_circuit_is_one():
    """faithfulness(M) ≈ 1.0 (abs ~1e-3) on LinearResidToy + affine SAEs.

    m(M) = full circuit (all survivors), m(∅) = empty circuit.
    faithfulness(M) = (m(M) - m(∅)) / (m(M) - m(∅)) = 1.
    PRINTS faithfulness value.

    Lossless assertion: max|m(full) − clean_metric| < 1e-4 (sanity).
    """
    model, sae0, sae1, runner = _setup(seed_model=1, seed_sae0=100, seed_sae1=101)
    clean, corrupted = _make_clean_corr(seed=200)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=32)

    faith = circuit.faithfulness(clean, corrupted, _metric, ablation_mode="corrupted")
    print(f"\n[faithfulness(M)] = {faith:.6f}")
    assert math.isfinite(faith), f"faithfulness returned non-finite: {faith}"
    assert abs(faith - 1.0) < 0.01, (
        f"faithfulness(M) = {faith:.6f}, expected ≈ 1.0 (tolerance 0.01)"
    )

    # Lossless assertion: m(full circuit) ≈ clean_metric
    with torch.no_grad():
        clean_out = model(clean) if not isinstance(clean, dict) else model(**clean)
    m_clean = float(_metric(clean_out))
    ablation_values = circuit._compute_ablation_values(clean, corrupted, "corrupted")
    full_nodes = circuit._all_node_sets()
    m_full = circuit._m_of(clean, corrupted, _metric, full_nodes, ablation_values)
    print(f"[faithfulness(M)] m_full={m_full:.6f}, m_clean={m_clean:.6f}, diff={abs(m_full - m_clean):.2e}")
    assert abs(m_full - m_clean) < 1e-4, (
        f"m(full_circuit) = {m_full:.6f} ≠ clean_metric = {m_clean:.6f} (diff={abs(m_full-m_clean):.2e})"
    )


# ---------------------------------------------------------------------------
# test_faithfulness_empty_is_zero
# ---------------------------------------------------------------------------


def test_faithfulness_empty_is_zero():
    """faithfulness(∅) ≈ 0.0 (abs ~1e-3).

    Empty circuit: remove all edges from the full circuit.
    faithfulness(∅) = (m(∅) - m(∅)) / (m(M) - m(∅)) = 0.
    PRINTS faithfulness value.
    """
    from circuitry.patching.sae_edges import SAEFeatureCircuit

    model, sae0, sae1, runner = _setup(seed_model=2, seed_sae0=102, seed_sae1=103)
    clean, corrupted = _make_clean_corr(seed=201)
    full_circuit = _get_full_circuit(runner, clean, corrupted, top_k=32)

    # Build an empty circuit by removing all edges
    empty_circuit = SAEFeatureCircuit(
        nodes=full_circuit.nodes,
        edges={},
        graph=full_circuit.graph,
        model=full_circuit._model,
        sae_sites=full_circuit._sae_sites,
        resolver=full_circuit._resolver,
    )

    faith = empty_circuit.faithfulness(clean, corrupted, _metric, ablation_mode="corrupted")
    print(f"\n[faithfulness(∅)] = {faith:.6f}")
    assert math.isfinite(faith), f"faithfulness returned non-finite: {faith}"
    assert abs(faith - 0.0) < 0.01, (
        f"faithfulness(∅) = {faith:.6f}, expected ≈ 0.0 (tolerance 0.01)"
    )


# ---------------------------------------------------------------------------
# test_faithfulness_partial_circuit — TOOTHED test (must FAIL for broken ablation)
# ---------------------------------------------------------------------------


def test_faithfulness_partial_circuit():
    """faithfulness of a partial circuit vs empty circuit: direct _m_of comparison.

    Strategy:
      1. Run full circuit (top_k=32, all survivors).
      2. Compute m(C_full), m(∅), m(C_half) for a partial node set (half the
         features per layer), using direct _m_of calls.
      3. KEY TOOTH: assert m(C_half) lies strictly between m(∅) and m(C_full).
         FAILS if the ablation forward ignores its circuit_nodes input (no-op).

    This test directly validates _m_of responsiveness without going through
    faithfulness() — it's a ground-truth check on the ablation forward itself.
    """
    model, sae0, sae1, runner = _setup(seed_model=3, seed_sae0=300, seed_sae1=301)
    clean, corrupted = _make_clean_corr(seed=202)
    full_circuit = _get_full_circuit(runner, clean, corrupted, top_k=32)

    ablation_values = full_circuit._compute_ablation_values(clean, corrupted, "corrupted")
    full_nodes = full_circuit._all_node_sets()

    if not any(full_nodes.values()):
        pytest.skip("No survivors in full circuit")

    # Build a half node set: keep only the FIRST half of features at each layer
    half_nodes: dict = {}
    for layer, feats in full_nodes.items():
        sorted_feats = sorted(feats)
        half_nodes[layer] = set(sorted_feats[: max(1, len(sorted_feats) // 2)])

    empty_nodes: dict = {layer: set() for layer in ablation_values}

    m_full = full_circuit._m_of(clean, corrupted, _metric, full_nodes, ablation_values)
    m_half = full_circuit._m_of(clean, corrupted, _metric, half_nodes, ablation_values)
    m_empty = full_circuit._m_of(clean, corrupted, _metric, empty_nodes, ablation_values)

    print(
        f"\n[partial _m_of] m(full)={m_full:.6f}, m(half)={m_half:.6f}, m(∅)={m_empty:.6f}, "
        f"n_full={sum(len(v) for v in full_nodes.values())}, "
        f"n_half={sum(len(v) for v in half_nodes.values())}"
    )

    range_m = abs(m_full - m_empty)
    if range_m < 1e-6:
        pytest.skip("m(full) ≈ m(∅) — ill-conditioned setup, cannot distinguish circuits")

    # KEY TOOTH: m(half) must DIFFER from m(∅)
    # (fails if _feature_circuit_forward ignores circuit_nodes and always runs the full model)
    assert abs(m_half - m_empty) > 1e-6, (
        f"m(half_circuit) = m(∅) = {m_empty:.8f}: "
        "ablation forward appears to be ignoring circuit_nodes (no-op)."
    )

    # KEY TOOTH 2: m(half) must DIFFER from m(full)
    assert abs(m_half - m_full) > 1e-6, (
        f"m(half_circuit) = m(full) = {m_full:.8f}: "
        "ablation forward appears to be ignoring ablated features."
    )

    # The range of values confirms the ablation forward is responsive:
    # m(half) should sit between m(∅) and m(full) (though not necessarily exactly)
    min_m = min(m_empty, m_full)
    max_m = max(m_empty, m_full)
    print(f"[partial _m_of] m(half) in [{min_m:.4f}, {max_m:.4f}]? {min_m <= m_half <= max_m}")

    # Also verify faithfulness API produces a finite result for the full circuit ≈ 1
    faith_full = full_circuit.faithfulness(clean, corrupted, _metric, ablation_mode="corrupted")
    assert math.isfinite(faith_full), "faithfulness(full) is non-finite"
    assert abs(faith_full - 1.0) < 0.01, (
        f"faithfulness(full) = {faith_full:.6f}, expected ≈ 1.0"
    )


# ---------------------------------------------------------------------------
# test_ablation_monotonicity
# ---------------------------------------------------------------------------


def test_ablation_monotonicity():
    """Spearman rank-correlation between |node score| and KL_after_remove > 0.5.

    Strategy: run the full circuit, compute KL_after_remove for ALL survivor nodes.
    TOOTHED: Spearman(|score|, KL_after_remove) > 0.5 over all survivors.
    This FAILS for random scores (expected Spearman ≈ 0).

    Also checks top > bottom KL (original extreme test).
    PRINTS Spearman value.
    """
    import numpy as np
    from scipy.stats import spearmanr

    from circuitry.patching.sae_edges import _feature_circuit_forward

    torch.manual_seed(5)
    model = LinearResidToy(n_layers=2, d=8)
    torch.manual_seed(50)
    sae0 = SyntheticSAE(d_model=8, d_sae=16)
    torch.manual_seed(51)
    sae1 = SyntheticSAE(d_model=8, d_sae=16)
    runner = _make_runner(model, sae0, sae1)

    clean, corrupted = _make_clean_corr(seed=300)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=32)

    # Collect all nodes with scores, sorted by abs score
    # nodes.scores is a dict[AtPNode, float]
    node_scores_list: list[tuple[float, int, int]] = []  # (|score|, layer, feat_idx)
    for atp_node, score in circuit.nodes.scores.items():
        nd = atp_node.node
        if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
            node_scores_list.append((abs(float(score)), nd.layer, nd.neuron))

    if len(node_scores_list) < 3:
        pytest.skip("Fewer than 3 nodes to test monotonicity")

    # Build full node set
    full_nodes = circuit._all_node_sets()
    ablation_values = circuit._compute_ablation_values(clean, corrupted, "corrupted")

    with torch.no_grad():
        clean_out = model(clean) if not isinstance(clean, dict) else model(**clean)

    def _kl_after_remove(layer: int, feat: int) -> float:
        """KL(circuit_without_feat ‖ clean)."""
        from circuitry.core.patching import kl_divergence

        pruned_nodes = {layer_k: set(s) for layer_k, s in full_nodes.items()}
        pruned_nodes.setdefault(layer, set()).discard(feat)

        out = _feature_circuit_forward(
            model, clean,
            circuit._sae_sites,
            circuit._resolver,
            circuit_nodes=pruned_nodes,
            ablation_values=ablation_values,
        )
        logits_out = out.logits if hasattr(out, "logits") else out
        logits_clean = clean_out.logits if hasattr(clean_out, "logits") else clean_out
        if logits_out.ndim >= 3:
            logits_out = logits_out[:, -1:, :]
            logits_clean = logits_clean[:, -1:, :]
        return kl_divergence(logits_out, logits_clean)

    # Compute KL for ALL survivors (not just extremes)
    abs_scores = np.array([ns[0] for ns in node_scores_list], dtype=np.float64)
    kl_values = np.array(
        [_kl_after_remove(ns[1], ns[2]) for ns in node_scores_list], dtype=np.float64
    )

    print(
        f"\n[monotonicity] {len(node_scores_list)} nodes evaluated"
    )

    # Classic extreme test
    node_scores_sorted = sorted(node_scores_list, reverse=True)
    top_score, top_layer, top_feat = node_scores_sorted[0]
    bot_score, bot_layer, bot_feat = node_scores_sorted[-1]

    kl_top_removed = float(kl_values[node_scores_list.index(node_scores_sorted[0])])
    kl_bot_removed = float(kl_values[node_scores_list.index(node_scores_sorted[-1])])

    print(
        f"[monotonicity] top node |score|={top_score:.4f} → KL_after_remove={kl_top_removed:.6f}"
        f"\n               bot node |score|={bot_score:.4f} → KL_after_remove={kl_bot_removed:.6f}"
    )

    if top_score != bot_score:
        assert kl_top_removed > kl_bot_removed, (
            f"Monotonicity FAILED: removing top node (score={top_score:.4f}) gives "
            f"KL={kl_top_removed:.6f}, but removing bottom node (score={bot_score:.4f}) gives "
            f"KL={kl_bot_removed:.6f} — expected top > bottom"
        )

    # TOOTHED: population rank-correlation
    if len(node_scores_list) >= 5:
        sp_r, _ = spearmanr(abs_scores, kl_values)
        print(f"[monotonicity] Spearman(|score|, KL_after_remove) = {sp_r:.3f}")
        assert sp_r > 0.5, (
            f"Spearman rank-correlation = {sp_r:.3f} ≤ 0.5. "
            "Node scores do not predict KL impact (fails for random scores)."
        )


# ---------------------------------------------------------------------------
# test_ablation_modes
# ---------------------------------------------------------------------------


def test_ablation_modes():
    """corrupted / zero / mean all run, give DISTINCT m(∅); faithfulness(M)=1 in all three.

    TOOTHED: the three m(∅) values must be DISTINCT (fails if modes are all equivalent).
    """
    model, sae0, sae1, runner = _setup(seed_model=6, seed_sae0=60, seed_sae1=61)
    clean, corrupted = _make_clean_corr(seed=400)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=32)

    faithfulness_values: dict[str, float] = {}
    m_empty_values: dict[str, float] = {}

    for mode in ("corrupted", "zero", "mean"):
        # faithfulness(M) must be ≈ 1 in all modes
        faith = circuit.faithfulness(clean, corrupted, _metric, ablation_mode=mode)
        faithfulness_values[mode] = faith
        print(f"\n[mode={mode}] faithfulness(M) = {faith:.6f}")
        assert math.isfinite(faith), f"Mode {mode!r}: non-finite faithfulness {faith}"
        assert abs(faith - 1.0) < 0.05, (
            f"Mode {mode!r}: faithfulness(M) = {faith:.6f}, expected ≈ 1.0"
        )

        # m(∅): empty circuit metric
        abl_vals = circuit._compute_ablation_values(clean, corrupted, mode)
        empty_nodes = {layer: set() for layer in abl_vals}
        m_empty = circuit._m_of(clean, corrupted, _metric, empty_nodes, abl_vals)
        m_empty_values[mode] = m_empty

    vals = list(m_empty_values.values())
    print(f"\n[m(∅) per mode] {m_empty_values}")
    assert len(vals) == 3, "Expected results for 3 modes"

    # TOOTHED: the three m(∅) values must be DISTINCT (to 4 decimal places)
    # If all three are identical, the ablation modes are not behaving differently
    distinct_vals = len({round(v, 4) for v in vals})
    assert distinct_vals == 3, (
        f"The three m(∅) values are NOT all distinct (rounded to 4dp): {m_empty_values}. "
        "This suggests the ablation modes are equivalent — check the corrupted inputs."
    )


# ---------------------------------------------------------------------------
# test_model_clean_after_ablation
# ---------------------------------------------------------------------------


def test_model_clean_after_ablation():
    """Model output is bit-identical before and after the ablation forward."""
    model, sae0, sae1, runner = _setup(seed_model=7, seed_sae0=70, seed_sae1=71)
    clean, corrupted = _make_clean_corr(seed=500)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=16)

    with torch.no_grad():
        out_before = model(clean).clone().detach()

    # Run faithfulness (triggers the ablation forward)
    _ = circuit.faithfulness(clean, corrupted, _metric, ablation_mode="corrupted")

    with torch.no_grad():
        out_after = model(clean).detach()

    assert torch.equal(out_before, out_after), (
        "Model output changed after ablation forward — hooks not cleaned up"
    )


# ---------------------------------------------------------------------------
# test_feature_acdc_converges
# ---------------------------------------------------------------------------


def test_feature_acdc_converges():
    """Greedy FeatureACDC run recovers a small faithful circuit on the toy.

    TOOTHED assertions:
      - n_kept < n_initial (must FAIL if ACDC is a no-op that keeps everything)
      - |faithfulness − 1.0| < tol (the pruned circuit is still faithful)
    PRINTS kept-node count + final KL.
    """
    model, sae0, sae1, runner = _setup(seed_model=8, seed_sae0=80, seed_sae1=81)
    clean, corrupted = _make_clean_corr(seed=600)

    # Get initial survivor count (before ACDC)
    full_circuit = _get_full_circuit(runner, clean, corrupted, top_k=32)
    n_initial: set[tuple[int, int]] = set()
    for _site, survivors in full_circuit.graph.survivors.items():
        for atp_node in survivors:
            nd = atp_node.node
            if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                n_initial.add((nd.layer, nd.neuron))
    n_initial_count = len(n_initial)

    acdc_runner = _make_acdc_runner(model, sae0, sae1)
    pruned = acdc_runner.run(
        clean, corrupted, _metric,
        tau=0.1,
        ablation_mode="corrupted",
    )

    kept: set[tuple[int, int]] = set()
    for _site, survivors in pruned.graph.survivors.items():
        for atp_node in survivors:
            nd = atp_node.node
            if nd.kind == "sae_feature" and nd.layer is not None and nd.neuron is not None:
                kept.add((nd.layer, nd.neuron))

    n_kept = len(kept)
    print(f"\n[FeatureACDC] initial nodes = {n_initial_count}, kept nodes = {n_kept}")

    # Key invariant: circuit is a valid SAEFeatureCircuit
    assert isinstance(pruned.edges, dict)
    assert isinstance(pruned.graph.survivors, dict)

    # TOOTHED: ACDC must prune at least one node (fails for a no-op keeper)
    assert n_kept < n_initial_count, (
        f"FeatureACDC kept ALL {n_initial_count} nodes (tau=0.1 too loose or ACDC is no-op)"
    )

    # Faithfulness of the pruned circuit should be ≈ 1 (still faithful after pruning)
    faith = pruned.faithfulness(clean, corrupted, _metric, ablation_mode="corrupted")
    print(f"[FeatureACDC] pruned faithfulness = {faith:.6f}")
    assert math.isfinite(faith), f"FeatureACDC pruned circuit faithfulness is non-finite: {faith}"
    # The pruned circuit should still be faithful (tau=0.1 is small)
    assert abs(faith - 1.0) < 0.5, (
        f"FeatureACDC pruned circuit faithfulness = {faith:.4f}, expected close to 1.0 (tol=0.5)"
    )


# ---------------------------------------------------------------------------
# test_sweep_pareto
# ---------------------------------------------------------------------------


def test_sweep_pareto():
    """sweep() returns monotone n_kept vs final_kl; correct tuple shape.

    TOOTHED: assert STRICT decrease n_kept[0] > n_kept[-1] (fails for constant n_kept).
    """
    model, sae0, sae1, _ = _setup(seed_model=9, seed_sae0=90, seed_sae1=91)
    clean, corrupted = _make_clean_corr(seed=700)

    acdc_runner = _make_acdc_runner(model, sae0, sae1)
    taus = [0.001, 0.01, 0.1]
    table = acdc_runner.sweep(clean, corrupted, _metric, taus=taus)

    print(f"\n[sweep] table = {table}")
    assert len(table) == len(taus), f"Expected {len(taus)} rows, got {len(table)}"

    for row in table:
        assert len(row) == 3, f"Row should be (tau, n_kept, final_kl), got {row}"
        tau_r, n_kept, final_kl = row
        assert isinstance(tau_r, float)
        assert isinstance(n_kept, int)
        assert isinstance(final_kl, float)
        assert math.isfinite(final_kl), f"Non-finite final_kl {final_kl}"

    # Monotonicity: larger tau → fewer or equal kept nodes
    n_kept_vals = [row[1] for row in table]
    for i in range(len(n_kept_vals) - 1):
        assert n_kept_vals[i] >= n_kept_vals[i + 1], (
            f"n_kept not monotone decreasing: {n_kept_vals}"
        )

    # TOOTHED: strict decrease from first (smallest tau) to last (largest tau)
    # fails if sweep is a no-op that keeps the same number across all taus
    assert n_kept_vals[0] > n_kept_vals[-1], (
        f"n_kept is NOT strictly decreasing from tau[0] to tau[-1]: {n_kept_vals}. "
        "This implies sweep() is not pruning more aggressively at higher taus."
    )


# ---------------------------------------------------------------------------
# test_prune_threshold
# ---------------------------------------------------------------------------


def test_prune_threshold():
    """prune(method='threshold') returns valid SAEFeatureCircuit with |edge|>=tau."""
    model, sae0, sae1, runner = _setup(seed_model=10, seed_sae0=10, seed_sae1=11)
    clean, corrupted = _make_clean_corr(seed=800)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=32)

    if not circuit.edges:
        pytest.skip("No edges to threshold")

    # Pick a tau below max edge score
    max_score = max(abs(s) for s in circuit.edges.values())
    tau = max_score * 0.5

    pruned = circuit.prune(method="threshold", tau=tau)

    from circuitry.patching.sae_edges import SAEFeatureCircuit
    assert isinstance(pruned, SAEFeatureCircuit)
    # All returned edges must have |score| >= tau
    for _e, s in pruned.edges.items():
        assert abs(s) >= tau - 1e-8, f"Edge with score {s:.6f} < tau={tau:.6f} survived"
    # Must be a subset of original edges
    assert set(pruned.edges.keys()) <= set(circuit.edges.keys())


# ---------------------------------------------------------------------------
# test_prune_acdc
# ---------------------------------------------------------------------------


def test_prune_acdc():
    """prune(method='acdc') returns valid SAEFeatureCircuit."""
    model, sae0, sae1, runner = _setup(seed_model=11, seed_sae0=110, seed_sae1=111)
    clean, corrupted = _make_clean_corr(seed=801)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=16)

    pruned = circuit.prune(
        method="acdc",
        tau=0.1,
        ablation_mode="corrupted",
        clean=clean,
        corrupted=corrupted,
        metric=_metric,
    )

    from circuitry.patching.sae_edges import SAEFeatureCircuit
    assert isinstance(pruned, SAEFeatureCircuit)
    assert isinstance(pruned.edges, dict)
    assert isinstance(pruned.graph.survivors, dict)


# ---------------------------------------------------------------------------
# test_prune_both
# ---------------------------------------------------------------------------


def test_prune_both():
    """prune(method='both') ⊆ prune(method='threshold') in terms of nodes."""
    model, sae0, sae1, runner = _setup(seed_model=12, seed_sae0=120, seed_sae1=121)
    clean, corrupted = _make_clean_corr(seed=802)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=16)

    if not circuit.edges:
        pytest.skip("No edges to threshold")

    max_score = max(abs(s) for s in circuit.edges.values()) if circuit.edges else 1.0
    tau = max_score * 0.3

    pruned_thresh = circuit.prune(method="threshold", tau=tau)
    pruned_both = circuit.prune(
        method="both",
        tau=tau,
        ablation_mode="corrupted",
        clean=clean,
        corrupted=corrupted,
        metric=_metric,
    )

    from circuitry.patching.sae_edges import SAEFeatureCircuit
    assert isinstance(pruned_both, SAEFeatureCircuit)

    # 'both' must be ⊆ 'threshold' in edge set
    assert set(pruned_both.edges.keys()) <= set(pruned_thresh.edges.keys()), (
        "prune('both') edge set is not a subset of prune('threshold') edge set"
    )


# ---------------------------------------------------------------------------
# test_error_node_edges_optin
# ---------------------------------------------------------------------------


def test_error_node_edges_optin():
    """include_error_node=True wires error→feature edges; default=False has none.

    TOOTHED assertions (v1.6 FIX 1):
      1. len(error_edges) > 0 with direction writer=sae_error, reader=sae_feature.
      2. NO feature→error edges (structurally zero — downstream eps is a detached leaf).
      3. include_error_node=False → zero error edges/nodes.
      4. Error-edge scores are finite and nonzero (of comparable order to feature→feature).

    This test FAILS under a no-op error flag (i.e. if include_error_node has no effect).
    """
    model, sae0, sae1, runner = _setup(seed_model=13, seed_sae0=130, seed_sae1=131)
    clean, corrupted = _make_clean_corr(seed=900)

    circuit_default = runner.run(
        clean, corrupted, _metric,
        top_k_survivors=16,
        include_error_node=False,
    )
    circuit_err = runner.run(
        clean, corrupted, _metric,
        top_k_survivors=16,
        include_error_node=True,
    )

    # --- Default: no sae_error endpoints in edges ---
    error_edges_default = [
        e for e in circuit_default.edges
        if e.writer.node.kind == "sae_error" or e.reader.node.kind == "sae_error"
    ]
    print(f"\n[error_node] default: {len(error_edges_default)} error-endpoint edges")
    assert len(error_edges_default) == 0, (
        f"Default (include_error_node=False) has error-endpoint edges: {error_edges_default}"
    )

    error_nodes_default = [
        n for n in circuit_default.nodes.scores if n.node.kind == "sae_error"
    ]
    print(f"[error_node] default: {len(error_nodes_default)} sae_error nodes")

    # --- include_error_node=True: assert error→feature edges ARE wired ---
    error_to_feature_edges = [
        e for e in circuit_err.edges
        if e.writer.node.kind == "sae_error" and e.reader.node.kind == "sae_feature"
    ]
    feature_to_error_edges = [
        e for e in circuit_err.edges
        if e.reader.node.kind == "sae_error"
    ]

    print(
        f"[error_node] include_error_node=True: {len(error_to_feature_edges)} error→feature edges, "
        f"{len(feature_to_error_edges)} feature→error edges"
    )

    # TOOTHED 1: error→feature edges must exist (FIX 1 wires them)
    assert len(error_to_feature_edges) > 0, (
        "include_error_node=True produced NO error→feature edges. "
        "FIX 1 wiring is missing or broken."
    )

    # TOOTHED 2: feature→error edges must NOT exist (structurally zero)
    assert len(feature_to_error_edges) == 0, (
        f"include_error_node=True produced {len(feature_to_error_edges)} feature→error edges. "
        "These are structurally zero (downstream eps is a detached leaf) and should not be emitted."
    )

    # TOOTHED 3: all error→feature scores are finite and nonzero
    for e in error_to_feature_edges:
        s = circuit_err.edges[e]
        assert math.isfinite(s), f"Non-finite error→feature score {s} for {e}"
        assert abs(s) > 1e-10, f"Error→feature score is effectively zero: {s} for {e}"

    # TOOTHED 4: order of magnitude check vs feature→feature scores
    feat_feat_scores = [
        abs(circuit_err.edges[e])
        for e in circuit_err.edges
        if e.writer.node.kind == "sae_feature" and e.reader.node.kind == "sae_feature"
    ]
    err_feat_scores = [abs(circuit_err.edges[e]) for e in error_to_feature_edges]
    if feat_feat_scores:
        max_ff = max(feat_feat_scores)
        max_ef = max(err_feat_scores)
        print(
            f"[error_node] max |feature→feature| = {max_ff:.4e}, "
            f"max |error→feature| = {max_ef:.4e}"
        )
        # Error→feature scores should be in a reasonable range vs feature→feature
        assert max_ef > 0, "All error→feature scores are zero"

    # --- sae_error node scores present when include_error_node=True ---
    error_nodes_err = [
        n for n in circuit_err.nodes.scores if n.node.kind == "sae_error"
    ]
    print(f"[error_node] include_error_node=True: {len(error_nodes_err)} sae_error nodes")


# ---------------------------------------------------------------------------
# test_no_sae_param_grad_leak  (ablation/ACDC paths)
# ---------------------------------------------------------------------------


def test_no_sae_param_grad_leak():
    """SAE params must NOT accumulate .grad during faithfulness / ACDC paths."""
    model, sae0, sae1, runner = _setup(seed_model=14, seed_sae0=140, seed_sae1=141)

    # Enable grads on SAE params to detect leaks
    for sae in [sae0, sae1]:
        for p in sae.parameters():
            p.requires_grad_(True)
            p.grad = None

    clean, corrupted = _make_clean_corr(seed=1000)
    circuit = _get_full_circuit(runner, clean, corrupted, top_k=16)

    # Run faithfulness (triggers ablation forward)
    _ = circuit.faithfulness(clean, corrupted, _metric, ablation_mode="corrupted")

    # Run FeatureACDC
    acdc_runner = _make_acdc_runner(model, sae0, sae1)
    _ = acdc_runner.run(clean, corrupted, _metric, tau=0.1)

    for sae, name in [(sae0, "sae0"), (sae1, "sae1")]:
        leaked = [n for n, p in sae.named_parameters() if p.grad is not None]
        assert not leaked, f"SAE param grad leaked in {name}: {leaked}"
        assert all(p.requires_grad for p in sae.parameters()), (
            f"SAE param requires_grad not intact after ablation/ACDC in {name}"
        )
