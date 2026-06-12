"""Activation patching — interventional diagnostics. See design spec §2.

Sub-spec 1 (core primitive): Site, patch_site, PatchRunner.
Sub-spec 2 (EAP): EAPRunner, EAPResult, the Node/Edge graph model.
Sub-spec 3 (AtP*): AtPRunner, AtPResult, AtPNode.
Sub-spec 4 (ACDC): ACDCRunner, ACDCResult — greedy circuit discovery.
Sub-spec 5 (SAE features): SAEFeatureRunner — node-level SAE feature attribution.
         Lazy import: SAEFeatureRunner is resolved on first access to avoid
         pulling sae_lens (and transitively transformer_lens) at patching
         import time.  All other names are eagerly imported.
"""
from __future__ import annotations

from circuitry.patching.acdc import ACDCResult, ACDCRunner
from circuitry.patching.atp import AtPNode, AtPResult, AtPRunner
from circuitry.patching.causal_trace import CausalTraceResult, CausalTraceRunner
from circuitry.patching.cd import CDResult, cd_token_contributions
from circuitry.patching.certified import CertifiedCircuitResult, CertifiedCircuitRunner
from circuitry.patching.clt import CLTEdge, CLTGraphResult, CLTGraphRunner, CLTNode
from circuitry.patching.consensus import CircuitConsensus
from circuitry.patching.das import DASResult, DASRunner
from circuitry.patching.eap import EAPResult, EAPRunner
from circuitry.patching.edge_pruning import EdgePruningResult, EdgePruningRunner
from circuitry.patching.export import (
    save_html,
    save_neuronpedia_graph,
    to_html,
    to_neuronpedia_graph,
)
from circuitry.patching.generation import (
    GenerationAttributionSetup,
    GenerationTrace,
    StepRecord,
    apply_steer_steps,
    generation_attribution,
    patch_site_steps,
    prepare_generation_attribution,
    trace_generation,
)
from circuitry.patching.graph import Edge, Node
from circuitry.patching.hap import HAPRunner
from circuitry.patching.head_knockout import HeadKnockoutResult, HeadKnockoutRunner
from circuitry.patching.hyperdas import HyperDASNet, HyperDASResult, HyperDASRunner
from circuitry.patching.intervene import PatchHandle, patch_site
from circuitry.patching.iti import ITIConfig, apply_iti, fit_iti
from circuitry.patching.mean_ablation import compute_mean_activation, mean_ablation
from circuitry.patching.patch_grid import PatchGridResult, PatchGridRunner
from circuitry.patching.relp import ReLPRunner
from circuitry.patching.runner import PatchResult, PatchRunner
from circuitry.patching.scrubbing import CausalScrubResult, CausalScrubRunner, CircuitHypothesis
from circuitry.patching.sites import Site
from circuitry.patching.spd import SPDResult, SPDRunner
from circuitry.patching.steer import apply_steer, steer_vector
from circuitry.patching.tl_bridge import to_hooked_transformer

__all__ = [
    "ACDCResult",
    "ACDCRunner",
    "AtPNode",
    "AtPResult",
    "AtPRunner",
    "CausalTraceResult",
    "CausalTraceRunner",
    "CDResult",
    "PatchGridResult",
    "PatchGridRunner",
    "CLTEdge",
    "CLTGraphResult",
    "CLTGraphRunner",
    "CLTNode",
    "CausalScrubResult",
    "ITIConfig",
    "CausalScrubRunner",
    "CertifiedCircuitResult",
    "CertifiedCircuitRunner",
    "CircuitConsensus",
    "CircuitHypothesis",
    "CrosscoderWrapper",
    "DASResult",
    "DASRunner",
    "EAPResult",
    "EAPRunner",
    "Edge",
    "EdgePruningResult",
    "EdgePruningRunner",
    "FeatureACDCRunner",
    "GenerationAttributionSetup",
    "GenerationTrace",
    "HAPRunner",
    "HeadKnockoutResult",
    "HeadKnockoutRunner",
    "compute_mean_activation",
    "mean_ablation",
    "HyperDASNet",
    "HyperDASResult",
    "HyperDASRunner",
    "Node",
    "PatchHandle",
    "PatchResult",
    "PatchRunner",
    "ReLPRunner",
    "SAEFeatureCircuit",
    "SAEFeatureEdge",
    "SAEFeatureEdgeGraph",
    "SAEFeatureEdgeRunner",
    "SAEFeatureRunner",
    "SAEFeatureTemporalRunner",
    "SPDResult",
    "SPDRunner",
    "Site",
    "StepRecord",
    "TemporalAtPResult",
    "TranscoderWrapper",
    "apply_iti",
    "apply_steer",
    "apply_steer_steps",
    "cd_token_contributions",
    "fit_iti",
    "generation_attribution",
    "patch_site",
    "patch_site_steps",
    "prepare_generation_attribution",
    "save_html",
    "save_neuronpedia_graph",
    "steer_vector",
    "to_hooked_transformer",
    "trace_generation",
    "to_html",
    "to_neuronpedia_graph",
]

_LAZY = {
    "CrosscoderWrapper": "circuitry.patching.sae_features",
    "SAEFeatureRunner": "circuitry.patching.sae_features",
    "TranscoderWrapper": "circuitry.patching.sae_features",
    "SAEFeatureEdge": "circuitry.patching.sae_edges",
    "SAEFeatureEdgeGraph": "circuitry.patching.sae_edges",
    "SAEFeatureCircuit": "circuitry.patching.sae_edges",
    "SAEFeatureEdgeRunner": "circuitry.patching.sae_edges",
    "FeatureACDCRunner": "circuitry.patching.sae_edges",
    "SAEFeatureTemporalRunner": "circuitry.patching.sae_temporal",
    "TemporalAtPResult": "circuitry.patching.sae_temporal",
}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
