"""Activation patching — interventional diagnostics. See design spec §2.

Sub-spec 1 (core primitive): Site, patch_site, PatchRunner.
Sub-spec 2 (EAP): EAPRunner, EAPResult, the Node/Edge graph model.
Sub-spec 3 (AtP*): AtPRunner, AtPResult, AtPNode.
Sub-spec 4 (ACDC): ACDCRunner, ACDCResult — greedy circuit discovery.
"""
from __future__ import annotations

from circuitry.patching.acdc import ACDCResult, ACDCRunner
from circuitry.patching.atp import AtPNode, AtPResult, AtPRunner
from circuitry.patching.eap import EAPResult, EAPRunner
from circuitry.patching.graph import Edge, Node
from circuitry.patching.intervene import PatchHandle, patch_site
from circuitry.patching.runner import PatchResult, PatchRunner
from circuitry.patching.sites import Site

__all__ = [
    "ACDCResult",
    "ACDCRunner",
    "AtPNode",
    "AtPResult",
    "AtPRunner",
    "EAPResult",
    "EAPRunner",
    "Edge",
    "Node",
    "PatchHandle",
    "PatchResult",
    "PatchRunner",
    "Site",
    "patch_site",
]
