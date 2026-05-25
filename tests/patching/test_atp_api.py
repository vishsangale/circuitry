"""Verify AtP* types are exported in the public circuitry.patching API."""
from __future__ import annotations


def test_atp_public_api():
    import circuitry.patching as p
    assert {"AtPRunner", "AtPResult", "AtPNode"} <= set(p.__all__)
    assert hasattr(p, "AtPRunner")
    assert hasattr(p, "AtPResult")
    assert hasattr(p, "AtPNode")
