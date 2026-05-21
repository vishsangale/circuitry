"""Smoke tests for the parity comparator — exercise _compare with synthetic
scalar streams, no mendu dependency."""
from collections import namedtuple

import scripts.parity_check as pc  # noqa: E402 — scripts/ is not a package

Scalar = namedtuple("Scalar", ["step", "value", "wall_time"])


def _mk(value, step=0):
    return Scalar(step=step, value=value, wall_time=0.0)


def test_compare_identical_scalars_passes():
    same = {tag: [_mk(0.5, 0), _mk(0.4, 1)] for tag in pc.UNIVERSAL_TAGS.keys()}
    out = pc._compare(same, same)
    # All UNIVERSAL_TAGS present in both; values identical → no failures.
    assert out == []


def test_compare_missing_tag_in_mendu():
    mendu = {}  # empty
    circuitry = {tag: [_mk(0.0)] for tag in pc.UNIVERSAL_TAGS.values()}
    out = pc._compare(mendu, circuitry)
    assert any("missing in mendu" in f for f in out)


def test_compare_off_by_too_much():
    mendu = {tag: [_mk(1.0, 0)] for tag in pc.UNIVERSAL_TAGS.keys()}
    circuitry = {tag: [_mk(2.0, 0)] for tag in pc.UNIVERSAL_TAGS.values()}
    out = pc._compare(mendu, circuitry)
    # All tags off by 1.0 (rtol=1e-5 way too tight) → all should fail
    assert len(out) >= len(pc.UNIVERSAL_TAGS)
