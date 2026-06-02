"""Regression tests for eval findings F8 and F23 from the v1.7 real-model evaluation.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md`` (F8, F23).

Each test encodes the *correct* behaviour, so it is RED under the current code and
flips GREEN once the finding is fixed. They are marked ``xfail(strict=True)`` so the
suite stays green today while the fix is pending; when a fix lands the test XPASSes,
strict-xfail turns that into a failure, and the marker must be removed.

To watch them actually fail (reproduce the findings), run with ``--runxfail``::

    .venv/bin/pytest tests/recorder/test_scan_compare_eval_findings.py --runxfail

Reproduced evidence (captured 2026-06-01):

  F8 — scan_run with ``activation_diagnostics=["dead_fraction"]`` + 3 checkpoints emits
  ZERO activation scalar rows while ``weight/effective_rank`` emits 6 rows normally.
  A live Recorder driven by the same recipe with a forward pass emits 3 activation rows
  from the same model.  Root cause: scan.py calls ``rec.step()`` with no preceding
  forward pass → ``ctx.activations`` is always ``{}`` → every activation diagnostic
  loop body is skipped silently.

  F23 — ``compare_runs(live_run, scan_run)`` where *live_run* has both
  ``activation/dead_fraction`` and ``weight/effective_rank`` families while *scan_run*
  has only ``weight/effective_rank`` (because F8 suppresses activation output).  The
  ``activation/dead_fraction`` FamilyDelta gets ``last_b=nan``, ``delta=nan``.
  ``warnings.catch_warnings(record=True)`` around the call captures **zero** warnings.
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.scan import scan_run
from circuitry.writers.base import RecordingWriter

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _toy(seed: int) -> nn.Module:
    """Small 4→8→4 Sequential; fast and deterministic."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _register_act_recipe(name: str) -> None:
    """Recipe with both weight AND activation diagnostics."""
    register_recipe(Recipe(
        name=name,
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT, pattern=r"^\d+$"),
            HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$"),
        ],
        weight_diagnostics=["effective_rank"],
        activation_diagnostics=["dead_fraction"],
    ))


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


# ---------------------------------------------------------------------------
# F8 — scan_run activation diagnostics silently emit nothing
#
# Root cause: scan.py calls rec.step() with no preceding forward pass →
# ctx.activations == {} → dead_fraction / kurtosis / participation_ratio /
# gate_stats / logit_lens_kl all skip silently.
#
# Correct behaviour: scan_run MUST either
#   (a) guard empty activations cleanly (skip with a single warning per run,
#       NOT silently), or
#   (b) support a forward_fn= so activation diagnostics can run on checkpoints.
#
# The crispest RED assertion is (b): if a forward_fn is supplied, scan_run
# MUST produce at least one activation scalar.  We also encode (a) as a
# second, independent sub-finding: if no forward_fn is supplied and
# activation diagnostics are in the recipe, scan_run should at minimum emit
# a warning (currently it emits nothing at all for dead_fraction).
# ---------------------------------------------------------------------------


def test_F8_scan_run_activation_diagnostics_emit_nonzero_rows(tmp_path):
    """scan_run MUST produce at least one activation scalar when the recipe
    contains activation diagnostics and checkpoints exist.

    Confirmed failure mode (current code):
      - weight/effective_rank emits 4 rows across 2 checkpoints (correct).
      - activation/dead_fraction emits 0 rows across 2 checkpoints (bug).
      - No warning or error is raised — the silence is total.

    This test asserts the *correct* post-fix behaviour: at least one
    ``activation/dead_fraction/…`` tag must be present in the writer output.
    Fixed by adding forward_fn= parameter to scan_run so activation hooks fire.
    """
    _register_act_recipe("scan-f8-a")

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step, seed in [(100, 0), (200, 1)]:
        torch.save(_toy(seed).state_dict(), ckpts / f"step{step:09d}.pt")

    writer = RecordingWriter()
    # Provide a forward_fn so OUTPUT hooks fire and ctx.activations is populated.
    dummy_input = torch.zeros(1, 4)
    scan_run(
        run_dir=tmp_path,
        recipe="scan-f8-a",
        out_dir=tmp_path / "out",
        model_factory=lambda: _toy(0),
        writer=writer,
        forward_fn=lambda m: m(dummy_input),
    )

    activation_rows = [tag for tag, _v, _s in writer.scalars if "activation" in tag]
    weight_rows = [tag for tag, _v, _s in writer.scalars if "weight" in tag]

    # Sanity: weight diagnostics DID emit (proves scan_run ran and checkpoints loaded).
    assert weight_rows, (
        "scan_run emitted no weight rows at all — something else is wrong; "
        "this test cannot meaningfully check the F8 activation symptom."
    )

    # CORRECT behaviour: at least one activation row.
    # CURRENT (buggy) behaviour: zero activation rows (ctx.activations always empty).
    assert activation_rows, (
        f"scan_run emitted ZERO activation rows despite the recipe containing "
        f"activation_diagnostics=['dead_fraction'] and {len(weight_rows)} weight "
        f"rows being produced. ctx.activations is empty because scan_run never "
        f"runs a forward pass before calling rec.step() (F8)."
    )


def test_F8_scan_run_warns_when_activation_diagnostics_skipped(tmp_path):
    """If activation diagnostics are in the recipe but ctx.activations is always
    empty, scan_run MUST emit a warning (UserWarning or logging.WARNING) so the
    user knows activation families are absent — not silently produce zero rows.

    Confirmed failure mode (current code):
      - dead_fraction skips with no output and no warning.
      - logit_lens_kl does log a warning AND permanently sets _lens_meta=None,
        but dead_fraction/kurtosis/participation_ratio/gate_stats are completely
        silent. The fix should either warn once per family or per-run.
    """
    _register_act_recipe("scan-f8-b")

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step, seed in [(100, 0), (200, 1)]:
        torch.save(_toy(seed).state_dict(), ckpts / f"step{step:09d}.pt")

    writer = RecordingWriter()

    # Capture Python warnings (UserWarning) AND log-level warnings via caplog is
    # not available here — we use warnings.catch_warnings to capture warnings
    # issued via warnings.warn().  A logging.WARNING is harder to capture
    # portably, so we also accept if the activation rows are non-empty (meaning
    # the fix took path (b) — adding a forward_fn — instead of just warning).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scan_run(
            run_dir=tmp_path,
            recipe="scan-f8-b",
            out_dir=tmp_path / "out",
            model_factory=lambda: _toy(0),
            writer=writer,
        )

    activation_rows = [tag for tag, _v, _s in writer.scalars if "activation" in tag]

    # CORRECT behaviour (either path): warn OR produce activation output.
    # CURRENT (buggy) behaviour: no warning, no output.
    assert caught or activation_rows, (
        "scan_run produced no activation rows AND issued no Python warnings despite "
        "the recipe containing activation_diagnostics=['dead_fraction']. "
        "The silent skip is the F8 bug; fix must either warn or support forward_fn."
    )


# ---------------------------------------------------------------------------
# F23 — compare_runs is silently asymmetric: NaN sentinel with no warning
#
# When run_a (a live run) has activation/dead_fraction but run_b (a scan run)
# does not, compare_runs silently emits last_b=nan / delta=nan with NO warning.
# The user has no indication that the family-set mismatch occurred.
#
# Correct behaviour: compare_runs / build_compare_report MUST emit a warning
# (UserWarning or logging.WARNING) when the family sets of the two runs differ.
# ---------------------------------------------------------------------------


def test_F23_compare_runs_warns_on_family_set_mismatch(tmp_path):
    """compare_runs MUST emit a warning when the two runs have different
    metric-family sets (e.g. live run has activation families, scan run does not).

    Confirmed failure mode (current code):
      compare_runs(live_run, scan_run) where live_run has
        - weight/effective_rank  AND  activation/dead_fraction
      and scan_run has only
        - weight/effective_rank
      produces a FamilyDelta for activation/dead_fraction with
        last_b=nan, delta=nan, trend_agrees=False
      AND raises ZERO warnings — the mismatch is completely silent.

    This test asserts the correct post-fix behaviour: at least one UserWarning
    must be raised when the runs' family sets differ.
    """
    from circuitry.recorder.compare import compare_runs

    # Simulate a live recorder run: weight + activation families.
    run_live = tmp_path / "live"
    _write_jsonl(run_live / "metrics.jsonl", [
        {"kind": "scalar", "tag": "weight/effective_rank/0", "step": 0, "value": 3.5},
        {"kind": "scalar", "tag": "weight/effective_rank/0", "step": 1, "value": 3.6},
        {"kind": "scalar", "tag": "activation/dead_fraction/0", "step": 0, "value": 0.10},
        {"kind": "scalar", "tag": "activation/dead_fraction/0", "step": 1, "value": 0.12},
    ])

    # Simulate a scan run: weight families only (activation silently absent — F8).
    run_scan = tmp_path / "scan"
    _write_jsonl(run_scan / "metrics.jsonl", [
        {"kind": "scalar", "tag": "weight/effective_rank/0", "step": 100, "value": 4.0},
        {"kind": "scalar", "tag": "weight/effective_rank/0", "step": 200, "value": 4.1},
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deltas = compare_runs(run_live, run_scan)

    # Confirm the NaN sentinel IS present (verify this test has teeth).
    fd_map = {fd.section: fd for fd in deltas}
    assert "activation/dead_fraction" in fd_map, (
        "activation/dead_fraction section unexpectedly absent from compare output; "
        "the test setup may be wrong."
    )
    act_fd = fd_map["activation/dead_fraction"]
    assert math.isnan(act_fd.last_b), (
        f"Expected last_b=nan for the absent activation/dead_fraction family in "
        f"run_scan, got last_b={act_fd.last_b!r}. The test setup is incorrect."
    )

    # CORRECT behaviour: at least one warning about the family-set mismatch.
    # CURRENT (buggy) behaviour: zero warnings — completely silent NaN injection.
    assert caught, (
        "compare_runs produced a NaN sentinel for the 'activation/dead_fraction' "
        "family (present in run_a but absent in run_b) and emitted ZERO Python "
        "warnings. The silent NaN is the F23 bug; fix must warn on family-set "
        "mismatch (e.g. UserWarning listing the families that differ)."
    )
