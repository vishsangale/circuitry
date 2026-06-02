"""Regression tests for v1.7 real-model evaluation findings in ``recipes/vision``.

Source: ``docs/observations/2026-05-31-real-model-evaluation.md`` (F7, F30).

Each test encodes the *correct* behaviour, so it is RED under the current code and
flips GREEN once the finding is fixed.  They are marked ``xfail(strict=True)`` so the
suite stays green today while the fix is pending; when a fix lands the test XPASSes,
strict-xfail turns that into a failure, and the marker must be removed.

To watch them actually fail (reproduce the findings), run with ``--runxfail``::

    .venv/bin/pytest tests/recipes/test_vision_eval_findings.py --runxfail
"""

from __future__ import annotations

import pytest
import torch

from circuitry.recipes import _clear_registry_for_tests
from circuitry.recipes.vision import register as register_vision
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    register_vision()
    yield
    _clear_registry_for_tests()


# ---------------------------------------------------------------------------
# F7 — vision recipe misses ResNet-18 `fc` classifier head and `downsample`
#      convs because every pattern requires a DIGIT suffix (conv\d+|fc\d+).
#
#      Confirmed on torchvision resnet18(weights=None): only 17 of 67
#      non-root modules matched (25%), and no tag containing "fc" is ever
#      emitted.  The `fc` Linear (classifier head) and the 3 `downsample.0`
#      Conv2d layers are all silently ignored.
#
#      Fix: broaden pattern to allow bare `fc` and `downsample` variants,
#      e.g. `(conv\d*|fc\d*|downsample\.\d+|patch_embed|...)`.
# ---------------------------------------------------------------------------


def test_F7_resnet18_fc_head_captured(tmp_path):
    """vision recipe on resnet18 must capture the `fc` classifier head.

    Correct behaviour: after a full forward+step cycle the emitted scalars
    include at least one tag whose module component contains 'fc'.  This proves
    the classifier head (``model.fc``, an nn.Linear) is matched by the recipe's
    hook patterns rather than being silently skipped.

    The test also checks that the overall coverage is materially above the
    currently broken 17/67 (25%) — we require at least 20 matched unique
    module names in the emitted tags, which is a conservative lower-bound for
    a correct pattern (the true number should be >=21 once fc is included).

    Confirmed failure mode (current code):
        - 17 modules matched (conv* only, DIGIT-suffix required).
        - 0 tags containing 'fc'.
        - model.fc (nn.Linear(512, 1000)) is completely invisible to the recipe.
    """
    torchvision = pytest.importorskip("torchvision")

    model = torchvision.models.resnet18(weights=None)
    writer = RecordingWriter()
    rec = Recorder(
        model, run_dir=tmp_path, recipe="vision", writer=writer, every_n_steps=1
    )
    rec.attach()
    out = model(torch.randn(1, 3, 224, 224))
    out.sum().backward()
    rec.step(0)
    rec.detach()

    tags = {t for t, _, _ in writer.scalars}

    # 1. The classifier head must appear in at least one emitted tag.
    assert any("fc" in t for t in tags), (
        "No emitted tag contains 'fc' — the ResNet-18 classifier head (model.fc) "
        "is completely missed by the vision recipe (F7). "
        f"Tags found: {sorted(tags)}"
    )

    # 2. Coverage must be materially above the broken baseline of 17 modules.
    #    Extract unique module names from tags (format: "diag_type/metric/module_name").
    unique_modules = {t.rsplit("/", 1)[-1] for t in tags}
    assert len(unique_modules) >= 20, (
        f"Vision recipe captured only {len(unique_modules)} unique modules on "
        f"resnet18 — expected >= 20 (current broken baseline: 17). "
        "The fc head and/or downsample convs are likely still missing (F7)."
    )


# ---------------------------------------------------------------------------
# F30 — vision recipe matches 0/151 torchvision ViT-B/16 modules, causing a
#        hard RuntimeError on attach().  The patterns are timm/DeiT-locked
#        (`blocks.N.*`); torchvision ViT uses `encoder.layers.encoder_layer_N.*`.
#        The recipe docstring claims ViT support.
#
#        Confirmed RuntimeError message (current code):
#        "HookPoint 0 ((conv\d+|fc\d+|patch_embed|blocks\.\d+\.(attn|mlp))
#        (\.weight)?$) matched 0 modules — refusing to attach
#        (pass strict=False to skip unmatched HookPoints with a warning)"
#
#        Fix: extend the pattern to also cover torchvision ViT module names,
#        e.g. `encoder\.layers\.encoder_layer_\d+\.(self_attention|mlp)`.
# ---------------------------------------------------------------------------


def test_F30_vit_b16_attach_does_not_raise(tmp_path):
    """vision recipe on torchvision vit_b_16 must NOT raise RuntimeError on attach().

    Correct behaviour:
    1. ``rec.attach()`` completes without raising.
    2. A subsequent forward + step() also completes without raising.
    3. At least one scalar tag is emitted — the recipe captures a non-zero number
       of ViT modules.

    Confirmed failure mode (current code):
        RuntimeError: HookPoint 0 (...(conv\\d+|fc\\d+|patch_embed|
        blocks\\.\\d+\\.(attn|mlp))(\\.weight)?$) matched 0 modules —
        refusing to attach (pass strict=False to skip unmatched HookPoints
        with a warning).
    Raised inside Recorder.attach() because every torchvision ViT module name
    uses the `encoder.layers.encoder_layer_N.*` namespace rather than the
    timm/DeiT `blocks.N.*` namespace that the recipe patterns assume.
    """
    torchvision = pytest.importorskip("torchvision")

    model = torchvision.models.vit_b_16(weights=None)
    writer = RecordingWriter()
    rec = Recorder(
        model, run_dir=tmp_path, recipe="vision", writer=writer, every_n_steps=1
    )

    # This is the call that currently raises RuntimeError on torchvision ViT.
    rec.attach()  # must NOT raise

    out = model(torch.randn(1, 3, 224, 224))
    out.sum().backward()
    rec.step(0)
    rec.detach()

    tags = {t for t, _, _ in writer.scalars}

    # Correct behaviour: at least one module was matched and at least one metric
    # was emitted.  (Any single tag is enough to prove the recipe is not blind
    # to torchvision ViT's module namespace.)
    assert len(tags) > 0, (
        "vision recipe emitted 0 scalar tags on torchvision vit_b_16 — "
        "no ViT modules were matched after attach() completed (F30). "
        "The recipe patterns likely still do not cover the torchvision ViT namespace."
    )
