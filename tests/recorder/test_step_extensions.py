import json

import torch
import torch.nn as nn

from circuitry import Recorder
from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource


def _setup_recipe():
    _clear_registry_for_tests()
    register_recipe(Recipe(
        name="probe",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))


def test_loss_components_emitted_as_train_scalars(tmp_path):
    _setup_recipe()
    model = nn.Sequential(nn.Linear(4, 4))

    rec = Recorder(
        model, run_dir=tmp_path, recipe="probe", writer="jsonl", every_n_steps=1,
    )
    rec.attach()
    rec.step(step=0, loss=0.5, loss_components={"lm_loss": 0.4, "aux_loss": 0.1, "lr": 3e-4})
    rec.detach()

    text = (tmp_path / "metrics.jsonl").read_text().splitlines()
    tags = set()
    for line in text:
        import json
        rec_obj = json.loads(line)
        tags.add(rec_obj["tag"])
    assert "train/lm_loss" in tags
    assert "train/aux_loss" in tags
    assert "train/lr" in tags
    assert "train/loss" in tags  # the existing aggregate-loss tag


def test_loss_components_optional(tmp_path):
    _setup_recipe()
    model = nn.Sequential(nn.Linear(4, 4))
    rec = Recorder(model, run_dir=tmp_path, recipe="probe", writer="null", every_n_steps=1)
    rec.attach()
    rec.step(step=0, loss=0.5)  # no loss_components
    rec.detach()


def test_gradient_norms_per_param(tmp_path):
    _clear_registry_for_tests()
    register_recipe(Recipe(
        name="probe_grad",
        hook_points=[
            HookPoint(source=TensorSource.WEIGHT, pattern=r"^linear$"),
            HookPoint(source=TensorSource.GRAD,   pattern=r"^linear$"),
        ],
        gradient_diagnostics=["norms_per_param"],
    ))
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(4, 4))
    rec = Recorder(model, run_dir=tmp_path, recipe="probe_grad", writer="jsonl", every_n_steps=1)
    rec.attach()
    # Run a backward pass to populate .grad
    x = torch.randn(2, 4)
    y = model(x).sum()
    y.backward()
    rec.step(step=0, loss=float(y))
    rec.detach()

    keys = set()
    for line in (tmp_path / "metrics.jsonl").read_text().splitlines():
        keys.add(json.loads(line)["tag"])
    assert any(k.startswith("grad/per_param/") and k.endswith("/norm") for k in keys)
    assert "grad/global/total_norm" in keys


def test_sv_histogram_emitted(tmp_path):
    _clear_registry_for_tests()
    register_recipe(Recipe(
        name="probe_hist",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^linear$")],
        weight_diagnostics=["effective_rank", "sv_histogram"],
    ))
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(8, 8))
    rec = Recorder(model, run_dir=tmp_path, recipe="probe_hist", writer="jsonl", every_n_steps=1)
    rec.attach()
    rec.step(step=0, loss=0.0)
    rec.detach()

    # JsonlWriter stores histograms as ".npy" side files
    art_dir = tmp_path / "circuitry" / "artifacts"
    sv_files = list(art_dir.glob("*sv_histogram*.npy"))
    assert len(sv_files) >= 1
