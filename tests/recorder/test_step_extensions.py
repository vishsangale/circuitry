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
