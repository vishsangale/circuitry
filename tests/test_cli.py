"""CLI entry point tests."""
from __future__ import annotations

import subprocess
import sys

import torch
import torch.nn as nn


def test_cli_list_recipes_prints_stock_names():
    out = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "list-recipes"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "llm" in out
    assert "vision" in out
    assert "two_tower" in out


def test_cli_scan_and_report(tmp_path):
    # Lay down a single checkpoint.
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    torch.save(model.state_dict(), ckpts / "step000000100.pt")

    # `scan` needs a model factory; for the CLI smoke test we use the
    # built-in stub recipe (registered by --recipe llm requires real model
    # structure). Instead, use the "null" recipe wired below.
    # For test purposes we just call `report` against an empty jsonl path,
    # which is enough to exercise the entry point. (Real scan tested in
    # tests/recorder/test_scan.py.)
    (tmp_path / "metrics.jsonl").write_text(
        '{"tag": "loss", "value": 1.0, "step": 0, "kind": "scalar"}\n'
    )
    subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "report",
         "--run", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "inspect" / "report.md").exists()


def test_cli_version_flag_prints_version_and_exits_zero():
    from circuitry import __version__
    out = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "--version"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert __version__ in out
    assert out.strip().startswith("circuitry ")


def test_circuit_compare_acdc_vs_acdc(tmp_path):
    """circuit-compare of two ACDC JSONs prints a markdown diff table."""
    import json

    # Build two minimal ACDC JSONs with overlapping and non-overlapping edges
    a = {
        "kind": "acdc",
        "n_layers": 2,
        "n_heads": 2,
        "final_kl": 0.01,
        "kept_edges": [
            {"writer": {"kind": "embed"}, "reader": {"kind": "mlp", "layer": 0}, "slot": "mlp_in"},
            {"writer": {"kind": "embed"}, "reader": {"kind": "attn_head", "layer": 0, "head": 0}, "slot": "q"},
        ],
        "n_total_edges": 10,
    }
    b = {
        "kind": "acdc",
        "n_layers": 2,
        "n_heads": 2,
        "final_kl": 0.02,
        "kept_edges": [
            {"writer": {"kind": "embed"}, "reader": {"kind": "mlp", "layer": 0}, "slot": "mlp_in"},
            {"writer": {"kind": "embed"}, "reader": {"kind": "attn_head", "layer": 0, "head": 1}, "slot": "v"},
        ],
        "n_total_edges": 10,
    }
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))

    out = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "circuit-compare",
         str(tmp_path / "a.json"), str(tmp_path / "b.json")],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "Circuit Comparison" in out
    assert "total edges" in out


def test_circuit_compare_to_file(tmp_path):
    """circuit-compare --out writes markdown to a file."""
    import json

    circuit = {
        "kind": "acdc",
        "n_layers": 1,
        "n_heads": 1,
        "final_kl": 0.0,
        "kept_edges": [],
        "n_total_edges": 2,
    }
    (tmp_path / "c1.json").write_text(json.dumps(circuit))
    (tmp_path / "c2.json").write_text(json.dumps(circuit))
    out_path = tmp_path / "diff.md"

    subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "circuit-compare",
         str(tmp_path / "c1.json"), str(tmp_path / "c2.json"),
         "--out", str(out_path)],
        capture_output=True, text=True, check=True,
    )
    assert out_path.exists()
    assert "Circuit Comparison" in out_path.read_text()


def test_scan_requires_model_factory_arg():
    """scan subcommand requires --model-factory; missing it → error exit."""
    result = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "scan",
         "--run", "/tmp/fake_run", "--recipe", "llm"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "model-factory" in result.stderr.lower() or "model_factory" in result.stderr.lower()


def test_scan_bad_entry_point_exits_nonzero():
    """scan --model-factory with a bad entry point should exit non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "scan",
         "--run", "/tmp/fake_run", "--recipe", "llm",
         "--model-factory", "nonexistent.module:factory"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_export_graph_html_and_neuronpedia(tmp_path):
    from circuitry.patching.eap import EAPResult
    from circuitry.patching.graph import build_graph

    graph = build_graph(n_layers=1, n_heads=2)
    result = EAPResult(
        graph=graph,
        scores={e: 0.1 * (i + 1) for i, e in enumerate(graph.edges)},
    )
    circuit_path = tmp_path / "circuit.json"
    result.save(circuit_path)

    html_out = tmp_path / "graph.html"
    proc = subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "export-graph",
         str(circuit_path), "--format", "html", "--out", str(html_out)],
        capture_output=True, text=True, check=True,
    )
    assert str(html_out) in proc.stdout
    assert html_out.read_text().startswith("<!DOCTYPE html>")

    np_out = tmp_path / "graph.np.json"
    subprocess.run(
        [sys.executable, "-m", "circuitry.cli.main", "export-graph",
         str(circuit_path), "--format", "neuronpedia",
         "--slug", "smoke", "--scan", "tiny", "--out", str(np_out)],
        capture_output=True, text=True, check=True,
    )
    import json as _json
    data = _json.loads(np_out.read_text())
    assert data["metadata"]["slug"] == "smoke"
    assert len(data["links"]) == len(result.scores)
