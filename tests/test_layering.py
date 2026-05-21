"""Layering rules from docs/design.md §3. Belt-and-suspenders with the
import-linter config — the unit test catches violations at pytest time and
gives a clearer error than import-linter's CLI output."""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).parent.parent / "src" / "circuitry"

FORBIDDEN = {
    "core": ("circuitry.recorder", "circuitry.recipes", "circuitry.writers", "circuitry.cli"),
    "recipes": ("circuitry.cli",),
}

SIBLING_FORBIDDEN = ("mendu", "rl_recsys", "rl-recsys", "bumblebee", "plum", "bonsai", "gpt_2", "llm_council", "latent_superpowers")


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_core_does_not_import_higher_layers():
    for py in (SRC / "core").rglob("*.py"):
        for imp in _imports(py):
            for forbidden in FORBIDDEN["core"]:
                assert not imp.startswith(forbidden), (
                    f"core/{py.relative_to(SRC / 'core')} imports {imp}, "
                    f"violating §3 layering rule"
                )


def test_recipes_do_not_import_cli():
    for py in (SRC / "recipes").rglob("*.py"):
        for imp in _imports(py):
            for forbidden in FORBIDDEN["recipes"]:
                assert not imp.startswith(forbidden), (
                    f"recipes/{py.relative_to(SRC / 'recipes')} imports {imp}, "
                    f"violating §3 layering rule"
                )


def test_no_sibling_workspace_imports():
    for py in SRC.rglob("*.py"):
        for imp in _imports(py):
            root = imp.split(".", 1)[0]
            assert root not in SIBLING_FORBIDDEN, (
                f"{py.relative_to(SRC)} imports {imp} from sibling workspace project — "
                f"reverse-dependency rule (§3)"
            )
